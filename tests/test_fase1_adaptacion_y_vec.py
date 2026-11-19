"""Pruebas unitarias e integración para la Fase 1 (ADR 008: sqlite-vec, slots y telemetría shadow)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import urllib.request

import numpy as np
import pytest

from nucleo.adaptacion import (
    BufferPistaTopK,
    DetectorAlertaEntorno,
    evaluar_compuertas_slot,
)
from nucleo.base_datos import RepositorioLIS
from nucleo.comparacion import IndiceEmbeddingsRAM
from portero.servidor_portero import GestorEventosPortero, ManejadorPortero


def _vec_desde_e0(dist_l2: float, eje_ortogonal: int = 1, dim: int = 512) -> np.ndarray:
    """Construye un vector unitario exacto en S^(dim-1) a distancia L2 dada respecto de e0."""
    v = np.zeros(dim, dtype=np.float32)
    cos_t = 1.0 - (dist_l2 ** 2) / 2.0
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))
    v[0] = cos_t
    v[eje_ortogonal] = sin_t
    return v


def _repo_aislado(tmp_path: Path, nombre: str = "fase1.db") -> RepositorioLIS:
    carpeta = tmp_path / nombre.replace(".db", "")
    carpeta.mkdir(parents=True, exist_ok=True)
    return RepositorioLIS(
        ruta_db=carpeta / nombre,
        ruta_caras_csv=carpeta / "caras.csv",
        ruta_asistencia_csv=carpeta / "asistencia.csv",
        ruta_ingresos_csv=carpeta / "registro_de_ingresos.csv",
    )


def test_buffer_pista_topk_filtra_y_fusiona():
    reloj_val = [100.0]
    buf = BufferPistaTopK(capacidad=3, max_yaw=20.0, max_pitch=15.0, reloj=lambda: reloj_val[0])

    v1 = _vec_desde_e0(0.10, eje_ortogonal=1)
    v2 = _vec_desde_e0(0.20, eje_ortogonal=2)
    v3 = _vec_desde_e0(0.25, eje_ortogonal=3)
    v4 = _vec_desde_e0(0.05, eje_ortogonal=4)
    v_fuerte = _vec_desde_e0(0.02, eje_ortogonal=5)

    # Rechazos por calidad o pose fuera de rango
    assert not buf.registrar(v1, nitidez=250.0, brillo=0.5, calidad_ok=False, yaw=0.0, pitch=0.0)
    assert not buf.registrar(v1, nitidez=250.0, brillo=0.5, calidad_ok=True, yaw=25.0, pitch=0.0)
    assert not buf.registrar(v1, nitidez=250.0, brillo=0.5, calidad_ok=True, yaw=0.0, pitch=-18.0)
    assert buf.cantidad == 0

    # Inserta 5 válidas: debe retener solo Top-3 por puntaje
    assert buf.registrar(v1, nitidez=60.0, brillo=0.5, calidad_ok=True, yaw=12.0, pitch=5.0)
    assert buf.registrar(v2, nitidez=180.0, brillo=0.5, calidad_ok=True, yaw=2.0, pitch=1.0)
    assert buf.registrar(v3, nitidez=90.0, brillo=0.5, calidad_ok=True, yaw=8.0, pitch=4.0)
    assert buf.registrar(v4, nitidez=240.0, brillo=0.5, calidad_ok=True, yaw=1.0, pitch=1.0)
    assert buf.registrar(v_fuerte, nitidez=290.0, brillo=0.5, calidad_ok=True, yaw=0.0, pitch=0.0)

    assert buf.cantidad == 3
    mejor = buf.mejor_muestra()
    assert mejor is not None and mejor.nitidez == pytest.approx(290.0)
    assert np.allclose(mejor.embedding, v_fuerte, atol=1e-6)

    fusionado = buf.embedding_fusionado()
    assert fusionado is not None
    assert fusionado.shape == (512,)
    assert float(np.linalg.norm(fusionado)) == pytest.approx(1.0, abs=1e-5)

    buf.limpiar()
    assert buf.cantidad == 0
    assert buf.embedding_fusionado() is None


def test_evaluar_compuertas_slot_y_veto_inter_clase():
    ancla = _vec_desde_e0(0.0, eje_ortogonal=1)
    tercero_lejano = _vec_desde_e0(1.25, eje_ortogonal=2)
    candidato_bueno = _vec_desde_e0(0.55, eje_ortogonal=3)

    indice_seguro = IndiceEmbeddingsRAM({"Jonas": [ancla], "Ana": [tercero_lejano]})

    # Aprobación limpia dentro de banda 0.42..0.75 y lejos de Ana
    ok = evaluar_compuertas_slot(
        candidato_bueno,
        [ancla],
        indice_seguro,
        "Jonas",
        pistas_activas=1,
        liveness_ok=True,
        calidad_ok=True,
        yaw=4.0,
        pitch=-2.0,
    )
    assert ok["aprobado"] is True
    assert ok["decision_shadow"] == "promover_slot"
    assert ok["dist_ancla"] == pytest.approx(0.55, abs=1e-3)

    # Compuerta 1: múltiples pistas
    r1 = evaluar_compuertas_slot(
        candidato_bueno, [ancla], indice_seguro, "Jonas",
        pistas_activas=2, liveness_ok=True, calidad_ok=True,
    )
    assert not r1["aprobado"] and r1["decision_shadow"] == "multiples_pistas"

    # Compuerta 2: sin liveness
    r2 = evaluar_compuertas_slot(
        candidato_bueno, [ancla], indice_seguro, "Jonas",
        pistas_activas=1, liveness_ok=False, calidad_ok=True,
    )
    assert not r2["aprobado"] and r2["decision_shadow"] == "sin_liveness"

    # Compuerta 3: calidad o pose inválida
    r3 = evaluar_compuertas_slot(
        candidato_bueno, [ancla], indice_seguro, "Jonas",
        pistas_activas=1, liveness_ok=True, calidad_ok=False, yaw=30.0,
    )
    assert not r3["aprobado"] and r3["decision_shadow"] == "calidad_o_pose_insuficiente"

    # Compuerta 4a: redundante (<= 0.42)
    r4a = evaluar_compuertas_slot(
        _vec_desde_e0(0.30, eje_ortogonal=3), [ancla], indice_seguro, "Jonas",
        pistas_activas=1, liveness_ok=True, calidad_ok=True,
    )
    assert not r4a["aprobado"] and r4a["decision_shadow"] == "redundante_ya_cercano"

    # Compuerta 4b: fuera de banda (>= 0.75)
    r4b = evaluar_compuertas_slot(
        _vec_desde_e0(0.85, eje_ortogonal=3), [ancla], indice_seguro, "Jonas",
        pistas_activas=1, liveness_ok=True, calidad_ok=True,
    )
    assert not r4b["aprobado"] and r4b["decision_shadow"] == "fuera_banda_ancla"

    # Compuerta 5: veto inter-clase (tercero en mismo eje a ~0.58 del candidato)
    cand_e1 = _vec_desde_e0(0.55, eje_ortogonal=1)
    tercero_cercano = _vec_desde_e0(1.10, eje_ortogonal=1)
    indice_riesgoso = IndiceEmbeddingsRAM({"Jonas": [ancla], "Clon": [tercero_cercano]})
    r5 = evaluar_compuertas_slot(
        cand_e1, [ancla], indice_riesgoso, "Jonas",
        pistas_activas=1, liveness_ok=True, calidad_ok=True,
    )
    assert not r5["aprobado"] and r5["decision_shadow"] == "veto_inter_clase"
    assert r5["otro_mas_cercano"] == "Clon"


def test_detector_alerta_entorno_usuarios_y_ventana():
    ahora = [1000.0]
    det = DetectorAlertaEntorno(ventana_seg=900.0, min_usuarios_distintos=3, reloj=lambda: ahora[0])

    # 3 fallos del mismo usuario no disparan alerta
    assert not det.registrar_fallo("Jonas")
    assert not det.registrar_fallo("jonas")
    assert not det.registrar_fallo("JONAS")

    # Otro usuario dentro de ventana -> aún 2 únicos
    ahora[0] += 60.0
    assert not det.registrar_fallo("Ana")

    # Tercer usuario dentro de ventana -> alerta activa
    ahora[0] += 60.0
    assert det.registrar_fallo("Pedro")
    assert det.alerta_activa()

    # Avanzar más de 15 min -> purga fallos antiguos
    ahora[0] += 950.0
    assert not det.alerta_activa()


def test_repositorio_slots_ttl_lru_rollback_y_proyecto(tmp_path: Path):
    repo = _repo_aislado(tmp_path, "slots.db")
    assert repo.sqlite_vec_activo is True

    e0 = _vec_desde_e0(0.0, eje_ortogonal=1)
    anclas = [_vec_desde_e0(0.05 + 0.01 * i, eje_ortogonal=i + 1) for i in range(5)]

    repo.crear_o_actualizar_persona(nombre="Jonas", proyecto="LIS", pin="123456")
    repo.guardar_embeddings_persona("Jonas", anclas, modelo="vggface2", reemplazar=True, sincronizar_csv=False)

    # Perfil de otra comisión (Modo Aula)
    v_aula = _vec_desde_e0(1.20, eje_ortogonal=10)
    repo.crear_o_actualizar_persona(nombre="Lucia Aula", proyecto="Aula-101", pin="654321")
    repo.guardar_embeddings_persona("Lucia Aula", [v_aula], modelo="vggface2", reemplazar=True, sincronizar_csv=False)

    # Agrega 4 slots adaptativos con max_slots=3 -> reemplazo LRU conservando las 5 anclas
    slots = [_vec_desde_e0(0.50 + 0.03 * i, eje_ortogonal=20 + i) for i in range(4)]
    for s in slots:
        repo.agregar_slot_adaptativo("Jonas", s, modelo="vggface2", max_slots=3)

    assert len(repo.cargar_anclas_persona("Jonas")) == 5
    gal_completa = repo.cargar_diccionario_embeddings(modelo="vggface2", con_fallback=False)
    # 5 anclas + 3 slots adaptativos vigentes = 8
    assert len(gal_completa["Jonas"]) == 8

    # Slot ya expirado se ignora en lectura y se purga
    v_expirado = _vec_desde_e0(0.62, eje_ortogonal=30)
    repo.agregar_slot_adaptativo(
        "Jonas", v_expirado, modelo="vggface2", max_slots=3, expira_en="2000-01-01T00:00:00Z"
    )
    gal_vigente = repo.cargar_diccionario_embeddings(modelo="vggface2", con_fallback=False)
    assert len(gal_vigente["Jonas"]) == 7  # 5 anclas + 2 adaptativos no expirados (el 3ro venció)

    # Filtro contextual por proyecto (Modo Aula)
    gal_aula = repo.cargar_diccionario_embeddings(modelo="vggface2", con_fallback=False, proyecto="Aula-101")
    assert list(gal_aula.keys()) == ["Lucia Aula"]

    # Búsqueda KNN nativa con sqlite-vec debe empatar con IndiceEmbeddingsRAM
    indice_ram = IndiceEmbeddingsRAM(gal_vigente)
    res_ram = indice_ram.consultar(slots[-1], umbral=0.68, metodo="minimo")
    res_vec = repo.consultar_knn_sqlite_vec(slots[-1], k=3, modelo="vggface2")
    assert res_vec and res_vec[0]["nombre"] == res_ram["candidato_mas_cercano"] == "Jonas"
    assert res_vec[0]["distancia"] == pytest.approx(res_ram["distancia"], abs=1e-3)

    # Filtro por proyecto en sqlite-vec
    res_vec_aula = repo.consultar_knn_sqlite_vec(e0, k=3, modelo="vggface2", proyecto="Aula-101")
    assert len(res_vec_aula) == 1 and res_vec_aula[0]["nombre"] == "Lucia Aula"

    # Rollback en 1 paso: elimina todos los adaptativos (2 vigentes + 1 expirado) y deja las 5 anclas intactas
    eliminados = repo.revertir_a_anclas("Jonas")
    assert eliminados == 3
    gal_post_rb = repo.cargar_diccionario_embeddings(modelo="vggface2", con_fallback=False)
    assert len(gal_post_rb["Jonas"]) == 5
    assert len(repo.cargar_anclas_persona("Jonas")) == 5


def test_telemetria_shadow_y_endpoints_portero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import portero.servidor_portero as sp
    from http.server import ThreadingHTTPServer

    ruta_jsonl = tmp_path / "portero.jsonl"
    repo_temp = _repo_aislado(tmp_path, "portero_shadow.db")
    gestor_temp = GestorEventosPortero(ruta_jsonl=ruta_jsonl, repositorio_db=repo_temp)

    ancla = _vec_desde_e0(0.0, eje_ortogonal=1)
    slot = _vec_desde_e0(0.55, eje_ortogonal=2)
    repo_temp.crear_o_actualizar_persona(nombre="Jonas", proyecto="LIS", pin="789456")
    repo_temp.guardar_embeddings_persona("Jonas", [ancla], modelo="vggface2", reemplazar=True, sincronizar_csv=False)
    repo_temp.agregar_slot_adaptativo("Jonas", slot, modelo="vggface2", max_slots=3)

    monkeypatch.setattr(sp, "repositorio", repo_temp)
    monkeypatch.setattr(sp, "_gestor_portero", gestor_temp)
    monkeypatch.setattr(sp, "RUTA_JSONL", ruta_jsonl)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ManejadorPortero)
    puerto = int(httpd.server_address[1])
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()

    try:
        # 1. Fichaje por PIN debe generar 1 fila en telemetria_shadow sin crear embeddings
        req_pin = urllib.request.Request(
            f"http://127.0.0.1:{puerto}/auth/pin",
            data=json.dumps({"pin": "789456", "tipo": "entrada"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_pin, timeout=3) as resp:
            assert resp.status == 200

        with urllib.request.urlopen(f"http://127.0.0.1:{puerto}/telemetria/shadow", timeout=3) as resp:
            data_shadow = json.loads(resp.read().decode("utf-8"))
        assert data_shadow["sqlite_vec_activo"] is True
        assert len(data_shadow["registros"]) == 1
        assert data_shadow["registros"][0]["decision_shadow"] == "pin_contingencia"
        assert data_shadow["registros"][0]["persona_id"] == "Jonas"

        # 2. Rollback HTTP via POST /perfiles/revertir_anclas
        req_rb = urllib.request.Request(
            f"http://127.0.0.1:{puerto}/perfiles/revertir_anclas",
            data=json.dumps({"persona_id": "Jonas"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_rb, timeout=3) as resp:
            data_rb = json.loads(resp.read().decode("utf-8"))
        assert data_rb["ok"] is True
        assert data_rb["slots_eliminados"] == 1
        assert len(repo_temp.cargar_diccionario_embeddings()["Jonas"]) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
