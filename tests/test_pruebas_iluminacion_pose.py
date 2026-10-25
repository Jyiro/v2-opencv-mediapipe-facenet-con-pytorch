from __future__ import annotations

import numpy as np

from nucleo.base_datos import RepositorioLIS
from nucleo.iluminacion import (
    MODELO_CRUDO,
    MODELO_NORMALIZADO,
    ConfigIluminacion,
    balance_gray_world,
    estadisticas_luz,
    evaluar_calidad,
    gamma_para_objetivo,
    normalizar_iluminacion,
)
from nucleo.pose import EstimadorPose, GuiaDePoses, PoseCabeza, POSES_3, POSES_5, pasos_para
from nucleo.pruebas import CarreraPista, SesionPruebas


# ---------- iluminación ----------

def _imagen_ruidosa(valor: int, tam: int = 120) -> np.ndarray:
    rng = np.random.default_rng(7)
    ruido = rng.integers(-25, 26, size=(tam, tam, 3))
    return np.clip(valor + ruido, 0, 255).astype(np.uint8)


def test_normalizar_aclara_imagen_oscura_y_oscurece_la_quemada() -> None:
    oscura = _imagen_ruidosa(40)
    quemada = _imagen_ruidosa(225)
    b_osc = estadisticas_luz(oscura)["brillo"]
    b_que = estadisticas_luz(quemada)["brillo"]
    n_osc = estadisticas_luz(normalizar_iluminacion(oscura))["brillo"]
    n_que = estadisticas_luz(normalizar_iluminacion(quemada))["brillo"]
    assert n_osc > b_osc + 0.1
    assert n_que < b_que - 0.05  # gamma acotado: una cara quemada se corrige sin destruir detalle
    assert abs(n_osc - n_que) < abs(b_osc - b_que)  # la luz se vuelve más parecida


def test_normalizar_no_modifica_la_entrada_y_conserva_forma() -> None:
    img = _imagen_ruidosa(90)
    copia = img.copy()
    salida = normalizar_iluminacion(img, ConfigIluminacion())
    assert salida.shape == img.shape and salida.dtype == np.uint8
    assert np.array_equal(img, copia)


def test_balance_gray_world_reduce_el_tinte_azul() -> None:
    img = _imagen_ruidosa(110)
    img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 70, 0, 255)  # canal B alto
    medias_antes = img.reshape(-1, 3).mean(axis=0)
    medias_despues = balance_gray_world(img).reshape(-1, 3).mean(axis=0)
    assert np.ptp(medias_despues) < np.ptp(medias_antes) / 3


def test_gamma_para_objetivo_aclara_si_oscuro_y_oscurece_si_claro() -> None:
    assert gamma_para_objetivo(0.2, 0.5, 0.45, 2.2) < 1.0
    assert gamma_para_objetivo(0.8, 0.5, 0.45, 2.2) > 1.0
    assert 0.45 <= gamma_para_objetivo(0.001, 0.5, 0.45, 2.2) <= 2.2


def test_config_iluminacion_desde_dict_ignora_claves_desconocidas() -> None:
    cfg = ConfigIluminacion.desde_dict({"clahe": False, "brillo_objetivo": "0.6", "otra": 1})
    assert cfg.clahe is False and cfg.brillo_objetivo == 0.6 and cfg.balance_blancos is True


def test_calidad_detecta_borroso_oscuro_y_chico() -> None:
    assert evaluar_calidad(_imagen_ruidosa(120))["ok"] is True
    plano = np.full((120, 120, 3), 120, dtype=np.uint8)
    assert "borrosa" in evaluar_calidad(plano)["motivo"]
    assert "poca luz" in evaluar_calidad(_imagen_ruidosa(8))["motivo"]
    assert "Acercate" in evaluar_calidad(_imagen_ruidosa(120, tam=40))["motivo"]
    assert evaluar_calidad(None)["ok"] is False


# ---------- pose ----------

class _Pt:
    def __init__(self, x: float = 0.5, y: float = 0.5) -> None:
        self.x, self.y = x, y


def _cara(nariz_x: float = 0.5, nariz_y: float = 0.6) -> list[_Pt]:
    pts = [_Pt() for _ in range(478)]
    pts[234] = _Pt(0.3, 0.5)    # borde izquierdo en imagen
    pts[454] = _Pt(0.7, 0.5)    # borde derecho en imagen
    pts[10] = _Pt(0.5, 0.2)     # frente
    pts[152] = _Pt(0.5, 0.8)    # mentón
    pts[1] = _Pt(nariz_x, nariz_y)
    return pts


def test_pose_frontal_es_cercana_a_cero() -> None:
    pose = EstimadorPose(espejo=True).estimar(_cara(0.5, 0.584))
    assert pose is not None and abs(pose.yaw) < 1 and abs(pose.pitch) < 1


def test_yaw_positivo_hacia_la_derecha_del_usuario_segun_espejo() -> None:
    cara = _cara(nariz_x=0.62)
    assert EstimadorPose(espejo=True).estimar(cara).yaw > 15
    assert EstimadorPose(espejo=False).estimar(cara).yaw < -15


def test_pitch_positivo_al_levantar_el_menton_y_referencia_propia() -> None:
    est = EstimadorPose()
    assert est.estimar(_cara(nariz_y=0.5)).pitch > 5     # nariz más arriba -> levanta
    assert est.estimar(_cara(nariz_y=0.7)).pitch < -5    # nariz más abajo -> baja
    est.fijar_referencia_frontal(_cara(nariz_y=0.61))
    assert abs(est.estimar(_cara(nariz_y=0.61)).pitch) < 1


def test_pose_con_datos_invalidos_devuelve_none() -> None:
    assert EstimadorPose().estimar(None) is None
    assert EstimadorPose().estimar([_Pt()] * 10) is None


# ---------- guía de poses ----------

class _Reloj:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_guia_captura_tras_sostener_la_pose_y_avanza() -> None:
    reloj = _Reloj()
    guia = GuiaDePoses(POSES_3, sostener_seg=0.5, reloj=reloj)
    frente = PoseCabeza(0.0, 0.0)

    r = guia.actualizar(frente)
    assert r.clave == "frente" and r.coincide and not r.capturar and 0.0 <= r.progreso_sostener < 1.0
    reloj.t += 0.3
    assert not guia.actualizar(frente).capturar
    reloj.t += 0.3
    r = guia.actualizar(frente)
    assert r.capturar and r.clave == "derecha" and r.hechas == ["frente"]


def test_guia_reinicia_el_conteo_si_se_pierde_la_pose_o_la_calidad() -> None:
    reloj = _Reloj()
    guia = GuiaDePoses(POSES_3, sostener_seg=0.5, reloj=reloj)
    guia.actualizar(PoseCabeza(0, 0))
    reloj.t += 0.4
    assert guia.actualizar(None).progreso_sostener == 0.0
    reloj.t += 0.4
    r = guia.actualizar(PoseCabeza(0, 0), calidad_ok=False, motivo_calidad="Hay muy poca luz.")
    assert not r.capturar and "poca luz" in r.mensaje
    reloj.t += 0.4
    assert not guia.actualizar(PoseCabeza(0, 0)).capturar  # volvió a empezar a contar


def test_guia_da_pistas_de_movimiento_y_completa_las_tres_poses() -> None:
    reloj = _Reloj()
    guia = GuiaDePoses(POSES_3, sostener_seg=0.2, reloj=reloj)
    for pose in (PoseCabeza(0, 0), PoseCabeza(25, 0), PoseCabeza(-25, 0)):
        guia.actualizar(pose)
        reloj.t += 0.25
        r = guia.actualizar(pose)
        assert r.capturar
    assert r.completado and guia.completado and guia.hechas == ["frente", "derecha", "izquierda"]

    guia2 = GuiaDePoses(POSES_3, sostener_seg=0.2, reloj=reloj)
    guia2.indice = 1  # pide derecha
    assert "derecha" in guia2.actualizar(PoseCabeza(5, 0)).mensaje


def test_pasos_para_elige_3_o_5_poses() -> None:
    assert pasos_para(3) is POSES_3 and pasos_para(5) is POSES_5 and len(POSES_5) == 5
    # Guiñada acotada a 28 grados para evitar pérdida de rasgos binoculares
    assert POSES_5[1].yaw == (10.0, 28.0)
    assert POSES_5[2].yaw == (-28.0, -10.0)


# ---------- estadísticas del modo de pruebas ----------

def test_carrera_primero_gana_quien_acierta_antes() -> None:
    c = CarreraPista()
    assert c.actualizar(False, False) is None
    assert c.actualizar(False, True) == "filtrado"
    assert c.actualizar(True, True) is None  # ya hay ganador
    assert c.pasos_hasta_acierto == {"crudo": 3, "filtrado": 2}

    e = CarreraPista()
    assert e.actualizar(True, True) == "empate"


def test_sesion_cuenta_aciertos_falsas_aceptaciones_y_rechazos_con_sujeto() -> None:
    s = SesionPruebas(sujeto="Lucas")
    ok = {"reconocido": True, "nombre": "lucas", "distancia": 0.4}
    otro = {"reconocido": True, "nombre": "pedro", "distancia": 0.7}
    no = {"reconocido": False, "nombre": "Desconocido", "distancia": 0.95}
    s.registrar(ok, otro)
    s.registrar(no, ok)
    d = s.a_dict()
    assert d["crudo"]["aciertos"] == 1 and d["crudo"]["falsos_rechazos"] == 1
    assert d["filtrado"]["falsas_aceptaciones"] == 1 and d["filtrado"]["aciertos"] == 1
    assert d["crudo"]["distancia_media"] == 0.4
    assert s.es_acierto(ok) and not s.es_acierto(otro) and not s.es_acierto(no)


def test_sesion_sin_sujeto_solo_cuenta_reconocimientos() -> None:
    s = SesionPruebas()
    s.registrar({"reconocido": True, "nombre": "x", "distancia": 0.5}, {"reconocido": False, "nombre": "?", "distancia": 0.9})
    d = s.a_dict()
    assert d["crudo"]["reconocidas"] == 1 and d["filtrado"]["reconocidas"] == 0
    assert s.es_acierto({"reconocido": True, "nombre": "cualquiera"})
    s.registrar_victoria("filtrado")
    assert s.a_dict()["victorias"]["filtrado"] == 1


# ---------- persistencia por variante ----------

def test_guardar_ambas_variantes_no_se_pisan_y_se_cargan_por_separado(tmp_path) -> None:
    repo = RepositorioLIS(
        ruta_db=tmp_path / "t.db",
        ruta_caras_csv=tmp_path / "caras.csv",
        ruta_asistencia_csv=tmp_path / "asis.csv",
        ruta_ingresos_csv=tmp_path / "ing.csv",
    )
    crudo = [np.ones(512, dtype=np.float32), np.full(512, 2.0, dtype=np.float32)]
    filtr = [np.full(512, 3.0, dtype=np.float32)]
    repo.guardar_embeddings_persona("Ana", crudo, modelo=MODELO_CRUDO, sincronizar_csv=False)
    repo.guardar_embeddings_persona("Ana", filtr, modelo=MODELO_NORMALIZADO, sincronizar_csv=False)

    assert len(repo.cargar_diccionario_embeddings(modelo=MODELO_CRUDO)["Ana"]) == 2
    assert len(repo.cargar_diccionario_embeddings(modelo=MODELO_NORMALIZADO)["Ana"]) == 1
    assert repo.obtener_persona("Ana")["cantidad_embeddings"] == 2

    repo.guardar_embeddings_persona("Ana", crudo[:1], modelo=MODELO_CRUDO, sincronizar_csv=False)
    assert len(repo.cargar_diccionario_embeddings(modelo=MODELO_NORMALIZADO)["Ana"]) == 1  # intacta


def test_orden_de_enrolamiento_guiado(tmp_path) -> None:
    repo = RepositorioLIS(
        ruta_db=tmp_path / "t.db", ruta_caras_csv=tmp_path / "c.csv",
        ruta_asistencia_csv=tmp_path / "a.csv", ruta_ingresos_csv=tmp_path / "i.csv",
    )
    orden = repo.solicitar_enrolamiento_camara("Ana", guiado=True, poses=3, modo="pruebas")
    assert orden["guiado"] and orden["muestras_totales"] == 3 and orden["modo"] == "pruebas"
    clasico = repo.solicitar_enrolamiento_camara("Ana", muestras=5)
    assert clasico["guiado"] is False and clasico["muestras_totales"] == 5


def test_asimetria_lateral_detecta_luz_de_costado() -> None:
    img_plana = np.full((120, 120, 3), 120, dtype=np.uint8)
    st_plana = estadisticas_luz(img_plana)
    assert st_plana["asimetria_lateral"] == 0.0

    # Lado izquierdo oscuro, lado derecho brillante
    img_costado = np.zeros((120, 120, 3), dtype=np.uint8)
    img_costado[:, :60] = 40
    img_costado[:, 60:] = 200
    st_costado = estadisticas_luz(img_costado)
    assert st_costado["asimetria_lateral"] > 0.5


def test_batch_generar_par_embeddings_mantiene_norma_unitaria() -> None:
    from nucleo.reconocimiento import ExtractorFaceNetRapido

    extractor = ExtractorFaceNetRapido()
    img_a = _imagen_ruidosa(100, tam=160)
    img_b = _imagen_ruidosa(180, tam=160)

    emb_a, emb_b = extractor.generar_par_embeddings(img_a, img_b)
    assert emb_a is not None and emb_b is not None
    assert emb_a.shape == (512,) and emb_b.shape == (512,)
    assert abs(np.linalg.norm(emb_a) - 1.0) < 1e-4
    assert abs(np.linalg.norm(emb_b) - 1.0) < 1e-4


def test_endpoints_pruebas_y_alta_guiada_servidor_portero(tmp_path: Path) -> None:
    import json
    import threading
    import urllib.request
    from hardware.camara import guardar_preview_pruebas_jpeg
    import portero.servidor_portero as sp
    from portero.servidor_portero import GestorEventosPortero, iniciar_servidor

    repo_tmp = RepositorioLIS(
        ruta_db=tmp_path / "temp_test.db",
        ruta_caras_csv=tmp_path / "caras.csv",
        ruta_asistencia_csv=tmp_path / "asistencia.csv",
        ruta_ingresos_csv=tmp_path / "ingresos.csv",
    )
    gestor_tmp = GestorEventosPortero(ruta_jsonl=tmp_path / "temp_events.jsonl")

    repo_orig = sp.repositorio
    gestor_orig = sp._gestor_portero
    sp.repositorio = repo_tmp
    sp._gestor_portero = gestor_tmp

    puerto = 8094
    srv = iniciar_servidor(puerto=puerto)
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()

    try:
        url_base = f"http://127.0.0.1:{puerto}"

        # 1. Configurar modo pruebas
        req = urllib.request.Request(
            f"{url_base}/pruebas/configurar",
            data=json.dumps({"activo": True, "sujeto": "Lucas", "reiniciar": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["ok"] is True and res["config"]["activo"] is True
            assert res["config"]["sujeto"] == "Lucas"

        # 2. Consultar estado de pruebas
        with urllib.request.urlopen(f"{url_base}/pruebas/estado") as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert "config" in res and res["config"]["sujeto"] == "Lucas"

        # 3. Consultar /state con campos de pruebas
        with urllib.request.urlopen(f"{url_base}/state") as resp:
            assert resp.status == 200
            st = json.loads(resp.read().decode("utf-8"))
            assert "pruebas" in st and "pruebas_config" in st

        # 4. Preview en memoria
        guardar_preview_pruebas_jpeg(b"\xff\xd8\xff\xe0\x00\x10JFIFtest")
        with urllib.request.urlopen(f"{url_base}/pruebas/preview.jpg") as resp:
            assert resp.status == 200
            data = resp.read()
            assert len(data) > 0 and data[:2] == b"\xff\xd8"

        # 5. Alta guiada
        req_alta = urllib.request.Request(
            f"{url_base}/perfiles/alta_guiada",
            data=json.dumps({"nombre": "Matias", "consentimiento": True, "poses": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_alta) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["ok"] is True and res["orden_enrolamiento"]["guiado"] is True

    finally:
        srv.shutdown()
        srv.server_close()
        sp.repositorio = repo_orig
        sp._gestor_portero = gestor_orig


def test_extractor_resoluciones_variables_y_redimension_inteligente() -> None:
    from nucleo.reconocimiento import ExtractorFaceNetRapido, redimensionar_recorte

    # 1. Redimensión inteligente: adaptación garantizada a 160x160 para InceptionResnetV1
    chico = np.zeros((50, 50, 3), dtype=np.uint8)
    assert redimensionar_recorte(chico, 160).shape == (160, 160, 3)
    grande = np.zeros((200, 200, 3), dtype=np.uint8)
    assert redimensionar_recorte(grande, 112).shape == (160, 160, 3)

    # 2. Extractor genera vectores unitarios de 512-D en todas las resoluciones
    ext = ExtractorFaceNetRapido(resolucion=160)
    img = _imagen_ruidosa(120, tam=90)
    for res in (112, 128, 160, 224):
        emb = ext.generar_embedding(img, resolucion=res)
        assert emb is not None and emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

