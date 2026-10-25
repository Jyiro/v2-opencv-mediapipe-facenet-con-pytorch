"""Pruebas unitarias para los Bloques 3 y 4 (Event Sourcing, Portero HTTP, Sincronizador y Serie).

Verifica la inmutabilidad de eventos, derivación analítica de estado,
idempotencia del Portero con archivo JSONL, sincronización pull a SQLite,
truncado de protocolo serie a 16 caracteres y cámara simulada.
"""

from pathlib import Path
import time
import uuid
import pytest

from hardware.camara import CamaraSimulada, guardar_frame_jpeg, obtener_frame_jpeg
from hardware.serie import ClienteFichadorSerie, truncar_16
from nucleo.base_datos import RepositorioLIS
from nucleo.eventos import crear_evento, derivar_estado, validar_evento
from portero.servidor_portero import GestorEventosPortero, ServidorPortero
from servidor.sincronizador import SincronizadorServidor


def test_creacion_y_validacion_eventos():
    """Verifica el esquema oficial de eventos inmutables de la Sección 6."""
    ev = crear_evento(
        tipo="entrada",
        persona_id="juan_perez",
        metodo="cara+gesto",
        origen="pipeline-facial",
        distancia=0.48,
        meta={"lux": 320, "umbral": 0.8, "modelo": "vggface2"},
    )

    assert validar_evento(ev)
    assert ev["tipo"] == "entrada"
    assert ev["persona_id"] == "juan_perez"
    assert ev["distancia"] == 0.48
    assert ev["reloj_confiable"] is True
    assert ev["meta"]["lux"] == 320

    with pytest.raises(ValueError):
        crear_evento(tipo="tipo_inexistente_invalido")


def test_derivar_estado_completo():
    """Verifica reingresos múltiples, préstamos, avisos y correcciones."""
    eventos = [
        # 1. Entrada mañana
        crear_evento("entrada", persona_id="carlos", t="2026-10-14T08:00:00Z"),
        # 2. Retiro de equipo
        crear_evento("retiro", persona_id="carlos", objeto_id="multimetro-1", t="2026-10-14T08:30:00Z"),
        # 3. Salida mediodía
        crear_evento("salida", persona_id="carlos", t="2026-10-14T12:00:00Z"),
        # 4. Devolución de equipo
        crear_evento("devolucion", objeto_id="multimetro-1", t="2026-10-14T12:05:00Z"),
        # 5. Reingreso tarde (segundo ciclo del día)
        crear_evento("entrada", persona_id="carlos", t="2026-10-14T14:00:00Z"),
        # 6. Aviso creado
        crear_evento("aviso_creado", objeto_id="aviso-reunion", t="2026-10-14T14:10:00Z", meta={"titulo": "Reunión LIS 17hs"}),
        # 7. Modo del laboratorio
        crear_evento("estado_lab", t="2026-10-14T14:15:00Z", meta={"modo": "reunion"}),
        # 8. Sensor ambiente
        crear_evento("ambiente", t="2026-10-14T14:20:00Z", meta={"lux": 450, "temperatura": 24.5, "ruido": 55}),
    ]

    estado = derivar_estado(eventos)

    # Carlos está presente con 2 ingresos en el día
    assert "carlos" in estado["presentes"]
    assert estado["presentes"]["carlos"]["reingresos_hoy"] == 2

    # Multímetro fue devuelto -> no figura en préstamos activos
    assert "multimetro-1" not in estado["prestamos"]

    # Aviso activo
    assert "aviso-reunion" in estado["avisos"]
    assert estado["avisos"]["aviso-reunion"]["titulo"] == "Reunión LIS 17hs"

    # Ambiente y modo
    assert estado["estado_lab"] == "reunion"
    assert estado["ambiente"]["lux"] == 450
    assert estado["ambiente"]["temperatura"] == 24.5


def test_correccion_anula_evento_previo():
    """Verifica que un evento de corrección anula el evento referenciado."""
    ev_original = crear_evento("entrada", persona_id="persona_falsa", evento_id="ev_erroneo", t="2026-10-14T09:00:00Z")
    ev_correccion = crear_evento(
        "correccion",
        evento_id="ev_corr",
        t="2026-10-14T09:05:00Z",
        meta={"corrige_id": "ev_erroneo"},
    )

    estado = derivar_estado([ev_original, ev_correccion])
    assert "persona_falsa" not in estado["presentes"]


def test_portero_gestor_idempotencia_jsonl(tmp_path: Path):
    """Verifica que el Portero persista en JSONL y deduplique eventos idénticos."""
    ruta_jsonl = tmp_path / "test_portero.jsonl"
    gestor = GestorEventosPortero(ruta_jsonl=ruta_jsonl)

    ev1 = crear_evento("entrada", persona_id="ana", evento_id="uuid-1")
    ev2 = crear_evento("entrada", persona_id="pedro", evento_id="uuid-2")

    # Primer envío
    aceptados, duplicados = gestor.registrar_eventos([ev1, ev2])
    assert len(aceptados) == 2
    assert len(duplicados) == 0
    assert gestor.total_eventos() == 2

    # Reenvío de los mismos eventos (prueba de idempotencia)
    aceptados2, duplicados2 = gestor.registrar_eventos([ev1, ev2])
    assert len(aceptados2) == 0
    assert len(duplicados2) == 2
    assert gestor.total_eventos() == 2  # No se duplicaron en disco

    # Verificación de cursor pagination para el pull del servidor
    cursor1, evs1 = gestor.obtener_desde(0)
    assert cursor1 == 2
    assert len(evs1) == 2

    # Nuevo evento posterior
    ev3 = crear_evento("salida", persona_id="ana", evento_id="uuid-3")
    gestor.registrar_eventos([ev3])

    cursor2, evs2 = gestor.obtener_desde(cursor1)
    assert cursor2 == 3
    assert len(evs2) == 1
    assert evs2[0]["id"] == "uuid-3"


def test_sincronizador_pull_servidor(tmp_path: Path):
    """Verifica que el Servidor sincronice por pull desde el Portero hacia SQLite."""
    import portero.servidor_portero as sp
    from portero.servidor_portero import GestorEventosPortero

    gestor_aislado = GestorEventosPortero(ruta_jsonl=tmp_path / "test_portero_pull.jsonl")
    repo_aislado_portero = RepositorioLIS(ruta_db=tmp_path / "portero_db.db")

    gestor_orig = sp._gestor_portero
    repo_orig = sp.repositorio

    sp._gestor_portero = gestor_aislado
    sp.repositorio = repo_aislado_portero

    servidor_http = ServidorPortero(host="127.0.0.1", puerto=8099)
    servidor_http.iniciar()

    cursor_previo = gestor_aislado.cursor_actual()
    (tmp_path / "cursor.txt").write_text(str(cursor_previo), encoding="utf-8")

    repo = RepositorioLIS(
        ruta_db=tmp_path / "servidor_maestro.db",
        ruta_caras_csv=tmp_path / "caras.csv",
        ruta_asistencia_csv=tmp_path / "asistencia.csv",
        ruta_ingresos_csv=tmp_path / "registro_de_ingresos.csv",
    )
    sincronizador = SincronizadorServidor(
        repositorio=repo,
        url_portero="http://127.0.0.1:8099/events",
        ruta_cursor=tmp_path / "cursor.txt",
    )

    try:
        id_dinamico = f"ev_pull_{uuid.uuid4().hex[:8]}"
        ev = crear_evento("entrada", persona_id="lucas", evento_id=id_dinamico, distancia=0.39)
        gestor_aislado.registrar_eventos([ev])

        # Pull manual
        cant = sincronizador.sincronizar_ahora()
        assert cant == 1
        assert repo.esta_adentro("lucas")

        # Segundo pull no trae duplicados
        cant2 = sincronizador.sincronizar_ahora()
        assert cant2 == 0

        # Simular servidor apagado
        sincronizador.pausar()
        assert sincronizador.esta_pausado()
        cant_pausa = sincronizador.sincronizar_ahora()
        assert cant_pausa == 0

        sincronizador.reanudar()
        assert not sincronizador.esta_pausado()

    finally:
        servidor_http.detener()
        sp._gestor_portero = gestor_orig
        sp.repositorio = repo_orig


def test_protocolo_serie_truncado_y_virtual():
    """Verifica el truncado a 16 caracteres del LCD y el estado virtual."""
    cliente = ClienteFichadorSerie(simulado=True)

    # Texto largo de más de 16 caracteres
    texto_largo = "Nombre Muy Extenso De Alguien"
    assert len(truncar_16(texto_largo)) == 16
    assert truncar_16(texto_largo) == "Nombre Muy Exten"

    cliente.enviar_mensaje_lcd("Universidad Tecnologica Nacional", "Reconocimiento OK")
    cliente.set_color_lcd(0, 255, 0)
    cliente.set_led("verde", 1)
    cliente.beep("ok")

    estado = cliente.obtener_estado_virtual()
    assert estado["linea1"] == "Universidad Tecn"
    assert estado["linea2"] == "Reconocimiento O"
    assert estado["rgb"] == [0, 255, 0]
    assert estado["leds"]["verde"] == 1
    assert estado["ultimo_beep"] == "ok"


def test_camara_simulada_y_exportacion_jpeg():
    """Verifica que la cámara simulada y la exportación JPEG generen cuadros válidos."""
    cam = CamaraSimulada(ancho=320, alto=240)
    assert cam.isOpened()

    ret, cuadro = cam.read()
    assert ret is True
    assert cuadro is not None
    assert cuadro.shape == (240, 320, 3)

    # Probar exportación a JPEG
    guardar_frame_jpeg(cuadro, calidad=80)
    jpeg_bytes = obtener_frame_jpeg()
    assert jpeg_bytes is not None
    assert len(jpeg_bytes) > 100
    # Cabecera estándar JPEG SOI marker (0xFF, 0xD8)
    assert jpeg_bytes[:2] == b"\xff\xd8"
    cam.release()
