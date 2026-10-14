"""Script lanzador todo-en-uno para el sistema LIS (UTN FRC).

Inicia de forma integrada y coordinada:
1. Servidor Portero HTTP en 0.0.0.0:8080 con Tablero Web y Streaming MJPEG.
2. Sincronizador por Pull del Servidor Maestro (tolerante a cortes).
3. Pipeline de Reconocimiento Facial y Gestos con Cámara Universal (Webcam, DroidCam IP o Simulada).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iniciar_demo")

CARPETA_RAIZ = Path(__file__).resolve().parent
if str(CARPETA_RAIZ) not in sys.path:
    sys.path.insert(0, str(CARPETA_RAIZ))

from hardware.camara import FuenteDeVideo, guardar_frame_jpeg, obtener_frame_jpeg
from hardware.serie import ClienteFichadorSerie
from nucleo.base_datos import RepositorioLIS
from portero.servidor_portero import ServidorPortero, configurar_callbacks_servidor, obtener_ip_local
from servidor.sincronizador import SincronizadorServidor


def main():
    parser = argparse.ArgumentParser(description="Lanzador todo-en-uno del Sistema LIS")
    parser.add_argument("--camara", type=str, default=None, help="Índice (0, 1), URL DroidCam (http://IP:4747/video) o 'simulada'")
    parser.add_argument("--sin-ventana", action="store_true", help="No abrir ventana local OpenCV (para usar exclusivamente el celular/tablet)")
    parser.add_argument("--puerto", type=int, default=8080, help="Puerto HTTP para el Tablero Web y Portero (por defecto 8080)")
    parser.add_argument("--serie", type=str, default=None, help="Puerto serie para el Fichador Arduino UNO")
    args = parser.parse_args()

    ip_local = obtener_ip_local()
    puerto = args.puerto

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cartel = (
        "\n" + "=" * 76 + "\n"
        "  SISTEMA DE ASISTENCIA Y FICHADOR LIS (UTN FRC) -- MODO DEMO EN VIVO\n"
        + "=" * 76 + "\n"
        f"  [PC] Tablero en esta PC:\n"
        f"       http://localhost:{puerto}/tablero\n\n"
        f"  [MOVIL] Abrir en tu Celular o Tablet (mismo Wi-Fi):\n"
        f"       http://{ip_local}:{puerto}/tablero\n\n"
        f"  [STREAM] Streaming de Video directo:\n"
        f"       http://{ip_local}:{puerto}/video_feed\n\n"
        f"  [DROIDCAM] Para usar tu celular como camara con DroidCam:\n"
        f"       1. Abri la app DroidCam en tu celular conectado al mismo Wi-Fi.\n"
        f"       2. Copia la URL de video (ej: http://{ip_local.rsplit('.', 1)[0]}.XX:4747/video).\n"
        f"       3. Pegala en el cuadro 'URL DroidCam' en la web o pasa --camara <URL>.\n"
        + "=" * 76 + "\n"
    )
    print(cartel, flush=True)

    # 1. Base de datos y sincronizador
    repo = RepositorioLIS()
    sincronizador = SincronizadorServidor(
        repositorio=repo,
        url_portero=f"http://127.0.0.1:{puerto}/events",
        intervalo_segundos=2.0,
    )
    sincronizador.iniciar()

    # 2. Servidor Portero HTTP
    portero = ServidorPortero(host="0.0.0.0", puerto=puerto)

    # 3. Importar y coordinar pipeline de mediapipe dinámicamente para no colisionar con el paquete mediapipe
    import importlib.util

    ruta_pipeline = CARPETA_RAIZ / "mediapipe" / "detector_de_caras.py"
    spec = importlib.util.spec_from_file_location("pipeline_detector_caras", ruta_pipeline)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo en {ruta_pipeline}")
    pipeline_mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_detector_caras"] = pipeline_mod
    spec.loader.exec_module(pipeline_mod)

    def callback_cambiar_camara(nueva_fuente: str) -> bool:
        if pipeline_mod.fuente_video_global:
            return pipeline_mod.fuente_video_global.cambiar_fuente(nueva_fuente)
        return False

    def callback_accion_hardware(accion: str, datos: dict) -> None:
        if accion == "touch":
            if pipeline_mod.cliente_fichador_global:
                pipeline_mod.cliente_fichador_global.simular_touch()
        elif accion == "gesto":
            gesto = datos.get("gesto")
            logger.info("Gesto virtual recibido: %s", gesto)
            if pipeline_mod.cliente_fichador_global:
                if gesto == "Open_Palm":
                    pipeline_mod.cliente_fichador_global.enviar_mensaje_lcd("Gesto Simulado", "Open_Palm Ingreso")
                    pipeline_mod.cliente_fichador_global.set_color_lcd(0, 255, 0)
                    pipeline_mod.cliente_fichador_global.beep("ok")
                elif gesto == "Thumb_Up":
                    pipeline_mod.cliente_fichador_global.enviar_mensaje_lcd("Gesto Simulado", "Thumb_Up Egreso")
                    pipeline_mod.cliente_fichador_global.set_color_lcd(0, 150, 255)
                    pipeline_mod.cliente_fichador_global.beep("ok")
        elif accion == "toggle_servidor":
            if datos.get("pausado"):
                sincronizador.pausar()
            else:
                sincronizador.reanudar()
        elif accion == "iniciar_alta":
            nombre = datos.get("nombre", "")
            if nombre and hasattr(pipeline_mod, "solicitar_alta_remota"):
                pipeline_mod.solicitar_alta_remota(nombre)

    def callback_estado_hardware() -> dict:
        if pipeline_mod.cliente_fichador_global:
            return pipeline_mod.cliente_fichador_global.obtener_estado_virtual()
        return {}

    configurar_callbacks_servidor(
        cambio_camara=callback_cambiar_camara,
        accion_hardware=callback_accion_hardware,
        estado_hardware=callback_estado_hardware,
        frame_jpeg=obtener_frame_jpeg,
    )

    portero.iniciar()

    # 4. Lanzar pipeline de video
    try:
        pipeline_mod.ejecutar_pipeline(
            fuente_camara=args.camara,
            mostrar_ventana=not args.sin_ventana,
            puerto_serie=args.serie,
        )
    except KeyboardInterrupt:
        logger.info("Deteniendo sistema...")
    finally:
        portero.detener()
        sincronizador.detener()
        logger.info("Sistema detenido correctamente.")


if __name__ == "__main__":
    main()
