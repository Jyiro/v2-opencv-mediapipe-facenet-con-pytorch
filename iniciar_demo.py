from __future__ import annotations
import argparse
import logging
import socket
import threading
import webbrowser
from pathlib import Path
import mediapipe

_dir_mediapipe = Path(__file__).resolve().parent / "mediapipe"
if str(_dir_mediapipe) not in mediapipe.__path__:
    mediapipe.__path__.append(str(_dir_mediapipe))

from mediapipe.detector_de_caras import PipelineReconocimientoLIS
from portero.servidor_portero import iniciar_servidor


def obtener_ip_local() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanzador Integrado LIS (Cámara + Portero + Tablero)")
    parser.add_argument("--camara", default=None, help="Índice (0, 1) o URL de DroidCam (http://IP:4747/video)")
    parser.add_argument("--puerto", type=int, default=8080, help="Puerto HTTP del Portero y Tablero")
    parser.add_argument("--sin-ventana", action="store_true", help="Oculta cv.imshow y transmite solo al navegador web")
    args = parser.parse_args()

    servidor_http = iniciar_servidor(puerto=args.puerto)
    hilo_http = threading.Thread(target=servidor_http.serve_forever, daemon=True)
    hilo_http.start()

    ip_red = obtener_ip_local()
    print("\n" + "=" * 72)
    print(" PLATAFORMA LIS INICIADA CORRECTAMENTE")
    print(f" -> Tablero en esta PC:           http://localhost:{args.puerto}/tablero")
    print(f" -> Abrir desde Celular / Tablet: http://{ip_red}:{args.puerto}/tablero")
    print("=" * 72 + "\n")

    try:
        webbrowser.open(f"http://localhost:{args.puerto}/tablero")
    except Exception:
        pass

    fuente = args.camara
    if fuente is not None and str(fuente).isdigit():
        fuente = int(fuente)

    pipeline = PipelineReconocimientoLIS(
        fuente_camara=fuente,
        mostrar_ventana=not args.sin_ventana,
    )
    try:
        pipeline.ejecutar()
    finally:
        servidor_http.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
