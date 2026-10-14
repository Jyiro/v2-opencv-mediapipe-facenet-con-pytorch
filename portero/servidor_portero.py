"""Servicio del Portero HTTP (Galileo / Borde) y servidor de Tablero Web.

Desarrollado EXCLUSIVAMENTE con la biblioteca estándar de Python (Sección 9 y Regla 2):
http.server, json, urllib, os, socket, pathlib, uuid, threading, time.

Persiste en 'datos/portero_eventos.jsonl' (append-only con os.fsync) con deduplicación por id.
Sirve la interfaz web responsive (tablero/index.html) y el streaming MJPEG para celulares/tablets.
"""

from __future__ import annotations

from datetime import datetime, timezone
import http.server
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable
import urllib.parse
import uuid

logger = logging.getLogger(__name__)

CARPETA_PORTERO = Path(__file__).resolve().parent
CARPETA_RAIZ = CARPETA_PORTERO.parent
RUTA_TABLERO_HTML = CARPETA_RAIZ / "tablero" / "index.html"
RUTA_EVENTOS_JSONL = CARPETA_RAIZ / "datos" / "portero_eventos.jsonl"


def obtener_ip_local() -> str:
    """Detecta la IP local de la máquina en la red Wi-Fi o Ethernet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No requiere conexión real a internet
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class GestorEventosPortero:
    """Maneja el archivo append-only JSONL del Portero con deduplicación en memoria."""

    def __init__(self, ruta_jsonl: Path = RUTA_EVENTOS_JSONL) -> None:
        self.ruta_jsonl = ruta_jsonl
        self.ruta_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.ids_vistos: set[str] = set()
        self.eventos_memoria: list[dict[str, Any]] = []

        self._cargar_existentes()

    def _cargar_existentes(self) -> None:
        with self.lock:
            if not self.ruta_jsonl.exists():
                return
            try:
                with open(self.ruta_jsonl, "r", encoding="utf-8") as f:
                    for linea in f:
                        linea = linea.strip()
                        if linea:
                            try:
                                ev = json.loads(linea)
                                eid = ev.get("id")
                                if eid and eid not in self.ids_vistos:
                                    self.ids_vistos.add(eid)
                                    self.eventos_memoria.append(ev)
                            except Exception:
                                pass
                logger.info("Portero cargó %d eventos previos de %s", len(self.eventos_memoria), self.ruta_jsonl)
            except Exception:
                logger.exception("Error al leer archivo JSONL del Portero")

    def registrar_eventos(self, nuevos_eventos: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """Guarda eventos nuevos de forma idempotente con fsync.

        Devuelve: (lista_ids_aceptados, lista_ids_duplicados).
        """
        aceptados: list[str] = []
        duplicados: list[str] = []

        with self.lock:
            a_guardar: list[dict[str, Any]] = []
            for ev in nuevos_eventos:
                eid = ev.get("id") or str(uuid.uuid4())
                ev["id"] = eid
                if eid in self.ids_vistos:
                    duplicados.append(eid)
                else:
                    self.ids_vistos.add(eid)
                    self.eventos_memoria.append(ev)
                    a_guardar.append(ev)
                    aceptados.append(eid)

            if a_guardar:
                try:
                    with open(self.ruta_jsonl, "a", encoding="utf-8") as f:
                        for ev in a_guardar:
                            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    logger.exception("Error en fsync al persistir eventos en JSONL")

        return aceptados, duplicados

    def obtener_desde(self, cursor: int) -> tuple[int, list[dict[str, Any]]]:
        """Devuelve eventos a partir del índice cursor para el pull del servidor."""
        with self.lock:
            if cursor < 0:
                cursor = 0
            eventos_nuevos = self.eventos_memoria[cursor:]
            nuevo_cursor = len(self.eventos_memoria)
            return nuevo_cursor, eventos_nuevos

    def total_eventos(self) -> int:
        with self.lock:
            return len(self.eventos_memoria)

    def cursor_actual(self) -> int:
        with self.lock:
            return len(self.eventos_memoria)

    def todos_los_eventos(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.eventos_memoria)


# Instancia singleton del gestor para el servidor HTTP
_gestor_portero = GestorEventosPortero()
_callback_cambio_camara: Callable[[str], bool] | None = None
_callback_accion_hardware: Callable[[str, dict[str, Any]], None] | None = None
_proveedor_estado_hardware: Callable[[], dict[str, Any]] | None = None
_proveedor_frame_jpeg: Callable[[], bytes | None] | None = None


def configurar_callbacks_servidor(
    cambio_camara: Callable[[str], bool] | None = None,
    accion_hardware: Callable[[str, dict[str, Any]], None] | None = None,
    estado_hardware: Callable[[], dict[str, Any]] | None = None,
    frame_jpeg: Callable[[], bytes | None] | None = None,
) -> None:
    """Asocia las funciones del pipeline de OpenCV y serie con los endpoints HTTP."""
    global _callback_cambio_camara, _callback_accion_hardware, _proveedor_estado_hardware, _proveedor_frame_jpeg
    _callback_cambio_camara = cambio_camara
    _callback_accion_hardware = accion_hardware
    _proveedor_estado_hardware = estado_hardware
    _proveedor_frame_jpeg = frame_jpeg


class ManejadorPortero(http.server.BaseHTTPRequestHandler):
    """Manejador REST y streaming MJPEG compatible con tablets y celulares."""

    server_version = "PorteroGalileo/0.4.0"

    def log_message(self, format: str, *args: Any) -> None:
        try:
            msg = format % args
            if "/state" in msg or "/video_feed" in msg or "favicon.ico" in msg:
                return
            logger.debug("%s - %s", self.address_string(), msg)
        except Exception:
            pass

    def _responder_json(self, datos: Any, codigo: int = 200) -> None:
        payload = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        url_parseada = urllib.parse.urlparse(self.path)
        ruta = url_parseada.path
        params = urllib.parse.parse_qs(url_parseada.query)

        # 1. Tablero Web principal (HTML para celular o tablet)
        if ruta in ("/", "/tablero", "/index.html"):
            if RUTA_TABLERO_HTML.exists():
                contenido = RUTA_TABLERO_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(contenido)))
                self.end_headers()
                self.wfile.write(contenido)
            else:
                self.send_error(404, "tablero/index.html no encontrado")
            return

        # 2. Streaming de video MJPEG (/video_feed)
        if ruta == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            while True:
                frame_bytes = None
                if _proveedor_frame_jpeg:
                    frame_bytes = _proveedor_frame_jpeg()

                if frame_bytes:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.033)  # ~30 FPS
            return

        # 3. GET /health (Sección 9)
        if ruta == "/health":
            self._responder_json(
                {
                    "estado": "ok",
                    "hora": datetime.now(timezone.utc).isoformat(),
                    "version": "0.4.0",
                    "total_eventos": _gestor_portero.total_eventos(),
                    "reloj_confiable": True,
                }
            )
            return

        # 4. GET /events?since=<cursor> (Sección 9 para Pull del Servidor)
        if ruta == "/events":
            cursor_str = params.get("since", ["0"])[0]
            try:
                cursor = int(cursor_str)
            except ValueError:
                cursor = 0
            nuevo_cursor, eventos = _gestor_portero.obtener_desde(cursor)
            self._responder_json({"cursor": nuevo_cursor, "events": eventos})
            return

        # 5. GET /state (Sección 9: estado derivado + hardware)
        if ruta == "/state":
            todos = _gestor_portero.todos_los_eventos()
            # Derivar estado analítico
            from nucleo.eventos import derivar_estado

            estado = derivar_estado(todos)

            # Agregar estado virtual de hardware si está disponible
            if _proveedor_estado_hardware:
                estado["fichador"] = _proveedor_estado_hardware()
            else:
                estado["fichador"] = {
                    "linea1": "LIS Control",
                    "linea2": "Listo",
                    "rgb": [0, 100, 255],
                    "leds": {"azul": 0, "verde": 0, "rojo": 0},
                    "simulado": True,
                }

            self._responder_json(estado)
            return

        self.send_error(404, "Ruta no encontrada")

    def do_POST(self) -> None:
        url_parseada = urllib.parse.urlparse(self.path)
        ruta = url_parseada.path

        longitud = int(self.headers.get("Content-Length", 0))
        cuerpo = self.rfile.read(longitud).decode("utf-8") if longitud > 0 else "{}"
        try:
            datos = json.loads(cuerpo)
        except Exception:
            datos = {}

        # 1. POST /events (Sección 9: idempotente)
        if ruta == "/events":
            lista = datos if isinstance(datos, list) else datos.get("events", [datos])
            aceptados, duplicados = _gestor_portero.registrar_eventos(lista)
            self._responder_json({"aceptados": aceptados, "duplicados": duplicados}, 201)
            return

        # 2. POST /camara/configurar (Cambio de cámara en caliente: USB o DroidCam)
        if ruta == "/camara/configurar":
            fuente = datos.get("fuente")
            exito = False
            if fuente is not None and _callback_cambio_camara:
                exito = _callback_cambio_camara(str(fuente))
            self._responder_json({"ok": exito, "fuente": fuente})
            return

        # 3. POST /hardware/accion (Acción simulada: TOUCH, botón, torre, pin)
        if ruta == "/hardware/accion":
            accion = datos.get("accion")
            params = datos.get("datos", {})
            if accion and _callback_accion_hardware:
                _callback_accion_hardware(accion, params)
            self._responder_json({"ok": True, "accion": accion})
            return

        self.send_error(404, "Ruta no encontrada")


class ServidorPortero:
    """Clase controladora para levantar el servidor en segundo plano."""

    def __init__(self, host: str = "0.0.0.0", puerto: int = 8080) -> None:
        self.host = host
        self.puerto = puerto
        self.httpd: http.server.ThreadingHTTPServer | None = None
        self.hilo: threading.Thread | None = None

    def iniciar(self) -> None:
        self.httpd = http.server.ThreadingHTTPServer((self.host, self.puerto), ManejadorPortero)
        self.hilo = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="ServidorPorteroHTTP")
        self.hilo.start()
        ip_local = obtener_ip_local()
        logger.info("============================================================")
        logger.info("PORTERO HTTP ACTIVO en http://%s:%d", self.host, self.puerto)
        logger.info("-> Tablero en PC: http://localhost:%d/tablero", self.puerto)
        logger.info("-> Abrir en Celular/Tablet: http://%s:%d/tablero", ip_local, self.puerto)
        logger.info("============================================================")

    def detener(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logger.info("Servidor Portero detenido.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    servidor = ServidorPortero()
    servidor.iniciar()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        servidor.detener()
