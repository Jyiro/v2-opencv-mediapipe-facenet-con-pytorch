"""Módulo de captura y fuente de video universal para el sistema LIS.

Soporta:
1. Webcams locales y drivers virtuales (0, 1, 2) vía DirectShow/V4L2.
2. Streams de video IP y DroidCam (HTTP/RTSP) con hilo dedicado de baja latencia (buffer=1).
3. Cámara simulada para pruebas sin hardware (canvas sintético interactivo).
4. Exportación en memoria del último cuadro procesado en JPEG para streaming MJPEG web.
5. Normalización inteligente de URLs de DroidCam e IPs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import threading
import time
from typing import Any

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)

_frame_jpeg_lock = threading.Lock()
_ultimo_frame_jpeg: bytes | None = None

_frame_subido_lock = threading.Lock()
_ultimo_frame_subido: bytes | None = None


def guardar_frame_jpeg(cuadro_bgr: np.ndarray, calidad: int = 75) -> None:
    """Codifica y guarda en memoria el último cuadro procesado para el stream web."""
    global _ultimo_frame_jpeg
    try:
        if cuadro_bgr is None or cuadro_bgr.size == 0:
            return
        parametros = [int(cv.IMWRITE_JPEG_QUALITY), calidad]
        exito, buffer = cv.imencode(".jpg", cuadro_bgr, parametros)
        if exito:
            with _frame_jpeg_lock:
                _ultimo_frame_jpeg = buffer.tobytes()
    except Exception:
        logger.exception("Error al codificar cuadro JPEG para streaming")


def obtener_frame_jpeg() -> bytes | None:
    """Devuelve los bytes JPEG del último cuadro disponible para clientes web."""
    with _frame_jpeg_lock:
        return _ultimo_frame_jpeg


def guardar_frame_subido_jpeg(datos_jpeg: bytes) -> None:
    """Almacena en memoria el cuadro JPEG enviado por un cliente web (celular/tablet)."""
    global _ultimo_frame_subido
    if not datos_jpeg:
        return
    with _frame_subido_lock:
        _ultimo_frame_subido = datos_jpeg


def obtener_frame_subido_jpeg() -> bytes | None:
    """Obtiene los bytes JPEG del último cuadro subido desde el navegador."""
    with _frame_subido_lock:
        return _ultimo_frame_subido


def guardar_frame_disco_seguro(datos_jpeg: bytes, ruta_destino: Path) -> None:
    """Escribe a disco de manera segura en Windows sin fallar por bloqueos concurrentes de lectura."""
    if not datos_jpeg:
        return
    try:
        ruta_destino.parent.mkdir(parents=True, exist_ok=True)
        try:
            ruta_destino.write_bytes(datos_jpeg)
        except (PermissionError, OSError):
            tmp_path = ruta_destino.with_suffix(".tmp")
            try:
                tmp_path.write_bytes(datos_jpeg)
                tmp_path.replace(ruta_destino)
            except (PermissionError, OSError):
                pass
    except Exception:
        pass


def normalizar_origen_camara(origen: int | str) -> int | str:
    """Normaliza automáticamente cadenas de IP, DroidCam y puertos."""
    if isinstance(origen, int):
        return origen

    texto = str(origen).strip()
    if texto.isdigit():
        return int(texto)

    if texto.lower() in ("simulada", "simulado", "mock", "virtual"):
        return "simulada"

    # Si parece una IP o URL con puerto (ej. 192.168.100.18:4747)
    if ":" in texto or "." in texto:
        if not (texto.startswith("http://") or texto.startswith("https://") or texto.startswith("rtsp://")):
            texto = "http://" + texto
        # Si es DroidCam (puerto típico 4747) y no tiene endpoint de video
        if ":4747" in texto and not texto.endswith("/video") and not texto.endswith("/mjpegfeed"):
            texto = texto.rstrip("/") + "/video"
        return texto

    return 0


class CamaraSimulada:
    """Generador sintético de video a 30 FPS para desarrollo y pruebas sin cámara física."""

    def __init__(self, ancho: int = 640, alto: int = 480, mensaje_extra: str = "") -> None:
        self.ancho = ancho
        self.alto = alto
        self.mensaje_extra = mensaje_extra
        self.contador = 0
        self.abierta = True
        self.ultimo_tiempo = time.perf_counter()

    def isOpened(self) -> bool:
        return self.abierta

    def read(self) -> tuple[bool, np.ndarray]:
        if not self.abierta:
            return False, np.zeros((self.alto, self.ancho, 3), dtype=np.uint8)

        ahora = time.perf_counter()
        delta = ahora - self.ultimo_tiempo
        if delta < 0.033:
            time.sleep(0.033 - delta)
        self.ultimo_tiempo = time.perf_counter()

        self.contador += 1
        cuadro = np.zeros((self.alto, self.ancho, 3), dtype=np.uint8)

        # Fondo con gradiente
        for y in range(self.alto):
            valor = int(25 + 30 * (y / self.alto))
            cuadro[y, :] = (valor, valor, valor + 10)

        # Rejilla suave
        for x in range(0, self.ancho, 40):
            cv.line(cuadro, (x, 0), (x, self.alto), (40, 45, 50), 1)
        for y in range(0, self.alto, 40):
            cv.line(cuadro, (0, y), (self.ancho, y), (40, 45, 50), 1)

        # Cara sintética que se mueve suavemente
        cx = int(self.ancho / 2 + 50 * np.sin(self.contador * 0.04))
        cy = int(self.alto / 2 + 25 * np.cos(self.contador * 0.03))
        w_cara, h_cara = 140, 170

        cv.ellipse(cuadro, (cx, cy), (w_cara // 2, h_cara // 2), 0, 0, 360, (180, 205, 235), -1)
        cv.circle(cuadro, (cx - 25, cy - 20), 8, (60, 40, 40), -1)
        cv.circle(cuadro, (cx + 25, cy - 20), 8, (60, 40, 40), -1)
        cv.ellipse(cuadro, (cx, cy + 30), (20, 10), 0, 0, 180, (50, 50, 180), 2)

        titulo = self.mensaje_extra or "MODO SIMULADO (Sin Camara Fisica)"
        cv.putText(cuadro, titulo, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.putText(cuadro, f"Frame: {self.contador} | Teclas: [I]ngreso [E]greso [T]ouch", (15, self.alto - 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return True, cuadro

    def release(self) -> None:
        self.abierta = False


class LectorStreamIP(threading.Thread):
    """Hilo de lectura continua para streams HTTP/RTSP (DroidCam/IP Webcam) con buffer=1."""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True, name="LectorStreamIP")
        self.url = url
        self.activo = True
        self.cap: cv.VideoCapture | None = None
        self.ultimo_frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.conectado = False
        self.intentos = 0

    def run(self) -> None:
        while self.activo:
            if self.cap is None or not self.cap.isOpened():
                self.intentos += 1
                logger.info("Conectando a stream IP (%s)... (intento %d)", self.url, self.intentos)
                self.cap = cv.VideoCapture(self.url)
                if not self.cap.isOpened():
                    time.sleep(2.0)
                    continue
                logger.info("Conectado exitosamente al stream IP: %s", self.url)
                self.conectado = True
                self.intentos = 0

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ultimo_frame = frame
                self.conectado = True
            else:
                self.conectado = False
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1.0)

    def leer(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if self.ultimo_frame is not None:
                return True, self.ultimo_frame.copy()
            return False, None

    def release(self) -> None:
        self.activo = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


class FuenteDeVideo:
    """Gestor unificado de captura de video resiliente a cambios en caliente."""

    def __init__(
        self,
        origen: int | str = 0,
        backend: str = "auto",
        espejo: bool = True,
    ) -> None:
        self.backend = backend
        self.espejo = espejo
        self.lock = threading.Lock()
        self.activo = True

        self.cap_local: cv.VideoCapture | None = None
        self.lector_ip: LectorStreamIP | None = None
        self.simulada: CamaraSimulada | None = None
        self.origen_actual: int | str = 0

        self.cambiar_fuente(origen)

    def cambiar_fuente(self, nuevo_origen: int | str) -> bool:
        """Cambia la fuente de video en caliente protegiendo el hilo principal de desconexiones."""
        origen_norm = normalizar_origen_camara(nuevo_origen)
        logger.info("FuenteDeVideo cambiando fuente a: %s", origen_norm)

        with self.lock:
            self._liberar_fuente_actual()
            self.origen_actual = origen_norm

            # 1. Modo Simulado
            if origen_norm == "simulada":
                self.simulada = CamaraSimulada()
                return True

            # 2. Modo Stream IP (DroidCam)
            if isinstance(origen_norm, str) and (origen_norm.startswith("http://") or origen_norm.startswith("https://") or origen_norm.startswith("rtsp://")):
                self.lector_ip = LectorStreamIP(origen_norm)
                self.lector_ip.start()
                # Mantener simulada temporal con mensaje mientras conecta
                self.simulada = CamaraSimulada(mensaje_extra=f"Conectando a {origen_norm}...")
                return True

            # 3. Modo Webcam Local
            indice = int(origen_norm)
            backend_str = str(self.backend).lower().strip()
            if backend_str == "dshow":
                self.cap_local = cv.VideoCapture(indice, cv.CAP_DSHOW)
            elif backend_str == "v4l2":
                self.cap_local = cv.VideoCapture(indice, cv.CAP_V4L2)
            elif backend_str == "auto":
                if sys.platform == "win32":
                    self.cap_local = cv.VideoCapture(indice, cv.CAP_DSHOW)
                    if not self.cap_local.isOpened():
                        self.cap_local = cv.VideoCapture(indice)
                else:
                    self.cap_local = cv.VideoCapture(indice)
            else:
                self.cap_local = cv.VideoCapture(indice)

            if not self.cap_local.isOpened():
                logger.warning("Cámara local %d no disponible. Activando MODO SIMULADO de respaldo.", indice)
                self.simulada = CamaraSimulada(mensaje_extra="Camara local no disponible - Fallback simulado")

            return True

    def isOpened(self) -> bool:
        return self.activo

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            ret, frame = False, None

            # Si hay stream IP conectado con cuadros válidos, usarlo
            if self.lector_ip is not None:
                ret, frame = self.lector_ip.leer()
                if not ret and self.simulada is not None:
                    # Mientras el stream IP conecta, mostrar canvas informativo
                    ret, frame = self.simulada.read()
            elif self.cap_local is not None and self.cap_local.isOpened():
                ret, frame = self.cap_local.read()
            elif self.simulada is not None:
                ret, frame = self.simulada.read()

            if ret and frame is not None and self.espejo and (self.lector_ip or self.cap_local):
                frame = cv.flip(frame, 1)

            return ret, frame

    def _liberar_fuente_actual(self) -> None:
        if self.lector_ip:
            self.lector_ip.release()
            self.lector_ip = None
        if self.cap_local:
            try:
                self.cap_local.release()
            except Exception:
                pass
            self.cap_local = None
        if self.simulada:
            self.simulada.release()
            self.simulada = None

    def liberar(self) -> None:
        with self.lock:
            self.activo = False
            self._liberar_fuente_actual()

    def release(self) -> None:
        self.liberar()
