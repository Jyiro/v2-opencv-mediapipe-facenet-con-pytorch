"""Módulo de extracción de embeddings faciales con arquitectura modular y de alto rendimiento.

Elimina la doble detección facial redundante (MTCNN sobre BlazeFace) y ofrece
soporte para FaceNet rápido (vggface2), ArcFace en ONNX Runtime y modo FaceNet Legacy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path

import cv2 as cv
import numpy as np
import torch


logger = logging.getLogger(__name__)


def recortar_con_margen(
    imagen: np.ndarray,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    margen_pct: float = 0.10,
) -> np.ndarray | None:
    """Extrae un recorte facial agregando un margen porcentual sin exceder los límites del cuadro."""
    alto_img, ancho_img = imagen.shape[:2]

    delta_x = int(width * margen_pct)
    delta_y = int(height * margen_pct)

    x1 = max(0, origin_x - delta_x)
    y1 = max(0, origin_y - delta_y)
    x2 = min(ancho_img, origin_x + width + delta_x)
    y2 = min(alto_img, origin_y + height + delta_y)

    if x2 <= x1 or y2 <= y1:
        return None

    recorte = imagen[y1:y2, x1:x2]
    return recorte if recorte.size > 0 else None


class ExtractorEmbeddings(ABC):
    """Interfaz base para extractores de representaciones faciales."""

    @property
    @abstractmethod
    def nombre_modelo(self) -> str:
        """Nombre identificador del modelo (ej. 'vggface2', 'arcface_onnx')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensión del vector de embedding resultante."""
        pass

    @abstractmethod
    def generar_embedding(self, recorte_bgr: np.ndarray) -> np.ndarray | None:
        """Genera un vector embedding 1D (np.float32) a partir de una cara en BGR."""
        pass


class ExtractorFaceNetRapido(ExtractorEmbeddings):
    """FaceNet optimizado: InceptionResnetV1 directamente sobre el recorte de BlazeFace."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        logger.info("Cargando InceptionResnetV1 (vggface2) en %s...", self.device)
        from facenet_pytorch import InceptionResnetV1

        self.modelo = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        logger.info("InceptionResnetV1 cargado correctamente (modo rápido sin MTCNN)")

    @property
    def nombre_modelo(self) -> str:
        return "vggface2"

    @property
    def dimension(self) -> int:
        return 512

    def generar_embedding(self, recorte_bgr: np.ndarray) -> np.ndarray | None:
        """Procesa el recorte facial de 160x160 y genera el embedding normalizado."""
        try:
            if recorte_bgr is None or recorte_bgr.size == 0:
                return None

            recorte_160 = cv.resize(recorte_bgr, (160, 160), interpolation=cv.INTER_AREA)
            recorte_rgb = cv.cvtColor(recorte_160, cv.COLOR_BGR2RGB)

            # Estandarización idéntica a facenet: (x - 127.5) / 128.0
            img_norm = (recorte_rgb.astype(np.float32) - 127.5) / 128.0
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                emb = self.modelo(tensor)
                vector = emb.detach().cpu().numpy().flatten().astype(np.float32)

            norma = np.linalg.norm(vector)
            if norma > 0:
                vector = vector / norma
            return vector
        except Exception:
            logger.exception("Error al generar embedding con FaceNet rápido")
            return None


class ExtractorArcFaceONNX(ExtractorEmbeddings):
    """Extractor ArcFace usando ONNX Runtime para hardware limitado."""

    def __init__(self, ruta_modelo: str | Path, fallback_facenet: bool = True) -> None:
        self.ruta_modelo = Path(ruta_modelo)
        self._sesion = None
        self._fallback: ExtractorEmbeddings | None = None

        try:
            import onnxruntime as ort

            if not self.ruta_modelo.exists():
                raise FileNotFoundError(f"Archivo de modelo ONNX no encontrado: {self.ruta_modelo}")

            self._sesion = ort.InferenceSession(
                str(self.ruta_modelo), providers=["CPUExecutionProvider"]
            )
            self._nombre_entrada = self._sesion.get_inputs()[0].name
            logger.info("Modelo ONNX ArcFace cargado exitosamente desde %s", self.ruta_modelo)
        except Exception as e:
            logger.warning(
                "No se pudo cargar el modelo ONNX ArcFace (%s). Usando fallback a FaceNet rápido.",
                e,
            )
            if fallback_facenet:
                self._fallback = ExtractorFaceNetRapido()
            else:
                raise

    @property
    def nombre_modelo(self) -> str:
        if self._sesion is not None:
            return "arcface_onnx"
        return self._fallback.nombre_modelo if self._fallback else "arcface_onnx"

    @property
    def dimension(self) -> int:
        return 512

    def generar_embedding(self, recorte_bgr: np.ndarray) -> np.ndarray | None:
        if self._sesion is None and self._fallback is not None:
            return self._fallback.generar_embedding(recorte_bgr)

        try:
            if recorte_bgr is None or recorte_bgr.size == 0:
                return None

            recorte_112 = cv.resize(recorte_bgr, (112, 112), interpolation=cv.INTER_AREA)
            recorte_rgb = cv.cvtColor(recorte_112, cv.COLOR_BGR2RGB)
            img_norm = (recorte_rgb.astype(np.float32) - 127.5) / 128.0
            blob = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

            salida = self._sesion.run(None, {self._nombre_entrada: blob})[0]
            vector = salida.flatten().astype(np.float32)
            norma = np.linalg.norm(vector)
            if norma > 0:
                vector = vector / norma
            return vector
        except Exception:
            logger.exception("Error al inferir embedding con ArcFace ONNX")
            return None


class ExtractorFaceNetLegacy(ExtractorEmbeddings):
    """Modo legacy: Conserva MTCNN + ResNet para pruebas comparativas de rendimiento."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        logger.info("Cargando modo legacy (MTCNN + InceptionResnetV1)...")
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from PIL import Image

        self._PIL_Image = Image
        self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @property
    def nombre_modelo(self) -> str:
        return "vggface2"

    @property
    def dimension(self) -> int:
        return 512

    def generar_embedding(self, recorte_bgr: np.ndarray) -> np.ndarray | None:
        try:
            if recorte_bgr is None or recorte_bgr.size == 0:
                return None
            recorte_rgb = cv.cvtColor(recorte_bgr, cv.COLOR_BGR2RGB)
            imagen_pil = self._PIL_Image.fromarray(recorte_rgb)
            cara_tensor = self.mtcnn(imagen_pil)
            if cara_tensor is None:
                return None
            embedding = self.resnet(cara_tensor.unsqueeze(0).to(self.device))
            vector = embedding.detach().cpu().numpy().flatten().astype(np.float32)
            norma = np.linalg.norm(vector)
            if norma > 0:
                vector = vector / norma
            return vector
        except Exception:
            logger.exception("Error en generar_embedding legacy")
            return None


def obtener_extractor(
    tipo_motor: str = "facenet_rapido",
    ruta_modelo_onnx: str | Path | None = None,
    dispositivo: str = "cpu",
) -> ExtractorEmbeddings:
    """Fábrica de extractores según la configuración del proyecto."""
    motor = (tipo_motor or "facenet_rapido").lower().strip()
    if motor == "facenet_rapido":
        return ExtractorFaceNetRapido(device=dispositivo)
    elif motor == "onnx_arcface":
        ruta = ruta_modelo_onnx or "modelos_onnx/w600k_mbf.onnx"
        return ExtractorArcFaceONNX(ruta_modelo=ruta, fallback_facenet=True)
    elif motor == "facenet_legacy":
        return ExtractorFaceNetLegacy(device=dispositivo)
    else:
        logger.warning("Motor desconocido '%s'. Usando facenet_rapido por defecto.", tipo_motor)
        return ExtractorFaceNetRapido(device=dispositivo)
