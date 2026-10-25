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


# Las 4 resoluciones que se guardan automáticamente en cada alta para poder hacer Hot-Swap
RESOLUCIONES_SOPORTADAS: tuple[int, ...] = (112, 128, 160, 224)


def nombre_modelo_resolucion(resolucion: int) -> str:
    """Devuelve la clave de modelo en SQLite para una resolución dada."""
    res = int(resolucion)
    if res == 160:
        return "vggface2"  # Retrocompatible con la galería estándar
    return f"vggface2_{res}"


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


def redimensionar_recorte(recorte: np.ndarray, target_size: int = 160) -> np.ndarray:
    """Lleva el recorte a la resolución deseada y lo adapta al tensor 160x160 de FaceNet.

    - Si target_size == 160: redimensiona directo a 160x160 (INTER_AREA al achicar, INTER_LINEAR al agrandar).
    - Si target_size != 160 (ej. 112, 128, 224): muestrea la cara exactamente a (target_size x target_size)
      y luego la adapta bilinealmente a 160x160 para que las capas convolucionales de InceptionResnetV1
      mantengan su geometría espacial de 3x3 sin colapsar.
    """
    if recorte is None or recorte.size == 0:
        return recorte
    h, w = recorte.shape[:2]
    interp_escala = cv.INTER_AREA if (w > target_size and h > target_size) else cv.INTER_LINEAR
    escalado = recorte if (w, h) == (target_size, target_size) else cv.resize(recorte, (target_size, target_size), interpolation=interp_escala)

    if target_size == 160:
        return escalado
    interp_red = cv.INTER_AREA if target_size > 160 else cv.INTER_LINEAR
    return cv.resize(escalado, (160, 160), interpolation=interp_red)


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

    def generar_par_embeddings(
        self, recorte_a: np.ndarray, recorte_b: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Genera dos embeddings en lote (batch B=2) o de forma secuencial."""
        return self.generar_embedding(recorte_a), self.generar_embedding(recorte_b)


class ExtractorFaceNetRapido(ExtractorEmbeddings):
    """FaceNet optimizado con soporte de inferencia en lote multi-resolución."""

    RESOLUCIONES_VALIDAS: tuple[int, ...] = RESOLUCIONES_SOPORTADAS

    def __init__(self, device: str = "cpu", resolucion: int = 160) -> None:
        self.device = torch.device(device)
        self.resolucion = int(resolucion) if int(resolucion) in self.RESOLUCIONES_VALIDAS else 160
        logger.info("Cargando InceptionResnetV1 (vggface2) en %s (resolución activa=%dpx)...", self.device, self.resolucion)
        from facenet_pytorch import InceptionResnetV1

        self.modelo = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        logger.info("InceptionResnetV1 listo.")

    def fijar_resolucion(self, resolucion: int) -> None:
        res = int(resolucion)
        if res in self.RESOLUCIONES_VALIDAS:
            self.resolucion = res

    @property
    def nombre_modelo(self) -> str:
        return nombre_modelo_resolucion(self.resolucion)

    @property
    def dimension(self) -> int:
        return 512

    def generar_embedding(self, recorte_bgr: np.ndarray, resolucion: int | None = None) -> np.ndarray | None:
        """Genera el embedding normalizado para la resolución activa o la indicada."""
        try:
            if recorte_bgr is None or recorte_bgr.size == 0:
                return None

            res = int(resolucion) if resolucion is not None else self.resolucion
            recorte_160 = redimensionar_recorte(recorte_bgr, res)
            recorte_rgb = cv.cvtColor(recorte_160, cv.COLOR_BGR2RGB)

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
            logger.exception("Error al generar embedding con FaceNet")
            return None

    def generar_embeddings_multiresolucion(
        self,
        recorte_bgr: np.ndarray,
        resoluciones: tuple[int, ...] = RESOLUCIONES_SOPORTADAS,
    ) -> dict[int, np.ndarray]:
        """Calcula en UNA SOLA pasada neuronal (batch) los embeddings de todas las resoluciones (112, 128, 160, 224).

        Se usa durante el alta para guardar el perfil en todas las resoluciones a la vez
        y permitir hacer Hot-Swap después sin re-enrolar.
        """
        if recorte_bgr is None or recorte_bgr.size == 0:
            return {}
        try:
            tensores = []
            for res in resoluciones:
                img_160 = redimensionar_recorte(recorte_bgr, res)
                rgb = cv.cvtColor(img_160, cv.COLOR_BGR2RGB)
                norm = (rgb.astype(np.float32) - 127.5) / 128.0
                tensores.append(norm.transpose(2, 0, 1))

            batch = np.stack(tensores, axis=0)
            t_batch = torch.from_numpy(batch).to(self.device)

            with torch.inference_mode():
                embs = self.modelo(t_batch).detach().cpu().numpy().astype(np.float32)

            resultado: dict[int, np.ndarray] = {}
            for i, res in enumerate(resoluciones):
                v = embs[i].flatten()
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                resultado[int(res)] = v
            return resultado
        except Exception:
            logger.exception("Error al generar embeddings multi-resolución")
            return {}

    def generar_par_embeddings(
        self, recorte_a: np.ndarray, recorte_b: np.ndarray, resolucion: int | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.generar_embedding(recorte_a, resolucion), self.generar_embedding(recorte_b, resolucion)


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
