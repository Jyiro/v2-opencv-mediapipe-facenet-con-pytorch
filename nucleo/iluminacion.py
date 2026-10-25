"""Normalización de iluminación y control de calidad del recorte facial.

Idea (sugerida por el profesor): acercar el recorte de la cara a una imagen "bien
iluminada" antes de pasarlo a FaceNet, para que dos fotos de la misma persona con
luz distinta se parezcan más.

Pipeline: balance de blancos (gray-world) -> CLAHE sobre luminosidad -> gamma adaptativa.

IMPORTANTE: si se usa para reconocer, se debe aplicar IGUAL al enrolar y al reconocer.
El modo de pruebas calcula ambos embeddings (crudo y normalizado) para compararlos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2 as cv
import numpy as np

MODELO_CRUDO = "vggface2"
MODELO_NORMALIZADO = "vggface2+norm"


@dataclass(frozen=True)
class ConfigIluminacion:
    balance_blancos: bool = True
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grilla: int = 8
    gamma_adaptativa: bool = True
    brillo_objetivo: float = 0.5
    gamma_min: float = 0.45
    gamma_max: float = 2.2

    @classmethod
    def desde_dict(cls, datos: dict[str, Any] | None) -> "ConfigIluminacion":
        datos = datos or {}
        base = cls()
        valores: dict[str, Any] = {}
        for campo in base.__dataclass_fields__:
            if campo in datos:
                valores[campo] = type(getattr(base, campo))(datos[campo])
        return cls(**valores)


def estadisticas_luz(bgr: np.ndarray) -> dict[str, float]:
    """Brillo medio, contraste y asimetría lateral del recorte, en escala 0..1."""
    if bgr is None or bgr.size == 0:
        return {"brillo": 0.0, "contraste": 0.0, "asimetria_lateral": 0.0}
    gris = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    _, ancho = gris.shape
    mitad = ancho // 2
    if mitad > 0:
        izq = float(gris[:, :mitad].mean())
        der = float(gris[:, mitad:].mean())
        asimetria = float(abs(izq - der))
    else:
        asimetria = 0.0
    return {
        "brillo": float(gris.mean()),
        "contraste": float(gris.std()),
        "asimetria_lateral": asimetria,
    }


def balance_gray_world(bgr: np.ndarray) -> np.ndarray:
    """Corrige el tinte de la luz asumiendo que el promedio de la escena es gris."""
    medias = bgr.reshape(-1, 3).astype(np.float32).mean(axis=0)
    medias = np.maximum(medias, 1.0)
    gris = float(medias.mean())
    escala = gris / medias
    salida = bgr.astype(np.float32) * escala
    return np.clip(salida, 0, 255).astype(np.uint8)


def aplicar_clahe(bgr: np.ndarray, clip: float = 2.0, grilla: int = 8) -> np.ndarray:
    """Ecualiza el contraste por zonas sobre la luminosidad, sin alterar el color."""
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    ecualizador = cv.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grilla), int(grilla)))
    l2 = ecualizador.apply(l)
    return cv.cvtColor(cv.merge((l2, a, b)), cv.COLOR_LAB2BGR)


def gamma_para_objetivo(brillo_actual: float, objetivo: float, gamma_min: float, gamma_max: float) -> float:
    """Gamma tal que brillo_actual ** gamma == objetivo (gamma < 1 aclara)."""
    m = min(max(brillo_actual, 0.02), 0.98)
    obj = min(max(objetivo, 0.05), 0.95)
    gamma = math.log(obj) / math.log(m)
    return float(min(max(gamma, gamma_min), gamma_max))


def aplicar_gamma_adaptativa(
    bgr: np.ndarray, objetivo: float = 0.5, gamma_min: float = 0.45, gamma_max: float = 2.2
) -> np.ndarray:
    brillo = estadisticas_luz(bgr)["brillo"]
    gamma = gamma_para_objetivo(brillo, objetivo, gamma_min, gamma_max)
    tabla = (np.linspace(0.0, 1.0, 256) ** gamma * 255.0).astype(np.uint8)
    return cv.LUT(bgr, tabla)


def normalizar_iluminacion(bgr: np.ndarray, cfg: ConfigIluminacion | None = None) -> np.ndarray:
    """Devuelve una copia del recorte con la iluminación normalizada."""
    cfg = cfg or ConfigIluminacion()
    if bgr is None or bgr.size == 0:
        return bgr
    salida = bgr.copy()
    if cfg.balance_blancos:
        salida = balance_gray_world(salida)
    if cfg.clahe:
        salida = aplicar_clahe(salida, cfg.clahe_clip, cfg.clahe_grilla)
    if cfg.gamma_adaptativa:
        salida = aplicar_gamma_adaptativa(salida, cfg.brillo_objetivo, cfg.gamma_min, cfg.gamma_max)
    return salida


def desglosar_etapas_iluminacion(
    bgr: np.ndarray, cfg: ConfigIluminacion | None = None
) -> dict[str, Any]:
    """Devuelve cada una de las capas secuenciales del pipeline fotométrico junto a sus métricas."""
    cfg = cfg or ConfigIluminacion()
    if bgr is None or bgr.size == 0:
        return {}

    salida: dict[str, Any] = {}

    # Capa 1: Crudo original
    crudo = bgr.copy()
    salida["crudo"] = crudo
    salida["stats_crudo"] = estadisticas_luz(crudo)

    # Capa 2: Balance de blancos Gray-World
    gw = balance_gray_world(crudo) if cfg.balance_blancos else crudo.copy()
    salida["gray_world"] = gw
    salida["stats_gray_world"] = estadisticas_luz(gw)

    # Capa 3: CLAHE en canal L
    cl = aplicar_clahe(gw, cfg.clahe_clip, cfg.clahe_grilla) if cfg.clahe else gw.copy()
    salida["clahe"] = cl
    salida["stats_clahe"] = estadisticas_luz(cl)

    # Capa 4: Gamma adaptativa (resultado final)
    final = aplicar_gamma_adaptativa(cl, cfg.brillo_objetivo, cfg.gamma_min, cfg.gamma_max) if cfg.gamma_adaptativa else cl.copy()
    salida["gamma_final"] = final
    salida["stats_final"] = estadisticas_luz(final)
    salida["resultado"] = final

    return salida


def evaluar_calidad(
    recorte_bgr: np.ndarray,
    *,
    lado_minimo_px: int = 70,
    nitidez_minima: float = 6.0,
    brillo_rango: tuple[float, float] = (0.10, 0.93),
) -> dict[str, Any]:
    """Evalúa si un recorte facial posee la calidad geométrica y fotométrica para enrolamiento."""
    if recorte_bgr is None or recorte_bgr.size == 0:
        return {"ok": False, "motivo": "Rostro no detectado.", "nitidez": 0.0, "brillo": 0.0}
    alto, ancho = recorte_bgr.shape[:2]
    gris = cv.cvtColor(recorte_bgr, cv.COLOR_BGR2GRAY)
    nitidez = float(cv.Laplacian(gris, cv.CV_64F).var())
    brillo = float(gris.mean() / 255.0)

    if min(alto, ancho) < lado_minimo_px:
        motivo = "Acérquese a la cámara (Acercate al objetivo)."
    elif nitidez < nitidez_minima:
        motivo = "Imagen borrosa: mantenga el rostro inmóvil frente a la cámara."
    elif brillo < brillo_rango[0]:
        motivo = "Iluminación insuficiente: se detecta muy poca luz sobre el rostro."
    elif brillo > brillo_rango[1]:
        motivo = "Exceso de iluminación sobre el rostro."
    else:
        motivo = ""
    return {"ok": not motivo, "motivo": motivo, "nitidez": nitidez, "brillo": brillo}
