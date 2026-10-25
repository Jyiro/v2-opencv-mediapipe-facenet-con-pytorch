"""Estimación de la pose de la cabeza y guía de enrolamiento por poses."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

IDX_NARIZ = 1
IDX_BORDE_IZQ_IMAGEN = 234
IDX_BORDE_DER_IMAGEN = 454
IDX_FRENTE = 10
IDX_MENTON = 152

PITCH_GRADOS_POR_UNIDAD = 110.0
RAZON_VERTICAL_NEUTRA = 0.64


@dataclass(frozen=True)
class PoseCabeza:
    yaw: float
    pitch: float


class EstimadorPose:
    """Convierte landmarks normalizados en yaw/pitch aproximados (en grados)."""

    def __init__(self, espejo: bool = True) -> None:
        self.espejo = espejo
        self.base_vertical: float | None = None

    def _razon_vertical(self, landmarks: Sequence[Any]) -> float:
        frente = landmarks[IDX_FRENTE].y
        menton = landmarks[IDX_MENTON].y
        alto = max(menton - frente, 1e-6)
        return (landmarks[IDX_NARIZ].y - frente) / alto

    def fijar_referencia_frontal(self, landmarks: Sequence[Any]) -> None:
        """Define la referencia vertical de mirada al frente propia del usuario."""
        razon = self._razon_vertical(landmarks)
        # Acotar la referencia para que si el usuario estaba algo inclinado no arruine las poses arriba/abajo
        self.base_vertical = max(0.56, min(0.70, razon))

    def reiniciar(self) -> None:
        self.base_vertical = None

    def estimar(self, landmarks: Sequence[Any] | None) -> PoseCabeza | None:
        if not landmarks or len(landmarks) <= max(IDX_BORDE_DER_IMAGEN, IDX_MENTON):
            return None
        x_izq = landmarks[IDX_BORDE_IZQ_IMAGEN].x
        x_der = landmarks[IDX_BORDE_DER_IMAGEN].x
        ancho = x_der - x_izq
        if ancho <= 1e-6:
            return None

        desplazamiento = 2.0 * ((landmarks[IDX_NARIZ].x - x_izq) / ancho - 0.5)
        yaw_imagen = math.degrees(math.asin(max(-1.0, min(1.0, desplazamiento))))
        yaw = yaw_imagen if self.espejo else -yaw_imagen

        razon = self._razon_vertical(landmarks)
        base = self.base_vertical if self.base_vertical is not None else RAZON_VERTICAL_NEUTRA
        pitch = -(razon - base) * PITCH_GRADOS_POR_UNIDAD
        return PoseCabeza(yaw=float(yaw), pitch=float(pitch))


@dataclass(frozen=True)
class PasoPose:
    clave: str
    instruccion: str
    yaw: tuple[float, float]
    pitch: tuple[float, float]


# Rangos suaves y naturales donde ambos ojos y la nariz permanecen bien visibles para FaceNet
POSES_5: tuple[PasoPose, ...] = (
    PasoPose("frente", "Mirada al frente", (-12.0, 12.0), (-14.0, 16.0)),
    PasoPose("derecha", "Giro leve a la derecha", (10.0, 28.0), (-18.0, 18.0)),
    PasoPose("izquierda", "Giro leve a la izquierda", (-28.0, -10.0), (-18.0, 18.0)),
    PasoPose("arriba", "Mentón levemente arriba", (-15.0, 15.0), (5.0, 22.0)),
    PasoPose("abajo", "Mentón levemente abajo", (-15.0, 15.0), (-20.0, -4.0)),
)
POSES_3: tuple[PasoPose, ...] = POSES_5[:3]


def pasos_para(cantidad: int) -> tuple[PasoPose, ...]:
    return POSES_3 if int(cantidad) <= 3 else POSES_5


@dataclass
class ResultadoGuia:
    indice: int
    total: int
    clave: str
    instruccion: str
    mensaje: str
    coincide: bool
    progreso_sostener: float
    capturar: bool = False
    completado: bool = False
    hechas: list[str] = field(default_factory=list)


class GuiaDePoses:
    """Máquina de estados: solicita una pose, verifica estabilidad y captura automáticamente."""

    def __init__(
        self,
        pasos: Sequence[PasoPose] = POSES_5,
        sostener_seg: float = 0.5,
        aviso_seg: float = 8.0,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pasos = tuple(pasos)
        self.sostener_seg = sostener_seg
        self.aviso_seg = aviso_seg
        self._reloj = reloj
        self.indice = 0
        self._desde_coincide: float | None = None
        self._inicio_paso = self._reloj()
        self.hechas: list[str] = []

    @property
    def completado(self) -> bool:
        return self.indice >= len(self.pasos)

    @property
    def paso_actual(self) -> PasoPose | None:
        return None if self.completado else self.pasos[self.indice]

    @staticmethod
    def _dentro(valor: float, rango: tuple[float, float]) -> bool:
        return rango[0] <= valor <= rango[1]

    def _pista_de_movimiento(self, paso: PasoPose, pose: PoseCabeza) -> str:
        if paso.clave == "frente":
            if not self._dentro(pose.yaw, paso.yaw):
                return "Gire levemente hacia la derecha." if pose.yaw < paso.yaw[0] else "Gire levemente hacia la izquierda."
            if not self._dentro(pose.pitch, paso.pitch):
                return "Eleve ligeramente el mentón." if pose.pitch < paso.pitch[0] else "Baje ligeramente el mentón."
            return "Mantenga el rostro centrado hacia la cámara."

        if paso.clave == "derecha":
            if pose.yaw < paso.yaw[0]:
                return "Gire la cabeza suavemente hacia la derecha."
            if pose.yaw > paso.yaw[1]:
                return "Gire un poco menos hacia la derecha."
            return "Mantenga el mentón nivelado."

        if paso.clave == "izquierda":
            if pose.yaw > paso.yaw[1]:
                return "Gire la cabeza suavemente hacia la izquierda."
            if pose.yaw < paso.yaw[0]:
                return "Gire un poco menos hacia la izquierda."
            return "Mantenga el mentón nivelado."

        if paso.clave == "arriba":
            if pose.pitch < paso.pitch[0]:
                return "Levante apenas el mentón."
            if pose.pitch > paso.pitch[1]:
                return "No levante tanto el mentón."
            return "Centre el rostro horizontalmente."

        if paso.clave == "abajo":
            if pose.pitch > paso.pitch[1]:
                return "Incline apenas el mentón hacia abajo."
            if pose.pitch < paso.pitch[0]:
                return "Levante un poco la mirada (no baje tanto)."
            return "Centre el rostro horizontalmente."

        return paso.instruccion

    def actualizar(self, pose: PoseCabeza | None, calidad_ok: bool = True, motivo_calidad: str = "") -> ResultadoGuia:
        ahora = self._reloj()
        paso = self.paso_actual
        total = len(self.pasos)

        if paso is None:
            return ResultadoGuia(total, total, "fin", "Completado", "Enrolamiento completado con éxito.", True, 1.0, False, True, list(self.hechas))

        if pose is None:
            self._desde_coincide = None
            return ResultadoGuia(
                self.indice, total, paso.clave, paso.instruccion,
                "Rostro no detectado: colóquese frente a la cámara.", False, 0.0, hechas=list(self.hechas),
            )

        en_pose = self._dentro(pose.yaw, paso.yaw) and self._dentro(pose.pitch, paso.pitch)
        if en_pose and calidad_ok:
            if self._desde_coincide is None:
                self._desde_coincide = ahora
            sostenido = ahora - self._desde_coincide
            progreso = min(1.0, sostenido / max(self.sostener_seg, 1e-6))
            if sostenido >= self.sostener_seg:
                self.hechas.append(paso.clave)
                self.indice += 1
                self._desde_coincide = None
                self._inicio_paso = ahora
                if self.completado:
                    return ResultadoGuia(
                        self.indice, total, "fin", "Completado", "Enrolamiento completado con éxito.", True, 1.0, True, True, list(self.hechas)
                    )
                siguiente = self.pasos[self.indice]
                return ResultadoGuia(
                    self.indice, total, siguiente.clave, siguiente.instruccion,
                    f"Muestra registrada. Ahora: {siguiente.instruccion}.", False, 0.0, True, False, list(self.hechas),
                )
            return ResultadoGuia(
                self.indice, total, paso.clave, paso.instruccion, "Mantenga la posición...", True, progreso,
                hechas=list(self.hechas),
            )

        self._desde_coincide = None
        if en_pose and not calidad_ok:
            mensaje = motivo_calidad or "Ajuste el encuadre o la iluminación."
        else:
            mensaje = self._pista_de_movimiento(paso, pose)
        return ResultadoGuia(
            self.indice, total, paso.clave, paso.instruccion, mensaje, False, 0.0, hechas=list(self.hechas)
        )
