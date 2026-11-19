"""Componentes puros de adaptación continua segura en uso (ADR 008).

Contiene únicamente lógica determinista sin E/S ni cámara:
- ``BufferPistaTopK``: retiene en RAM los Top-3 mejores embeddings ya calculados
  por pista para fusión ponderada efímera en inferencia (sin muestreo por reloj).
- ``evaluar_compuertas_slot``: aplica las 4 compuertas obligatorias + veto inter-clase
  antes de promover un slot adaptativo discreto en Modo LIS.
- ``DetectorAlertaEntorno``: detecta fallas ambientales simultáneas (>= 3 usuarios
  distintos fallando en 15 min) para diferenciar mala luz/óptica de deriva facial.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from nucleo.comparacion import (
    IndiceEmbeddingsRAM,
    calcular_distancia_persona,
    normalizar_vector,
)


@dataclass(frozen=True)
class MuestraPista:
    embedding: np.ndarray
    puntaje: float
    nitidez: float
    brillo: float
    yaw: float | None
    pitch: float | None
    timestamp: float


class BufferPistaTopK:
    """Retiene en RAM los Top-K mejores embeddings ya calculados para una pista facial."""

    def __init__(
        self,
        capacidad: int = 3,
        max_yaw: float = 20.0,
        max_pitch: float = 15.0,
        reloj: Callable[[], float] | None = None,
    ) -> None:
        self.capacidad = max(1, int(capacidad))
        self.max_yaw = float(max_yaw)
        self.max_pitch = float(max_pitch)
        self._reloj = reloj or time.monotonic
        self._muestras: list[MuestraPista] = []

    @property
    def cantidad(self) -> int:
        return len(self._muestras)

    def limpiar(self) -> None:
        self._muestras.clear()

    def mejor_muestra(self) -> MuestraPista | None:
        return self._muestras[0] if self._muestras else None

    def registrar(
        self,
        embedding: np.ndarray | None,
        *,
        nitidez: float,
        brillo: float,
        calidad_ok: bool,
        yaw: float | None = None,
        pitch: float | None = None,
        timestamp: float | None = None,
    ) -> bool:
        """Registra una muestra solo si supera filtros de calidad y frontalidad."""
        if embedding is None or not calidad_ok:
            return False
        if yaw is not None and abs(float(yaw)) > self.max_yaw:
            return False
        if pitch is not None and abs(float(pitch)) > self.max_pitch:
            return False

        arr = np.asarray(embedding, dtype=np.float32).flatten()
        if arr.size == 0 or float(np.linalg.norm(arr)) == 0.0:
            return False
        vec = normalizar_vector(arr).astype(np.float32, copy=False)

        norm_nitidez = min(max(float(nitidez), 0.0), 300.0) / 300.0
        if yaw is not None and pitch is not None:
            penalizacion = 0.5 * (
                abs(float(yaw)) / max(self.max_yaw, 1e-6)
                + abs(float(pitch)) / max(self.max_pitch, 1e-6)
            )
            bonus_frontal = max(0.0, 1.0 - penalizacion)
        else:
            bonus_frontal = 0.8

        puntaje = float(norm_nitidez * 0.6 + bonus_frontal * 0.4)
        ts = float(timestamp) if timestamp is not None else float(self._reloj())

        self._muestras.append(
            MuestraPista(
                embedding=vec,
                puntaje=puntaje,
                nitidez=float(nitidez),
                brillo=float(brillo),
                yaw=float(yaw) if yaw is not None else None,
                pitch=float(pitch) if pitch is not None else None,
                timestamp=ts,
            )
        )
        self._muestras.sort(key=lambda m: m.puntaje, reverse=True)
        if len(self._muestras) > self.capacidad:
            del self._muestras[self.capacidad :]
        return True

    def embedding_fusionado(self) -> np.ndarray | None:
        """Combina los Top-K embeddings retenidos ponderando por su puntaje de calidad."""
        if not self._muestras:
            return None
        if len(self._muestras) == 1:
            return self._muestras[0].embedding

        pesos = np.asarray([max(m.puntaje, 0.05) for m in self._muestras], dtype=np.float32)
        vectores = np.vstack([m.embedding for m in self._muestras]).astype(np.float32, copy=False)
        combinado = np.sum(vectores * pesos[:, None], axis=0, dtype=np.float32)
        if float(np.linalg.norm(combinado)) == 0.0:
            return self._muestras[0].embedding
        return normalizar_vector(combinado).astype(np.float32, copy=False)


def _respuesta_compuerta(
    aprobado: bool,
    motivo: str,
    decision_shadow: str,
    *,
    dist_ancla: float | None = None,
    dist_otro_mas_cercano: float | None = None,
    otro_mas_cercano: str | None = None,
    margen_inter_clase: float | None = None,
) -> dict[str, Any]:
    return {
        "aprobado": aprobado,
        "motivo": motivo,
        "decision_shadow": decision_shadow,
        "dist_ancla": round(float(dist_ancla), 4) if dist_ancla is not None else None,
        "dist_otro_mas_cercano": (
            round(float(dist_otro_mas_cercano), 4) if dist_otro_mas_cercano is not None else None
        ),
        "otro_mas_cercano": otro_mas_cercano,
        "margen_inter_clase": (
            round(float(margen_inter_clase), 4) if margen_inter_clase is not None else None
        ),
    }


def _buscar_competidor_mas_cercano(
    emb_c: np.ndarray,
    indice_galeria: IndiceEmbeddingsRAM | None,
    nombre_usuario: str,
    umbral_seguridad: float,
) -> tuple[float | None, str | None]:
    if indice_galeria is None or indice_galeria.total_personas == 0:
        return None, None
    res = indice_galeria.consultar(
        emb_c,
        umbral=umbral_seguridad,
        metodo="minimo",
        margen_separacion=0.0,
    )
    cand1 = res.get("candidato_mas_cercano")
    cand2 = res.get("segundo_candidato")
    if cand1 and cand1.lower() != nombre_usuario.strip().lower():
        return float(res["distancia"]), str(cand1)
    if cand2:
        dist2 = res.get("distancia_segundo")
        return (float(dist2) if dist2 is not None else None), str(cand2)
    return None, None


def evaluar_compuertas_slot(
    embedding_candidato: np.ndarray | None,
    anclas_usuario: list[np.ndarray],
    indice_galeria: IndiceEmbeddingsRAM | None,
    nombre_usuario: str,
    *,
    pistas_activas: int,
    liveness_ok: bool,
    calidad_ok: bool,
    yaw: float | None = None,
    pitch: float | None = None,
    dist_min_ancla: float = 0.42,
    dist_max_ancla: float = 0.75,
    umbral_seguridad: float = 0.68,
    margen_inter_clase: float = 0.12,
    max_yaw: float = 20.0,
    max_pitch: float = 15.0,
) -> dict[str, Any]:
    """Evalúa las 4 compuertas obligatorias + veto inter-clase antes de guardar un slot."""
    if int(pistas_activas) != 1:
        return _respuesta_compuerta(
            False,
            "Se requiere exactamente una persona en cuadro para actualizar perfil.",
            "multiples_pistas",
        )

    if not liveness_ok:
        return _respuesta_compuerta(
            False,
            "Confirmación de presencia viva (gesto/liveness) requerida.",
            "sin_liveness",
        )

    pose_invalida = (yaw is not None and abs(float(yaw)) > max_yaw) or (
        pitch is not None and abs(float(pitch)) > max_pitch
    )
    if embedding_candidato is None or not calidad_ok or pose_invalida:
        return _respuesta_compuerta(
            False,
            "Calidad de imagen o frontalidad insuficiente.",
            "calidad_o_pose_insuficiente",
        )

    emb_c = normalizar_vector(np.asarray(embedding_candidato, dtype=np.float32).flatten())
    if emb_c.size == 0 or float(np.linalg.norm(emb_c)) == 0.0:
        return _respuesta_compuerta(False, "Embedding candidato vacío o inválido.", "calidad_o_pose_insuficiente")

    if not anclas_usuario:
        return _respuesta_compuerta(False, "El usuario no posee anclas base registradas.", "sin_anclas")

    dist_ancla = calcular_distancia_persona(emb_c, anclas_usuario, metodo="minimo")
    if dist_ancla <= dist_min_ancla:
        return _respuesta_compuerta(
            False,
            "Muestra redundante: ya está cubierta con holgura por las anclas.",
            "redundante_ya_cercano",
            dist_ancla=dist_ancla,
        )

    if dist_ancla >= dist_max_ancla:
        return _respuesta_compuerta(
            False,
            "Muestra demasiado alejada de las anclas del usuario; requiere re-enrolamiento.",
            "fuera_banda_ancla",
            dist_ancla=dist_ancla,
        )

    dist_otro, otro_nombre = _buscar_competidor_mas_cercano(
        emb_c, indice_galeria, nombre_usuario, umbral_seguridad
    )
    margen_real = (dist_otro - dist_ancla) if dist_otro is not None else None
    if dist_otro is not None:
        limite_inter = max(float(umbral_seguridad), dist_ancla + float(margen_inter_clase))
        if dist_otro < limite_inter:
            return _respuesta_compuerta(
                False,
                f"Veto inter-clase: candidato demasiado próximo a '{otro_nombre}' (d={dist_otro:.3f}).",
                "veto_inter_clase",
                dist_ancla=dist_ancla,
                dist_otro_mas_cercano=dist_otro,
                otro_mas_cercano=otro_nombre,
                margen_inter_clase=margen_real,
            )

    return _respuesta_compuerta(
        True,
        "Compuertas aprobadas para registrar slot adaptativo discreto.",
        "promover_slot",
        dist_ancla=dist_ancla,
        dist_otro_mas_cercano=dist_otro,
        otro_mas_cercano=otro_nombre,
        margen_inter_clase=margen_real,
    )


class DetectorAlertaEntorno:
    """Detecta fallos repetidos de distintos usuarios en una ventana corta (luz/lente)."""

    def __init__(
        self,
        ventana_seg: float = 900.0,
        min_usuarios_distintos: int = 3,
        reloj: Callable[[], float] | None = None,
    ) -> None:
        self.ventana_seg = float(ventana_seg)
        self.min_usuarios_distintos = max(1, int(min_usuarios_distintos))
        self._reloj = reloj or time.monotonic
        self._fallos: list[tuple[float, str]] = []

    def _purgar(self, ahora: float) -> None:
        limite = ahora - self.ventana_seg
        self._fallos = [(t, u) for (t, u) in self._fallos if t >= limite]

    def registrar_fallo(self, clave_usuario: str, ts: float | None = None) -> bool:
        ahora = float(ts) if ts is not None else float(self._reloj())
        usuario = str(clave_usuario or "").strip().lower() or "desconocido"
        self._fallos.append((ahora, usuario))
        self._purgar(ahora)
        return self.alerta_activa(ts=ahora)

    def alerta_activa(self, ts: float | None = None) -> bool:
        ahora = float(ts) if ts is not None else float(self._reloj())
        self._purgar(ahora)
        unicos = {u for _, u in self._fallos}
        return len(unicos) >= self.min_usuarios_distintos

