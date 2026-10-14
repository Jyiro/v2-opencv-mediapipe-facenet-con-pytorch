"""Módulo de estabilización temporal y espacial de rostros (Caché de Cara Estable).

Evita la ejecución redundante de redes de embeddings en cada cuadro limitando la inferencia
a rostros que se mantengan inmóviles durante una cantidad configurable de fotogramas,
reutilizando la identidad espacialmente para triplicar los FPS del sistema.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PistaCara:
    """Representa el seguimiento continuo y estado de un rostro detectado."""

    id_pista: int
    centro_x: float
    centro_y: float
    bbox: tuple[int, int, int, int]  # (origin_x, origin_y, width, height)
    contador_estable: int = 1
    evaluado: bool = False
    reconocido: bool = False
    nombre: str = "Desconocido"
    distancia: float = float("inf")
    umbral: float = 0.8
    cuadros_desde_eval: int = 0
    cuadros_perdida: int = 0


class MedidorFPS:
    """Calcula FPS reales usando time.perf_counter y promedio móvil."""

    def __init__(self, ventana: int = 30) -> None:
        self.ventana = ventana
        self._tiempos: list[float] = []
        self._ultimo_tiempo: float = time.perf_counter()
        self.fps_actual: float = 0.0

    def actualizar(self) -> float:
        ahora = time.perf_counter()
        delta = ahora - self._ultimo_tiempo
        self._ultimo_tiempo = ahora

        if delta > 0:
            self._tiempos.append(delta)
            if len(self._tiempos) > self.ventana:
                self._tiempos.pop(0)
            promedio = sum(self._tiempos) / len(self._tiempos)
            self.fps_actual = 1.0 / promedio if promedio > 0 else 0.0

        return self.fps_actual


class EstabilizadorDeCaras:
    """Gestiona el seguimiento temporal de caras y decide cuándo ejecutar FaceNet."""

    def __init__(
        self,
        frames_cara_estable: int = 4,
        tolerancia_movimiento_px: float = 35.0,
        intervalo_reevaluacion: int = 90,
    ) -> None:
        self.frames_cara_estable = frames_cara_estable
        self.tolerancia_movimiento_px = float(tolerancia_movimiento_px)
        self.intervalo_reevaluacion = intervalo_reevaluacion

        self._pistas: dict[int, PistaCara] = {}
        self._siguiente_id = 1
        self.medidor_fps = MedidorFPS()

    def actualizar_pistas(
        self, bboxes: list[tuple[int, int, int, int]]
    ) -> list[dict[str, Any]]:
        """Asocia las detecciones del cuadro actual con las pistas existentes.

        Args:
            bboxes: Lista de tuplas (x, y, w, h) detectadas en el fotograma.

        Returns:
            Lista de diccionarios con el estado y datos de cada rostro para el HUD y el bucle.
        """
        self.medidor_fps.actualizar()
        nuevos_centros = [
            (x + w / 2.0, y + h / 2.0, bbox) for bbox in bboxes for x, y, w, h in [bbox]
        ]

        pistas_actualizadas: set[int] = set()
        resultados: list[dict[str, Any]] = []

        for cx, cy, bbox in nuevos_centros:
            mejor_id: int | None = None
            menor_dist = float("inf")

            # Buscar la pista existente más cercana
            for pid, pista in self._pistas.items():
                if pid in pistas_actualizadas:
                    continue
                dist = math.hypot(cx - pista.centro_x, cy - pista.centro_y)
                if dist < menor_dist:
                    menor_dist = dist
                    mejor_id = pid

            # Umbral de asociación espacial (tolerancia ampliada para seguimiento continuo)
            umbral_asociacion = self.tolerancia_movimiento_px * 2.5

            if mejor_id is not None and menor_dist <= umbral_asociacion:
                pista = self._pistas[mejor_id]
                desplazamiento = math.hypot(cx - pista.centro_x, cy - pista.centro_y)

                if desplazamiento <= self.tolerancia_movimiento_px:
                    pista.contador_estable += 1
                else:
                    # Movimiento brusco: reiniciar estabilidad
                    pista.contador_estable = 1

                pista.centro_x = cx
                pista.centro_y = cy
                pista.bbox = bbox
                pista.cuadros_perdida = 0
                pista.cuadros_desde_eval += 1
                pistas_actualizadas.add(mejor_id)
                pista_actual = pista
            else:
                # Nueva pista
                pid = self._siguiente_id
                self._siguiente_id += 1
                nueva_pista = PistaCara(
                    id_pista=pid,
                    centro_x=cx,
                    centro_y=cy,
                    bbox=bbox,
                    contador_estable=1,
                )
                self._pistas[pid] = nueva_pista
                pistas_actualizadas.add(pid)
                pista_actual = nueva_pista

            # Determinar si requiere inferencia de embedding
            requiere_extraccion = False
            if not pista_actual.evaluado:
                if pista_actual.contador_estable >= self.frames_cara_estable:
                    requiere_extraccion = True
            else:
                if pista_actual.cuadros_desde_eval >= self.intervalo_reevaluacion:
                    requiere_extraccion = True

            resultados.append(
                {
                    "id_pista": pista_actual.id_pista,
                    "bbox": pista_actual.bbox,
                    "estable": pista_actual.contador_estable >= self.frames_cara_estable,
                    "contador_estable": pista_actual.contador_estable,
                    "meta_estable": self.frames_cara_estable,
                    "evaluado": pista_actual.evaluado,
                    "reconocido": pista_actual.reconocido,
                    "nombre": pista_actual.nombre,
                    "distancia": pista_actual.distancia,
                    "umbral": pista_actual.umbral,
                    "requiere_extraccion": requiere_extraccion,
                }
            )

        # Manejo de pistas que no tuvieron detección en este frame
        pistas_a_eliminar: list[int] = []
        for pid, pista in self._pistas.items():
            if pid not in pistas_actualizadas:
                pista.cuadros_perdida += 1
                if pista.cuadros_perdida > 5:
                    pistas_a_eliminar.append(pid)

        for pid in pistas_a_eliminar:
            del self._pistas[pid]

        return resultados

    def registrar_evaluacion(
        self,
        id_pista: int,
        nombre: str,
        distancia: float,
        umbral: float,
        reconocido: bool,
    ) -> None:
        """Guarda en la caché de la pista el resultado de la inferencia."""
        if id_pista in self._pistas:
            pista = self._pistas[id_pista]
            pista.evaluado = True
            pista.nombre = nombre
            pista.distancia = distancia
            pista.umbral = umbral
            pista.reconocido = reconocido
            pista.cuadros_desde_eval = 0

    def reiniciar(self) -> None:
        """Limpia todas las pistas activas para forzar una nueva estabilización y reevaluación inmediata."""
        self._pistas.clear()
