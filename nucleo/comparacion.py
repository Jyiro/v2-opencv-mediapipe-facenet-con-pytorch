"""Módulo de comparación vectorizada de embeddings faciales.

Implementa la matemática de matching con soporte para múltiples muestras por persona
y métodos de agregación por distancia mínima o vector promedio normalizado.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def calcular_distancia_persona(
    embedding_actual: np.ndarray,
    muestras_persona: list[np.ndarray],
    metodo: str = "minimo",
) -> float:
    """Calcula la distancia euclídea entre el embedding actual y las muestras de una persona.

    Args:
        embedding_actual: Vector 1D (np.float32) de la cara detectada.
        muestras_persona: Lista de vectores 1D de la persona registrada.
        metodo: "minimo" toma la distancia más cercana; "promedio" compara contra
                el vector promedio normalizado.

    Returns:
        Distancia euclídea (float).
    """
    if not muestras_persona:
        return float("inf")

    matriz_muestras = np.vstack(muestras_persona)  # Shape (N, D)

    if metodo == "promedio":
        vector_promedio = np.mean(matriz_muestras, axis=0)
        norma = np.linalg.norm(vector_promedio)
        if norma > 0:
            vector_promedio = vector_promedio / norma
        return float(np.linalg.norm(embedding_actual - vector_promedio))

    # Método "minimo": distancia vectorizada contra cada fila
    distancias = np.linalg.norm(matriz_muestras - embedding_actual, axis=1)
    return float(np.min(distancias))


def consultar_embeddings(
    embedding_actual: np.ndarray | None,
    base_embeddings: dict[str, list[np.ndarray]],
    umbral: float = 0.8,
    metodo: str = "minimo",
) -> dict[str, Any]:
    """Consulta una base de embeddings agrupada por persona y determina la mejor coincidencia.

    Args:
        embedding_actual: Vector 1D del rostro a identificar.
        base_embeddings: Diccionario {nombre: [vector1, vector2, ...]}.
        umbral: Distancia euclídea máxima admisible (menor es más estricto).
        metodo: "minimo" o "promedio".

    Returns:
        Diccionario con las claves:
        - "reconocido": bool indicando si la distancia mínima superó el umbral.
        - "nombre": nombre de la persona si reconocido, o "Desconocido".
        - "distancia": distancia euclídea cruda mínima encontrada (float).
        - "umbral": umbral configurado.
    """
    if embedding_actual is None or not base_embeddings:
        return {
            "reconocido": False,
            "nombre": "Desconocido",
            "distancia": float("inf"),
            "umbral": float(umbral),
        }

    minima_distancia_global = float("inf")
    mejor_candidato: str | None = None

    for nombre, lista_vectores in base_embeddings.items():
        if not lista_vectores:
            continue
        distancia = calcular_distancia_persona(
            embedding_actual=embedding_actual,
            muestras_persona=lista_vectores,
            metodo=metodo,
        )
        if distancia < minima_distancia_global:
            minima_distancia_global = distancia
            mejor_candidato = nombre

    reconocido = minima_distancia_global < umbral and mejor_candidato is not None
    nombre_final = mejor_candidato if reconocido else "Desconocido"

    return {
        "reconocido": reconocido,
        "nombre": nombre_final,
        "distancia": round(minima_distancia_global, 4),
        "umbral": float(umbral),
    }


def comparar_caras_legacy(
    embedding_actual: np.ndarray | None,
    base_embeddings: dict[str, list[np.ndarray]],
    umbral: float = 0.8,
    metodo: str = "minimo",
) -> str | None:
    """Función de conveniencia retrocompatible con el código original.

    Devuelve el nombre de la persona si fue reconocida, o None si no.
    """
    resultado = consultar_embeddings(
        embedding_actual=embedding_actual,
        base_embeddings=base_embeddings,
        umbral=umbral,
        metodo=metodo,
    )
    return resultado["nombre"] if resultado["reconocido"] else None


def existe_coincidencia(
    embedding_actual: np.ndarray,
    base_embeddings: dict[str, list[np.ndarray]],
    umbral: float = 0.8,
) -> bool:
    """Verifica si un embedding ya pertenece a una persona en la base de datos."""
    resultado = consultar_embeddings(
        embedding_actual=embedding_actual,
        base_embeddings=base_embeddings,
        umbral=umbral,
        metodo="minimo",
    )
    return bool(resultado["reconocido"])
