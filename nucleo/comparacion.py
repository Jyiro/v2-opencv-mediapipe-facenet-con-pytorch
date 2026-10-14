from __future__ import annotations
from typing import Any
import numpy as np


def normalizar_vector(v: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(v)
    if norma == 0:
        return v
    return v / norma


def calcular_distancia_persona(
    embedding: np.ndarray | None = None,
    lista_vectores: list[np.ndarray] | None = None,
    metodo: str = "minimo",
    *,
    embedding_actual: np.ndarray | None = None,
    muestras_persona: list[np.ndarray] | None = None,
) -> float:
    emb = embedding if embedding is not None else embedding_actual
    muestras = lista_vectores if lista_vectores is not None else muestras_persona
    if emb is None or not muestras:
        return float("inf")
    matriz = np.vstack([np.asarray(v, dtype=np.float32).flatten() for v in muestras])
    emb_c = np.asarray(emb, dtype=np.float32).flatten()
    if metodo == "promedio":
        centroide = np.mean(matriz, axis=0)
        norma = np.linalg.norm(centroide)
        if norma > 0:
            centroide = centroide / norma
        return float(np.linalg.norm(centroide - emb_c))
    distancias = np.linalg.norm(matriz - emb_c, axis=1)
    return float(np.min(distancias))


def existe_coincidencia(
    embedding_actual: np.ndarray,
    base_embeddings: dict[str, list[np.ndarray]] | list[np.ndarray],
    umbral: float = 0.80,
    metodo: str = "minimo",
) -> bool:
    if isinstance(base_embeddings, dict):
        res = consultar_embeddings(
            embedding_actual=embedding_actual,
            base_embeddings=base_embeddings,
            umbral=umbral,
            metodo=metodo,
        )
        return bool(res["reconocido"])
    elif isinstance(base_embeddings, list):
        d = calcular_distancia_persona(embedding_actual=embedding_actual, muestras_persona=base_embeddings, metodo=metodo)
        return bool(d < umbral)
    return False


def consultar_embeddings(
    embedding: np.ndarray | None = None,
    base_embeddings: dict[str, list[np.ndarray]] | None = None,
    umbral: float = 0.80,
    metodo: str = "minimo",
    embedding_actual: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compara un embedding de 512 dimensiones contra el diccionario {nombre: [emb1, emb2, ...]}.
    Devuelve siempre la distancia euclídea cruda y el umbral empleado (Sección 6 y 11).
    """
    vector_consulta = embedding if embedding is not None else embedding_actual
    if vector_consulta is None or not base_embeddings:
        return {
            "reconocido": False,
            "nombre": "Desconocido",
            "distancia": 9.99,
            "umbral": umbral,
            "candidato_mas_cercano": None,
        }

    emb_consulta = np.asarray(vector_consulta, dtype=np.float32).flatten()
    mejor_nombre: str | None = None
    menor_distancia: float = float("inf")

    for nombre, lista_vectores in base_embeddings.items():
        if not lista_vectores:
            continue
        dist = calcular_distancia_persona(emb_consulta, lista_vectores, metodo=metodo)
        if dist < menor_distancia:
            menor_distancia = dist
            mejor_nombre = nombre

    reconocido = menor_distancia < umbral and mejor_nombre is not None
    return {
        "reconocido": reconocido,
        "nombre": mejor_nombre if reconocido else "Desconocido",
        "distancia": round(menor_distancia, 4) if menor_distancia != float("inf") else 9.99,
        "umbral": umbral,
        "candidato_mas_cercano": mejor_nombre,
    }
