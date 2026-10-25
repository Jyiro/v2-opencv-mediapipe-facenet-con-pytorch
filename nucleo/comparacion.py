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

    matriz = np.vstack([normalizar_vector(np.asarray(v, dtype=np.float32).flatten()) for v in muestras])
    emb_c = normalizar_vector(np.asarray(emb, dtype=np.float32).flatten())

    if metodo == "promedio":
        centroide = normalizar_vector(np.mean(matriz, axis=0))
        return float(np.linalg.norm(centroide - emb_c))

    distancias = np.linalg.norm(matriz - emb_c, axis=1)

    if metodo in ("top2", "top2_promedio") and len(distancias) >= 2:
        # Promedio de las 2 muestras más cercanas + 20% de peso al centroide global
        dos_menores = np.sort(distancias)[:2]
        centroide = normalizar_vector(np.mean(matriz, axis=0))
        dist_centroide = float(np.linalg.norm(centroide - emb_c))
        return float(0.8 * np.mean(dos_menores) + 0.2 * dist_centroide)

    return float(np.min(distancias))


def existe_coincidencia(
    embedding_actual: np.ndarray,
    base_embeddings: dict[str, list[np.ndarray]] | list[np.ndarray],
    umbral: float = 0.62,
    metodo: str = "minimo",
    margen_separacion: float = 0.0,
) -> bool:
    if isinstance(base_embeddings, dict):
        res = consultar_embeddings(
            embedding_actual=embedding_actual,
            base_embeddings=base_embeddings,
            umbral=umbral,
            metodo=metodo,
            margen_separacion=margen_separacion,
        )
        return bool(res["reconocido"])
    elif isinstance(base_embeddings, list):
        d = calcular_distancia_persona(embedding_actual=embedding_actual, muestras_persona=base_embeddings, metodo=metodo)
        return bool(d < umbral)
    return False


def consultar_embeddings(
    embedding: np.ndarray | None = None,
    base_embeddings: dict[str, list[np.ndarray]] | None = None,
    umbral: float = 0.62,
    metodo: str = "minimo",
    embedding_actual: np.ndarray | None = None,
    margen_separacion: float = 0.0,
) -> dict[str, Any]:
    """
    Compara un embedding de 512 dimensiones contra el diccionario {nombre: [emb1, emb2, ...]}.
    Devuelve siempre la distancia euclídea cruda y el umbral empleado (Sección 6 y 11).
    Incorpora ratio test de Lowe (margen_separacion) para mitigar falsas aceptaciones.
    """
    vector_consulta = embedding if embedding is not None else embedding_actual
    if vector_consulta is None or not base_embeddings:
        return {
            "reconocido": False,
            "nombre": "Desconocido",
            "distancia": 9.99,
            "distancia_segundo": None,
            "margen": None,
            "umbral": umbral,
            "candidato_mas_cercano": None,
        }

    emb_consulta = normalizar_vector(np.asarray(vector_consulta, dtype=np.float32).flatten())
    ranking: list[tuple[float, str]] = []

    for nombre, lista_vectores in base_embeddings.items():
        if not lista_vectores:
            continue
        dist = calcular_distancia_persona(emb_consulta, lista_vectores, metodo=metodo)
        ranking.append((dist, nombre))

    if not ranking:
        return {
            "reconocido": False,
            "nombre": "Desconocido",
            "distancia": 9.99,
            "distancia_segundo": None,
            "margen": None,
            "umbral": umbral,
            "candidato_mas_cercano": None,
        }

    ranking.sort(key=lambda item: item[0])
    menor_distancia, mejor_nombre = ranking[0]
    segunda_distancia = ranking[1][0] if len(ranking) > 1 else float("inf")
    diferencia_segundo = (segunda_distancia - menor_distancia) if segunda_distancia != float("inf") else float("inf")

    # Pasa si está debajo del umbral Y supera el margen de ambigüedad (o es un match clarísimo < 0.45)
    supera_margen = (diferencia_segundo >= margen_separacion) or (menor_distancia < 0.45)
    reconocido = (menor_distancia < umbral) and supera_margen

    return {
        "reconocido": reconocido,
        "nombre": mejor_nombre if reconocido else "Desconocido",
        "distancia": round(menor_distancia, 4) if menor_distancia != float("inf") else 9.99,
        "distancia_segundo": round(segunda_distancia, 4) if segunda_distancia != float("inf") else None,
        "margen": round(diferencia_segundo, 4) if segunda_distancia != float("inf") else None,
        "umbral": umbral,
        "candidato_mas_cercano": mejor_nombre,
    }
