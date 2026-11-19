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


def _resultado_desconocido(umbral: float) -> dict[str, Any]:
    return {
        "reconocido": False,
        "nombre": "Desconocido",
        "distancia": 9.99,
        "distancia_segundo": None,
        "margen": None,
        "umbral": umbral,
        "candidato_mas_cercano": None,
        "segundo_candidato": None,
    }


def _armar_resultado_consulta(
    ranking: list[tuple[float, str]],
    umbral: float,
    margen_separacion: float,
) -> dict[str, Any]:
    if not ranking:
        return _resultado_desconocido(umbral)

    ranking.sort(key=lambda item: item[0])
    menor_distancia, mejor_nombre = ranking[0]
    segunda_distancia = ranking[1][0] if len(ranking) > 1 else float("inf")
    segundo_nombre = ranking[1][1] if len(ranking) > 1 else None
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
        "segundo_candidato": segundo_nombre,
    }


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
        return _resultado_desconocido(umbral)

    emb_consulta = normalizar_vector(np.asarray(vector_consulta, dtype=np.float32).flatten())
    ranking: list[tuple[float, str]] = []

    for nombre, lista_vectores in base_embeddings.items():
        if not lista_vectores:
            continue
        dist = calcular_distancia_persona(emb_consulta, lista_vectores, metodo=metodo)
        ranking.append((dist, nombre))

    return _armar_resultado_consulta(ranking, umbral=umbral, margen_separacion=margen_separacion)


class IndiceEmbeddingsRAM:
    """Índice matricial pre-normalizado para consultas vectorizadas rápidas en RAM.

    Pre-apila y normaliza una sola vez todos los embeddings de la galería en matrices
    contiguas float32 para evitar llamadas repetidas a ``np.vstack`` en cada consulta.
    """

    def __init__(self, base_embeddings: dict[str, list[np.ndarray]] | None) -> None:
        self.nombres: list[str] = []
        self.muestras_por_persona: list[list[np.ndarray]] = []
        self._inicios_lista: list[int] = []
        self._conteos_lista: list[int] = []

        bloques: list[np.ndarray] = []
        centroides_lista: list[np.ndarray] = []
        offset = 0

        for nombre, lista_vectores in (base_embeddings or {}).items():
            if not lista_vectores:
                continue
            norm_persona = [
                normalizar_vector(np.asarray(v, dtype=np.float32).flatten())
                for v in lista_vectores
            ]
            bloque = np.vstack(norm_persona).astype(np.float32, copy=False)
            centroide = normalizar_vector(np.mean(bloque, axis=0)).astype(np.float32, copy=False)

            self.nombres.append(nombre)
            self.muestras_por_persona.append(norm_persona)
            self._inicios_lista.append(offset)
            self._conteos_lista.append(len(norm_persona))
            bloques.append(bloque)
            centroides_lista.append(centroide)
            offset += len(norm_persona)

        if not bloques:
            self.matriz = np.empty((0, 0), dtype=np.float32)
            self.centroides = np.empty((0, 0), dtype=np.float32)
            self.normas_cuad_matriz = np.empty((0,), dtype=np.float32)
            self.normas_cuad_centroides = np.empty((0,), dtype=np.float32)
            self.inicios = np.empty((0,), dtype=np.int64)
            self.conteos = np.empty((0,), dtype=np.int64)
            self.cantidad_uniforme: int | None = None
            return

        self.matriz = np.ascontiguousarray(np.vstack(bloques), dtype=np.float32)
        self.centroides = np.ascontiguousarray(np.vstack(centroides_lista), dtype=np.float32)
        self.normas_cuad_matriz = np.sum(self.matriz * self.matriz, axis=1, dtype=np.float32)
        self.normas_cuad_centroides = np.sum(self.centroides * self.centroides, axis=1, dtype=np.float32)
        self.inicios = np.asarray(self._inicios_lista, dtype=np.int64)
        self.conteos = np.asarray(self._conteos_lista, dtype=np.int64)
        unicos = set(self._conteos_lista)
        self.cantidad_uniforme = self._conteos_lista[0] if len(unicos) == 1 else None

    @property
    def total_personas(self) -> int:
        return len(self.nombres)

    @property
    def total_vectores(self) -> int:
        return int(self.matriz.shape[0])

    def _distancias_por_persona_aprox(self, emb_c: np.ndarray, metodo: str) -> np.ndarray:
        norma_emb_cuad = np.float32(np.dot(emb_c, emb_c))

        if metodo == "promedio":
            sim_cent = self.centroides @ emb_c
            dist_cuad = np.maximum(
                np.float32(0.0),
                self.normas_cuad_centroides + norma_emb_cuad - np.float32(2.0) * sim_cent,
            )
            return np.sqrt(dist_cuad)

        sim_all = self.matriz @ emb_c
        dist_all = np.sqrt(
            np.maximum(
                np.float32(0.0),
                self.normas_cuad_matriz + norma_emb_cuad - np.float32(2.0) * sim_all,
            )
        )

        if metodo in ("top2", "top2_promedio"):
            sim_cent = self.centroides @ emb_c
            dist_cent = np.sqrt(
                np.maximum(
                    np.float32(0.0),
                    self.normas_cuad_centroides + norma_emb_cuad - np.float32(2.0) * sim_cent,
                )
            )
            if self.cantidad_uniforme is not None and self.cantidad_uniforme >= 2:
                mat_d = dist_all.reshape(len(self.nombres), self.cantidad_uniforme)
                dos_menores = np.partition(mat_d, 1, axis=1)[:, :2]
                return 0.8 * np.mean(dos_menores, axis=1) + 0.2 * dist_cent

            dist_personas = np.empty(len(self.nombres), dtype=np.float32)
            for idx, (ini, cnt) in enumerate(zip(self._inicios_lista, self._conteos_lista)):
                sub = dist_all[ini : ini + cnt]
                if cnt >= 2:
                    dos = np.partition(sub, 1)[:2]
                    dist_personas[idx] = 0.8 * float(np.mean(dos)) + 0.2 * float(dist_cent[idx])
                else:
                    dist_personas[idx] = float(np.min(sub))
            return dist_personas

        return np.minimum.reduceat(dist_all, self.inicios)

    def consultar(
        self,
        embedding_actual: np.ndarray | None,
        umbral: float = 0.62,
        metodo: str = "minimo",
        margen_separacion: float = 0.0,
    ) -> dict[str, Any]:
        """Consulta el índice matricial manteniendo equivalencia con ``consultar_embeddings``."""
        if embedding_actual is None or len(self.nombres) == 0:
            return _resultado_desconocido(umbral)

        emb_c = normalizar_vector(np.asarray(embedding_actual, dtype=np.float32).flatten())
        dist_aprox = self._distancias_por_persona_aprox(emb_c, metodo=metodo)

        k_cand = min(16, len(self.nombres))
        if k_cand == len(self.nombres):
            indices_cand = range(len(self.nombres))
        else:
            indices_cand = np.argpartition(dist_aprox, k_cand - 1)[:k_cand].tolist()

        # Refina con cálculo L2 exacto sobre los mejores candidatos para evitar errores de redondeo
        ranking = [
            (
                calcular_distancia_persona(emb_c, self.muestras_por_persona[idx], metodo=metodo),
                self.nombres[idx],
            )
            for idx in indices_cand
        ]
        return _armar_resultado_consulta(ranking, umbral=umbral, margen_separacion=margen_separacion)

