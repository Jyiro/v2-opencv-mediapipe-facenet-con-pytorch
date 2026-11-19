"""Pruebas unitarias para IndiceEmbeddingsRAM y el banco de pruebas a escala."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from nucleo.benchmark_escala import ejecutar_benchmark_escala
from nucleo.comparacion import IndiceEmbeddingsRAM, consultar_embeddings, normalizar_vector


def _vector_unitario(rng: np.random.Generator, dim: int = 512) -> np.ndarray:
    v = rng.standard_normal(dim, dtype=np.float32)
    return normalizar_vector(v)


@pytest.mark.parametrize("metodo", ["minimo", "promedio", "top2"])
def test_indice_ram_equivalencia_exacta(metodo: str):
    """IndiceEmbeddingsRAM debe devolver resultados idénticos a consultar_embeddings."""
    rng = np.random.default_rng(2026)

    base_a = _vector_unitario(rng)
    base_b = _vector_unitario(rng)
    base_c = normalizar_vector(base_a + rng.normal(0.0, 0.015, 512).astype(np.float32))

    # Mezcla de cantidades distintas por persona (5, 3 y 1 muestra) y una lista vacía
    galeria = {
        "Ana": [normalizar_vector(base_a + rng.normal(0.0, 0.01, 512).astype(np.float32)) for _ in range(5)],
        "Beto": [normalizar_vector(base_b + rng.normal(0.0, 0.01, 512).astype(np.float32)) for _ in range(3)],
        "Clara": [base_c],
        "Vacio": [],
    }
    indice = IndiceEmbeddingsRAM(galeria)

    # 1. Acierto claro sobre Beto
    q_beto = normalizar_vector(base_b + rng.normal(0.0, 0.008, 512).astype(np.float32))
    # 2. Caso ambiguo entre Ana y Clara (margen de Lowe pequeño)
    q_ambiguo = normalizar_vector(0.5 * base_a + 0.5 * base_c)
    # 3. Rechazo por umbral (impostor)
    q_impostor = _vector_unitario(rng)

    for consulta in (q_beto, q_ambiguo, q_impostor):
        ref = consultar_embeddings(
            consulta,
            galeria,
            umbral=0.68,
            metodo=metodo,
            margen_separacion=0.04,
        )
        rapido = indice.consultar(
            consulta,
            umbral=0.68,
            metodo=metodo,
            margen_separacion=0.04,
        )
        assert rapido == ref


def test_indice_ram_galeria_vacia():
    indice = IndiceEmbeddingsRAM({"SoloVacio": []})
    res = indice.consultar(np.ones(512, dtype=np.float32), umbral=0.68)
    assert res == {
        "reconocido": False,
        "nombre": "Desconocido",
        "distancia": 9.99,
        "distancia_segundo": None,
        "margen": None,
        "umbral": 0.68,
        "candidato_mas_cercano": None,
        "segundo_candidato": None,
    }


def test_benchmark_escala_aislado_y_rapido(tmp_path: Path):
    """El benchmark debe correr sobre tmp_path sin tocar datos productivos."""
    resultados = ejecutar_benchmark_escala(
        tamanos=(20, 100),
        cantidad_consultas=25,
        umbral=0.68,
        metodo="minimo",
        margen_separacion=0.04,
        directorio_trabajo=tmp_path / "aislado",
        incluir_reales=False,
    )

    assert len(resultados) == 2
    r20, r100 = resultados

    assert r20.personas == 20
    assert r20.vectores == 100
    assert r100.personas == 100
    assert r100.vectores == 500

    for res in resultados:
        assert res.far_impostores == 0.0
        assert res.tar_sintetico >= 0.90

    # Para 100 personas el índice matricial debe ser más rápido que el bucle Python y < 5 ms
    assert r100.latencia_indice_ms < r100.latencia_bucle_ms
    assert r100.latencia_indice_ms < 5.0

