"""Pruebas unitarias para los módulos del paquete nucleo (LIS).

Verifica la base de datos SQLite en memoria, la lógica de matching
vectorizado, el soporte de múltiples embeddings, la derivación de estado
de asistencias/reingresos y la estabilización temporal de rostros.
"""

from pathlib import Path
import numpy as np
import pytest

from nucleo.base_datos import RepositorioLIS, normalizar_slug
from nucleo.comparacion import (
    calcular_distancia_persona,
    consultar_embeddings,
    existe_coincidencia,
)
from nucleo.estabilizador import EstabilizadorDeCaras
from nucleo.reconocimiento import ExtractorFaceNetRapido, recortar_con_margen


@pytest.fixture
def repo_en_memoria(tmp_path: Path) -> RepositorioLIS:
    """Provee un repositorio SQLite en un directorio temporal limpio."""
    ruta_db = tmp_path / "test_lis.db"
    ruta_caras_csv = tmp_path / "caras.csv"
    ruta_asistencia_csv = tmp_path / "asistencia.csv"
    ruta_ingresos_csv = tmp_path / "registro_de_ingresos.csv"

    return RepositorioLIS(
        ruta_db=ruta_db,
        ruta_caras_csv=ruta_caras_csv,
        ruta_asistencia_csv=ruta_asistencia_csv,
        ruta_ingresos_csv=ruta_ingresos_csv,
    )


def test_normalizar_slug():
    assert normalizar_slug("Juan Pérez") == "juan_perez"
    assert normalizar_slug("Nicolás") == "nicolas"
    assert normalizar_slug("  María José Gómez!  ") == "maria_jose_gomez"


def test_alta_persona_con_cinco_embeddings(repo_en_memoria: RepositorioLIS):
    """Test 1: Alta de persona con 5 embeddings en SQLite y verificación en espejo CSV."""
    np.random.seed(42)
    # Generar 5 embeddings sintéticos de 512 dimensiones
    muestras = [np.random.randn(512).astype(np.float32) for _ in range(5)]
    # Normalizar cada vector
    muestras = [v / np.linalg.norm(v) for v in muestras]

    persona_id = repo_en_memoria.registrar_persona_con_embeddings(
        nombre="Juan Pérez",
        embeddings=muestras,
        rol="integrante",
        consentimiento=True,
        modelo="vggface2",
        sincronizar_csv=True,
    )

    assert persona_id == "juan_perez"

    # Verificar que se recuperen las 5 muestras
    activos = repo_en_memoria.obtener_embeddings_activos(modelo="vggface2")
    assert "Juan Pérez" in activos
    assert len(activos["Juan Pérez"]) == 5
    for vector in activos["Juan Pérez"]:
        assert vector.shape == (512,)

    # Verificar archivo espejo caras.csv
    assert repo_en_memoria.ruta_caras_csv.exists()
    contenido_csv = repo_en_memoria.ruta_caras_csv.read_text(encoding="utf-8")
    assert "Juan Pérez" in contenido_csv


def test_comparacion_metodo_minimo_y_promedio():
    """Test 2: Comparación vectorizada por método mínimo y promedio."""
    np.random.seed(123)

    # Creamos un vector base y dos variaciones muy cercanas
    base = np.random.randn(512).astype(np.float32)
    base = base / np.linalg.norm(base)

    variacion1 = base + np.random.normal(0, 0.05, 512).astype(np.float32)
    variacion1 = variacion1 / np.linalg.norm(variacion1)

    variacion2 = base + np.random.normal(0, 0.05, 512).astype(np.float32)
    variacion2 = variacion2 / np.linalg.norm(variacion2)

    # Una muestra que se parece mucho a variacion1
    muestra_consulta = variacion1 + np.random.normal(0, 0.01, 512).astype(np.float32)
    muestra_consulta = muestra_consulta / np.linalg.norm(muestra_consulta)

    # Una persona no relacionada
    otra_persona = np.random.randn(512).astype(np.float32)
    otra_persona = otra_persona / np.linalg.norm(otra_persona)

    diccionario_personas = {
        "Carlos": [variacion1, variacion2],
        "Esteban": [otra_persona],
    }

    # Evaluación por mínimo
    res_min = consultar_embeddings(
        embedding_actual=muestra_consulta,
        base_embeddings=diccionario_personas,
        umbral=0.8,
        metodo="minimo",
    )
    assert res_min["reconocido"] is True
    assert res_min["nombre"] == "Carlos"
    assert res_min["distancia"] < 0.3

    # Evaluación por promedio
    res_prom = consultar_embeddings(
        embedding_actual=muestra_consulta,
        base_embeddings=diccionario_personas,
        umbral=0.8,
        metodo="promedio",
    )
    assert res_prom["reconocido"] is True
    assert res_prom["nombre"] == "Carlos"

    # Evaluación por top2
    res_top2 = consultar_embeddings(
        embedding_actual=muestra_consulta,
        base_embeddings=diccionario_personas,
        umbral=0.8,
        metodo="top2",
    )
    assert res_top2["reconocido"] is True
    assert res_top2["nombre"] == "Carlos"
    assert res_top2["distancia"] < 0.70

    # Evaluación con vector desconocido
    desconocido_vec = np.random.randn(512).astype(np.float32)
    desconocido_vec = desconocido_vec / np.linalg.norm(desconocido_vec)

    res_desc = consultar_embeddings(
        embedding_actual=desconocido_vec,
        base_embeddings=diccionario_personas,
        umbral=0.4,  # umbral estricto
    )
    assert res_desc["reconocido"] is False
    assert res_desc["nombre"] == "Desconocido"
    assert res_desc["distancia"] >= 0.4  # Informa la distancia cruda real


def test_margen_separacion_ratio_test():
    """Verifica que el margen de separación (Ratio Test) prevenga falsas aceptaciones por ambigüedad."""
    np.random.seed(42)
    base = np.random.randn(512).astype(np.float32)
    base = base / np.linalg.norm(base)

    # Dos personas sintéticas muy parecidas entre sí (ambigüedad)
    p1 = base + np.random.normal(0, 0.005, 512).astype(np.float32)
    p1 = p1 / np.linalg.norm(p1)

    p2 = base + np.random.normal(0, 0.008, 512).astype(np.float32)
    p2 = p2 / np.linalg.norm(p2)

    galeria = {
        "Persona1": [p1],
        "Persona2": [p2],
    }

    # Vector de consulta situado casi equidistante entre ambas (d1 ~ 0.50, d2 ~ 0.54)
    consulta = base + np.random.normal(0, 0.025, 512).astype(np.float32)
    consulta = consulta / np.linalg.norm(consulta)

    # Con margen 0.0, el candidato con menor distancia gana aunque la diferencia sea minúscula
    res_sin_margen = consultar_embeddings(
        embedding_actual=consulta,
        base_embeddings=galeria,
        umbral=0.70,
        margen_separacion=0.0,
    )
    assert res_sin_margen["reconocido"] is True

    # Con margen 0.10, la ambigüedad detectada rechaza el reconocimiento
    res_con_margen = consultar_embeddings(
        embedding_actual=consulta,
        base_embeddings=galeria,
        umbral=0.70,
        margen_separacion=0.10,
    )
    assert res_con_margen["reconocido"] is False
    assert res_con_margen["nombre"] == "Desconocido"
    assert res_con_margen["candidato_mas_cercano"] in ("Persona1", "Persona2")
    assert res_con_margen["margen"] is not None and res_con_margen["margen"] < 0.10

    # Si la distancia al 1er candidato es clarísima (< 0.45), debe reconocer aún con candidatos cercanos
    consulta_clarisima = p1 + np.random.normal(0, 0.01, 512).astype(np.float32)
    consulta_clarisima = consulta_clarisima / np.linalg.norm(consulta_clarisima)
    res_claro = consultar_embeddings(
        embedding_actual=consulta_clarisima,
        base_embeddings=galeria,
        umbral=0.62,
        margen_separacion=0.15,
    )
    assert res_claro["reconocido"] is True
    assert res_claro["nombre"] == "Persona1"
    assert res_claro["distancia"] < 0.45


def test_ciclo_reingreso_mismo_dia(repo_en_memoria: RepositorioLIS):
    """Test 3: Ciclo entrada -> salida -> reingreso en el mismo día (Event Sourcing)."""
    usuario = "Ana Gomez"

    # Inicialmente está afuera
    assert repo_en_memoria.obtener_estado_usuario(usuario) == "FUERA"
    assert not repo_en_memoria.esta_adentro(usuario)

    # 1. Primer ingreso a la mañana
    id_ev1 = repo_en_memoria.registrar_evento(
        tipo="entrada",
        nombre=usuario,
        metodo="cara+gesto",
        distancia=0.45,
    )
    assert id_ev1 is not None
    assert repo_en_memoria.esta_adentro(usuario)
    assert repo_en_memoria.obtener_conteo_presentes() == 1
    assert usuario in repo_en_memoria.obtener_presentes()

    # 2. Salida al mediodía
    id_ev2 = repo_en_memoria.registrar_evento(
        tipo="salida",
        nombre=usuario,
        metodo="cara+gesto",
        distancia=0.48,
    )
    assert id_ev2 is not None
    assert not repo_en_memoria.esta_adentro(usuario)
    assert repo_en_memoria.obtener_conteo_presentes() == 0

    # 3. Reingreso por la tarde
    id_ev3 = repo_en_memoria.registrar_evento(
        tipo="entrada",
        nombre=usuario,
        metodo="cara+gesto",
        distancia=0.42,
    )
    assert id_ev3 is not None
    assert repo_en_memoria.esta_adentro(usuario)
    assert repo_en_memoria.obtener_conteo_presentes() == 1

    # Verificar que los tres eventos existen en la base de datos (append-only)
    with repo_en_memoria._conectar() as conn:
        eventos = conn.execute(
            "SELECT tipo, distancia FROM eventos WHERE persona_id = ? ORDER BY t ASC;",
            (normalizar_slug(usuario),),
        ).fetchall()
        assert len(eventos) == 3
        tipos = [e["tipo"] for e in eventos]
        assert tipos == ["entrada", "salida", "entrada"]

    # Verificar que el CSV de ingresos preserva los registros
    assert repo_en_memoria.ruta_ingresos_csv.exists()
    lineas_ingresos = repo_en_memoria.ruta_ingresos_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas_ingresos) == 2  # Primer ciclo completado + segundo ciclo abierto


def test_recortar_con_margen():
    """Verifica que el recorte con margen del 10% no rompa límites de imagen."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    recorte = recortar_con_margen(img, origin_x=50, origin_y=50, width=100, height=100, margen_pct=0.10)
    assert recorte is not None
    # Con 10% de margen a cada lado, 100 + 20 = 120
    assert recorte.shape == (120, 120, 3)

    # Recorte en el borde superior izquierdo
    recorte_borde = recortar_con_margen(img, origin_x=0, origin_y=0, width=50, height=50, margen_pct=0.10)
    assert recorte_borde is not None
    assert recorte_borde.shape[0] <= 60 and recorte_borde.shape[1] <= 60


def test_estabilizador_de_caras():
    """Verifica la lógica temporal de la cara estable."""
    estabilizador = EstabilizadorDeCaras(frames_cara_estable=3, tolerancia_movimiento_px=20)

    # Cuadro 1: Detección inicial
    res1 = estabilizador.actualizar_pistas([(100, 100, 50, 50)])
    assert len(res1) == 1
    assert res1[0]["estable"] is False
    assert res1[0]["contador_estable"] == 1
    assert res1[0]["requiere_extraccion"] is False

    # Cuadro 2: Cara estable (movimiento mínimo)
    res2 = estabilizador.actualizar_pistas([(102, 101, 50, 50)])
    assert res2[0]["contador_estable"] == 2
    assert res2[0]["requiere_extraccion"] is False

    # Cuadro 3: Alcanza la meta de 3 frames estables -> requiere extracción
    res3 = estabilizador.actualizar_pistas([(101, 102, 50, 50)])
    assert res3[0]["contador_estable"] == 3
    assert res3[0]["estable"] is True
    assert res3[0]["requiere_extraccion"] is True

    # Simulamos que registramos la evaluación
    estabilizador.registrar_evaluacion(
        id_pista=res3[0]["id_pista"],
        nombre="Carlos",
        distancia=0.45,
        umbral=0.8,
        reconocido=True,
    )

    # Cuadro 4: Próximo frame utiliza los datos cacheados sin requerir extracción
    res4 = estabilizador.actualizar_pistas([(102, 102, 50, 50)])
    assert res4[0]["evaluado"] is True
    assert res4[0]["nombre"] == "Carlos"
    assert res4[0]["distancia"] == 0.45
    assert res4[0]["requiere_extraccion"] is False


def test_extractor_facenet_rapido_inferencia():
    """Verifica que ExtractorFaceNetRapido genere un vector de 512 dimensiones normalizado."""
    extractor = ExtractorFaceNetRapido(device="cpu")
    cara_sintetica = (np.random.rand(160, 160, 3) * 255).astype(np.uint8)

    embedding = extractor.generar_embedding(cara_sintetica)
    assert embedding is not None
    assert embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-3)

