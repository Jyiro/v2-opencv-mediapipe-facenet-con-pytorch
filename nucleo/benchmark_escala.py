"""Banco de pruebas aislado de escalabilidad y precisión multi-pose (LIS).

Evalúa latencia de carga SQLite, velocidad de consulta (bucle tradicional vs.
``IndiceEmbeddingsRAM``), tasas FAR/FRR y robustez Leave-One-Pose-Out de
perfiles reales del LIS mezclados en multitudes de 50 a 5.000+ identidades.

También soporta evaluación opcional con rostros reales multi-pose de LFW
(``--lfw``). Todas las operaciones de escritura se realizan en una base SQLite
temporal; nunca se modifica ``datos/lis_maestra.db``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any

import numpy as np

from nucleo.base_datos import RepositorioLIS, normalizar_slug
from nucleo.comparacion import IndiceEmbeddingsRAM, consultar_embeddings, normalizar_vector
from nucleo.eventos import ahora_iso

PROYECTO_DIR = Path(__file__).resolve().parent.parent
RUTA_DB_PRODUCTIVA = PROYECTO_DIR / "datos" / "lis_maestra.db"


@dataclass
class ResultadoEscala:
    personas: int
    vectores: int
    carga_sqlite_ms: float
    construccion_indice_ms: float
    latencia_bucle_ms: float
    latencia_indice_ms: float
    aceleracion: float
    tar_sintetico: float
    frr_sintetico: float
    far_impostores: float
    real_aciertos: int
    real_total: int
    real_dist_media: float | None
    real_segundo_media: float | None
    real_margen_medio: float | None


def generar_galeria_sintetica(
    cantidad_personas: int,
    muestras_por_persona: int = 5,
    dimension: int = 512,
    sigma_pose: float = 0.018,
    sigma_consulta: float = 0.020,
    cantidad_consultas: int = 40,
    semilla: int = 42,
) -> dict[str, Any]:
    """Genera identidades unitarias en R^512 con dispersión intra-persona realista."""
    rng = np.random.default_rng(semilla)

    centroides = rng.standard_normal((cantidad_personas, dimension), dtype=np.float32)
    normas_c = np.linalg.norm(centroides, axis=1, keepdims=True)
    centroides = centroides / np.maximum(normas_c, 1e-12)

    galeria: dict[str, list[np.ndarray]] = {}
    nombres: list[str] = []
    for i in range(cantidad_personas):
        nombre = f"Alumno Sintético {i + 1:04d}"
        nombres.append(nombre)
        ruido = rng.normal(0.0, sigma_pose, (muestras_por_persona, dimension)).astype(np.float32)
        poses = centroides[i][None, :] + ruido
        normas_p = np.linalg.norm(poses, axis=1, keepdims=True)
        poses = poses / np.maximum(normas_p, 1e-12)
        galeria[nombre] = [poses[k] for k in range(muestras_por_persona)]

    n_gen = min(cantidad_consultas, cantidad_personas)
    indices_q = rng.choice(cantidad_personas, size=n_gen, replace=False) if n_gen > 0 else []
    consultas_genuinas: list[tuple[str, np.ndarray]] = []
    for idx in indices_q:
        v = centroides[int(idx)] + rng.normal(0.0, sigma_consulta, dimension).astype(np.float32)
        consultas_genuinas.append((nombres[int(idx)], normalizar_vector(v)))

    impostores_mat = rng.standard_normal((cantidad_consultas, dimension), dtype=np.float32)
    normas_imp = np.linalg.norm(impostores_mat, axis=1, keepdims=True)
    impostores_mat = impostores_mat / np.maximum(normas_imp, 1e-12)
    consultas_impostores = [impostores_mat[i] for i in range(cantidad_consultas)]

    return {
        "galeria": galeria,
        "consultas_genuinas": consultas_genuinas,
        "consultas_impostores": consultas_impostores,
    }


def poblar_repositorio_temporal(
    repo: RepositorioLIS,
    galeria: dict[str, list[np.ndarray]],
    modelo: str = "vggface2",
) -> None:
    """Inserta en lote una galería en el SQLite temporal sin reescribir el CSV en cada fila."""
    ahora = ahora_iso()
    filas_personas: list[tuple[str, str, str, str, str, int, str, int]] = []
    filas_embeddings: list[tuple[str, bytes, int, str, str]] = []

    for idx, (nombre, vectores) in enumerate(galeria.items()):
        pid = f"{normalizar_slug(nombre)}_{idx:04d}"
        filas_personas.append((pid, nombre, "Alumno", "Aula-Test", "0000", 1, ahora, 1))
        for vec in vectores:
            arr = normalizar_vector(np.asarray(vec, dtype=np.float32).flatten()).astype(np.float32)
            filas_embeddings.append((pid, arr.tobytes(), int(arr.size), modelo, ahora))

    with repo._conectar() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO personas (id, nombre, rol, proyecto, pin, consentimiento, fecha_alta, activo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            filas_personas,
        )
        conn.executemany(
            """
            INSERT INTO embeddings (persona_id, vector, dimension, modelo, fecha_captura)
            VALUES (?, ?, ?, ?, ?)
            """,
            filas_embeddings,
        )


def cargar_perfiles_reales_solo_lectura(
    ruta_db: Path | None = None,
    modelo: str = "vggface2",
) -> dict[str, list[np.ndarray]]:
    """Lee perfiles activos con >=2 embeddings desde lis_maestra.db en modo estrictamente solo-lectura."""
    ruta = Path(ruta_db) if ruta_db is not None else RUTA_DB_PRODUCTIVA
    if not ruta.exists():
        return {}

    perfiles: dict[str, list[np.ndarray]] = {}
    conn = sqlite3.connect(f"file:{ruta}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        filas = conn.execute(
            """
            SELECT p.nombre, e.vector, e.dimension
            FROM embeddings e
            JOIN personas p ON p.id = e.persona_id
            WHERE p.activo = 1 AND e.modelo = ?
            ORDER BY e.id ASC
            """,
            (modelo,),
        ).fetchall()
        for f in filas:
            dim = int(f["dimension"] or 512)
            arr = np.frombuffer(f["vector"], dtype=np.float32).copy()
            if arr.size == dim:
                perfiles.setdefault(f["nombre"], []).append(normalizar_vector(arr))
    finally:
        conn.close()

    return {nombre: vecs for nombre, vecs in perfiles.items() if len(vecs) >= 2}


def evaluar_perfiles_reales_en_multitud(
    perfiles_reales: dict[str, list[np.ndarray]],
    galeria_multitud: dict[str, list[np.ndarray]],
    umbral: float = 0.68,
    metodo: str = "minimo",
    margen_separacion: float = 0.04,
) -> dict[str, Any]:
    """Evalúa Leave-One-Pose-Out de cada perfil real mezclado con una multitud."""
    if not perfiles_reales:
        return {
            "aciertos": 0,
            "total": 0,
            "tar": None,
            "distancia_media": None,
            "segundo_medio": None,
            "margen_medio": None,
        }

    aciertos = 0
    total = 0
    distancias: list[float] = []
    segundos: list[float] = []
    margenes: list[float] = []

    for nombre_real, poses in perfiles_reales.items():
        if len(poses) < 2:
            continue
        for fold_idx in range(len(poses)):
            consulta = poses[fold_idx]
            enroladas = [poses[i] for i in range(len(poses)) if i != fold_idx]
            galeria_fold = dict(galeria_multitud)
            galeria_fold[nombre_real] = enroladas
            indice = IndiceEmbeddingsRAM(galeria_fold)
            res = indice.consultar(
                consulta,
                umbral=umbral,
                metodo=metodo,
                margen_separacion=margen_separacion,
            )
            total += 1
            if res["reconocido"] and res["nombre"] == nombre_real:
                aciertos += 1
            distancias.append(float(res["distancia"]))
            if res["distancia_segundo"] is not None:
                segundos.append(float(res["distancia_segundo"]))
            if res["margen"] is not None:
                margenes.append(float(res["margen"]))

    return {
        "aciertos": aciertos,
        "total": total,
        "tar": round(aciertos / total, 4) if total else None,
        "distancia_media": round(float(np.mean(distancias)), 4) if distancias else None,
        "segundo_medio": round(float(np.mean(segundos)), 4) if segundos else None,
        "margen_medio": round(float(np.mean(margenes)), 4) if margenes else None,
    }


def _crear_repo_temporal(raiz: Path, nombre: str) -> RepositorioLIS:
    carpeta = raiz / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    return RepositorioLIS(
        ruta_db=carpeta / "temp_escala.db",
        ruta_caras_csv=carpeta / "caras.csv",
        ruta_asistencia_csv=carpeta / "asistencia.csv",
        ruta_ingresos_csv=carpeta / "registro_de_ingresos.csv",
    )


def _medir_tamano_escala(
    raiz: Path,
    tamano: int,
    *,
    muestras_por_persona: int,
    cantidad_consultas: int,
    umbral: float,
    metodo: str,
    margen_separacion: float,
    perfiles_reales: dict[str, list[np.ndarray]],
) -> ResultadoEscala:
    datos = generar_galeria_sintetica(
        cantidad_personas=tamano,
        muestras_por_persona=muestras_por_persona,
        cantidad_consultas=cantidad_consultas,
        semilla=42 + tamano,
    )
    repo = _crear_repo_temporal(raiz, f"escala_{tamano}")
    poblar_repositorio_temporal(repo, datos["galeria"], modelo="vggface2")

    t0 = time.perf_counter()
    galeria_sqlite = repo.cargar_diccionario_embeddings(modelo="vggface2", con_fallback=False)
    carga_sqlite_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    indice = IndiceEmbeddingsRAM(galeria_sqlite)
    construccion_indice_ms = (time.perf_counter() - t1) * 1000.0

    consultas_g = datos["consultas_genuinas"]
    consultas_imp = datos["consultas_impostores"]

    # Latencia bucle Python tradicional (acotada a 8 muestras en escalas grandes para agilidad)
    muestra_bucle = [v for _, v in consultas_g[: min(8, len(consultas_g))]]
    _ = consultar_embeddings(
        muestra_bucle[0],
        galeria_sqlite,
        umbral=umbral,
        metodo=metodo,
        margen_separacion=margen_separacion,
    )
    tb0 = time.perf_counter()
    for q in muestra_bucle:
        consultar_embeddings(
            q,
            galeria_sqlite,
            umbral=umbral,
            metodo=metodo,
            margen_separacion=margen_separacion,
        )
    latencia_bucle_ms = ((time.perf_counter() - tb0) / max(len(muestra_bucle), 1)) * 1000.0

    # Latencia + precisión completa con IndiceEmbeddingsRAM
    _ = indice.consultar(
        consultas_g[0][1],
        umbral=umbral,
        metodo=metodo,
        margen_separacion=margen_separacion,
    )
    aciertos_g = 0
    falsos_imp = 0
    ti0 = time.perf_counter()
    for esperado, q in consultas_g:
        res = indice.consultar(q, umbral=umbral, metodo=metodo, margen_separacion=margen_separacion)
        if res["reconocido"] and res["nombre"] == esperado:
            aciertos_g += 1
    for q_imp in consultas_imp:
        res_imp = indice.consultar(q_imp, umbral=umbral, metodo=metodo, margen_separacion=margen_separacion)
        if res_imp["reconocido"]:
            falsos_imp += 1
    total_q = max(len(consultas_g) + len(consultas_imp), 1)
    latencia_indice_ms = ((time.perf_counter() - ti0) / total_q) * 1000.0

    res_real = evaluar_perfiles_reales_en_multitud(
        perfiles_reales,
        galeria_sqlite,
        umbral=umbral,
        metodo=metodo,
        margen_separacion=margen_separacion,
    )

    n_g = max(len(consultas_g), 1)
    n_i = max(len(consultas_imp), 1)
    tar = aciertos_g / n_g
    frr = 1.0 - tar
    far = falsos_imp / n_i

    return ResultadoEscala(
        personas=indice.total_personas,
        vectores=indice.total_vectores,
        carga_sqlite_ms=round(carga_sqlite_ms, 2),
        construccion_indice_ms=round(construccion_indice_ms, 2),
        latencia_bucle_ms=round(latencia_bucle_ms, 3),
        latencia_indice_ms=round(latencia_indice_ms, 3),
        aceleracion=round(latencia_bucle_ms / max(latencia_indice_ms, 1e-6), 2),
        tar_sintetico=round(tar, 4),
        frr_sintetico=round(frr, 4),
        far_impostores=round(far, 4),
        real_aciertos=int(res_real["aciertos"]),
        real_total=int(res_real["total"]),
        real_dist_media=res_real["distancia_media"],
        real_segundo_media=res_real["segundo_medio"],
        real_margen_medio=res_real["margen_medio"],
    )


def ejecutar_benchmark_escala(
    tamanos: tuple[int, ...] = (50, 200, 500, 1000, 2500, 5000),
    muestras_por_persona: int = 5,
    cantidad_consultas: int = 40,
    umbral: float = 0.68,
    metodo: str = "minimo",
    margen_separacion: float = 0.04,
    directorio_trabajo: Path | None = None,
    incluir_reales: bool = True,
    ruta_db_reales: Path | None = None,
) -> list[ResultadoEscala]:
    """Ejecuta el benchmark completo de escala en una base SQLite temporal."""
    perfiles_reales = (
        cargar_perfiles_reales_solo_lectura(ruta_db_reales)
        if incluir_reales
        else {}
    )

    resultados: list[ResultadoEscala] = []
    if directorio_trabajo is not None:
        raiz = Path(directorio_trabajo)
        raiz.mkdir(parents=True, exist_ok=True)
        for n in tamanos:
            resultados.append(
                _medir_tamano_escala(
                    raiz,
                    int(n),
                    muestras_por_persona=muestras_por_persona,
                    cantidad_consultas=cantidad_consultas,
                    umbral=umbral,
                    metodo=metodo,
                    margen_separacion=margen_separacion,
                    perfiles_reales=perfiles_reales,
                )
            )
    else:
        with tempfile.TemporaryDirectory(prefix="lis_bench_") as tmpdir:
            raiz = Path(tmpdir)
            for n in tamanos:
                resultados.append(
                    _medir_tamano_escala(
                        raiz,
                        int(n),
                        muestras_por_persona=muestras_por_persona,
                        cantidad_consultas=cantidad_consultas,
                        umbral=umbral,
                        metodo=metodo,
                        margen_separacion=margen_separacion,
                        perfiles_reales=perfiles_reales,
                    )
                )
    return resultados


def ejecutar_benchmark_lfw(
    min_faces_per_person: int = 5,
    umbral: float = 0.68,
    metodo: str = "minimo",
    margen_separacion: float = 0.04,
) -> dict[str, Any]:
    """Descarga/carga LFW multi-pose y evalúa extracción real FaceNet en DB temporal."""
    import cv2 as cv
    from sklearn.datasets import fetch_lfw_people
    from nucleo.reconocimiento import ExtractorFaceNetRapido

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=True, resize=1.0)
    extractor = ExtractorFaceNetRapido(resolucion=160)

    embs_por_persona: dict[str, list[np.ndarray]] = {}
    for img, target in zip(lfw.images, lfw.target):
        nombre = str(lfw.target_names[int(target)])
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if float(np.max(arr)) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)
        emb = extractor.generar_embedding(bgr)
        if emb is not None:
            embs_por_persona.setdefault(nombre, []).append(emb)

    galeria: dict[str, list[np.ndarray]] = {}
    consultas_gen: list[tuple[str, np.ndarray]] = []
    consultas_imp: list[np.ndarray] = []

    nombres_ordenados = sorted(k for k, v in embs_por_persona.items() if len(v) >= 5)
    corte_imp = max(1, int(len(nombres_ordenados) * 0.85))
    enrolados_nombres = nombres_ordenados[:corte_imp]
    impostores_nombres = nombres_ordenados[corte_imp:]

    for nom in enrolados_nombres:
        vecs = embs_por_persona[nom]
        galeria[nom] = vecs[:4]
        for q in vecs[4:]:
            consultas_gen.append((nom, q))

    for nom_imp in impostores_nombres:
        consultas_imp.extend(embs_por_persona[nom_imp])

    with tempfile.TemporaryDirectory(prefix="lis_lfw_") as tmpdir:
        repo = _crear_repo_temporal(Path(tmpdir), "lfw")
        poblar_repositorio_temporal(repo, galeria, modelo="vggface2")
        galeria_sqlite = repo.cargar_diccionario_embeddings(modelo="vggface2", con_fallback=False)
        indice = IndiceEmbeddingsRAM(galeria_sqlite)

        aciertos = 0
        falsos_match = 0
        falsas_aceptaciones = 0
        t0 = time.perf_counter()
        for esperado, q in consultas_gen:
            res = indice.consultar(q, umbral=umbral, metodo=metodo, margen_separacion=margen_separacion)
            if res["reconocido"]:
                if res["nombre"] == esperado:
                    aciertos += 1
                else:
                    falsos_match += 1
        for q_imp in consultas_imp:
            res_imp = indice.consultar(q_imp, umbral=umbral, metodo=metodo, margen_separacion=margen_separacion)
            if res_imp["reconocido"]:
                falsas_aceptaciones += 1
        lat_ms = ((time.perf_counter() - t0) / max(len(consultas_gen) + len(consultas_imp), 1)) * 1000.0

    n_gen = max(len(consultas_gen), 1)
    n_imp = max(len(consultas_imp), 1)
    return {
        "personas_enroladas": len(galeria),
        "vectores_totales": sum(len(v) for v in galeria.values()),
        "consultas_genuinas": len(consultas_gen),
        "consultas_impostores": len(consultas_imp),
        "tar": round(aciertos / n_gen, 4),
        "frr": round(1.0 - (aciertos + falsos_match) / n_gen, 4),
        "far_impostores": round(falsas_aceptaciones / n_imp, 4),
        "latencia_indice_ms": round(lat_ms, 3),
    }


def formatear_tabla_markdown(resultados: list[ResultadoEscala]) -> str:
    """Devuelve una tabla Markdown lista para consola o ADR."""
    encabezado = (
        "| Personas | Vectores | Carga SQLite (ms) | Índice RAM (ms) | Bucle Python (ms) | "
        "IndiceEmbeddingsRAM (ms) | Aceleración | TAR Sint. | FAR Imp. | Real LOPO | Dist. Real | 2do Cand. | Margen Lowe |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|"
    )
    filas = [encabezado]
    for r in resultados:
        lopo = f"{r.real_aciertos}/{r.real_total}" if r.real_total else "-"
        d_real = f"{r.real_dist_media:.4f}" if r.real_dist_media is not None else "-"
        d_seg = f"{r.real_segundo_media:.4f}" if r.real_segundo_media is not None else "-"
        marg = f"{r.real_margen_medio:.4f}" if r.real_margen_medio is not None else "-"
        filas.append(
            f"| {r.personas} | {r.vectores} | {r.carga_sqlite_ms:.2f} | {r.construccion_indice_ms:.2f} | "
            f"{r.latencia_bucle_ms:.3f} | {r.latencia_indice_ms:.3f} | {r.aceleracion:.1f}x | "
            f"{r.tar_sintetico * 100:.1f}% | {r.far_impostores * 100:.1f}% | {lopo} | "
            f"{d_real} | {d_seg} | {marg} |"
        )
    return "\n".join(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Banco de pruebas aislado de escalabilidad LIS")
    parser.add_argument(
        "--tamanos",
        type=int,
        nargs="+",
        default=[50, 200, 500, 1000, 2500, 5000],
        help="Tamaños de galería (cantidad de personas) a evaluar.",
    )
    parser.add_argument("--umbral", type=float, default=0.68)
    parser.add_argument("--metodo", type=str, default="minimo")
    parser.add_argument("--margen", type=float, default=0.04)
    parser.add_argument("--consultas", type=int, default=40)
    parser.add_argument("--lfw", action="store_true", help="Ejecuta además evaluación con dataset LFW real.")
    args = parser.parse_args()

    print("=== Benchmark de Escala LIS (SQLite temporal aislado) ===")
    print(f"Parámetros: umbral={args.umbral}, metodo={args.metodo}, margen={args.margen}\n")
    resultados = ejecutar_benchmark_escala(
        tamanos=tuple(args.tamanos),
        cantidad_consultas=args.consultas,
        umbral=args.umbral,
        metodo=args.metodo,
        margen_separacion=args.margen,
    )
    print(formatear_tabla_markdown(resultados))

    if args.lfw:
        print("\n=== Evaluación Multi-Pose Real: LFW (fetch_lfw_people) ===")
        res_lfw = ejecutar_benchmark_lfw(
            umbral=args.umbral,
            metodo=args.metodo,
            margen_separacion=args.margen,
        )
        print(res_lfw)


if __name__ == "__main__":
    main()

