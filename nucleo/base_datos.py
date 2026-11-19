from __future__ import annotations
import contextlib
import csv
import json
import logging
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
import numpy as np

from nucleo.comparacion import IndiceEmbeddingsRAM, normalizar_vector
from nucleo.eventos import ahora_iso, derivar_estado

try:
    import sqlite_vec as _SQLITE_VEC_MODULO  # type: ignore[import-not-found]
except Exception:
    _SQLITE_VEC_MODULO = None

PROYECTO_DIR = Path(__file__).resolve().parent.parent


def normalizar_slug(nombre: str) -> str:
    nfkd = unicodedata.normalize("NFKD", nombre.strip())
    sin_tildes = "".join([c for c in nfkd if not unicodedata.combining(c)])
    limpio = "".join(c.lower() if c.isalnum() else "_" for c in sin_tildes)
    while "__" in limpio:
        limpio = limpio.replace("__", "_")
    return limpio.strip("_") or "usuario"


INVENTARIO_INICIAL = [
    ("kit-grove-1", "Grove Starter Kit Plus #1", "Kit IoT"),
    ("kit-grove-2", "Grove Starter Kit Plus #2", "Kit IoT"),
    ("galileo-1", "Intel Galileo Gen 2 #1", "Placa Edge"),
    ("tablet-intel-1", "Tablet Intel x86 Pared", "Tablet"),
    ("arduino-uno-1", "Arduino UNO Clon CH340 #1", "Microcontrolador"),
    ("armario-llave", "Juego de Llaves Armario", "Seguridad"),
]


class RepositorioLIS:
    def __init__(
        self,
        ruta_db: Path | None = None,
        ruta_caras_csv: Path | None = None,
        ruta_asistencia_csv: Path | None = None,
        ruta_ingresos_csv: Path | None = None,
    ) -> None:
        self.ruta_db = ruta_db or (PROYECTO_DIR / "datos" / "lis_maestra.db")
        self.ruta_caras_csv = ruta_caras_csv or (PROYECTO_DIR / "mediapipe" / "caras" / "caras.csv")
        self.ruta_asistencia_csv = ruta_asistencia_csv or (PROYECTO_DIR / "mediapipe" / "asistencia" / "asistencia.csv")
        self.ruta_ingresos_csv = ruta_ingresos_csv or (PROYECTO_DIR / "mediapipe" / "manos" / "registro_de_ingresos.csv")

        self.ruta_db.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_caras_csv.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_asistencia_csv.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_ingresos_csv.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_vec_activo: bool = False

        self._inicializar_esquema()
        self._migrar_caras_csv_si_corresponde()
        if self.sqlite_vec_activo:
            self.sincronizar_indice_vec()

    @contextlib.contextmanager
    def _conectar(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.ruta_db), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        if _SQLITE_VEC_MODULO is not None and hasattr(conn, "enable_load_extension"):
            try:
                conn.enable_load_extension(True)
                _SQLITE_VEC_MODULO.load(conn)
            except Exception:
                logging.debug("No se pudo cargar sqlite_vec en la conexión.", exc_info=True)
            finally:
                try:
                    conn.enable_load_extension(False)
                except Exception:
                    pass
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _inicializar_esquema(self) -> None:
        with self._conectar() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    nombre TEXT UNIQUE NOT NULL,
                    rol TEXT NOT NULL DEFAULT 'Integrante',
                    proyecto TEXT NOT NULL DEFAULT 'LIS',
                    pin TEXT NOT NULL DEFAULT '0000',
                    consentimiento INTEGER NOT NULL DEFAULT 1,
                    fecha_alta TEXT NOT NULL,
                    activo INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL DEFAULT 512,
                    modelo TEXT NOT NULL DEFAULT 'vggface2',
                    fecha_captura TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS eventos (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL,
                    tipo TEXT NOT NULL,
                    persona_id TEXT,
                    objeto_id TEXT,
                    t TEXT NOT NULL,
                    reloj_confiable INTEGER NOT NULL DEFAULT 1,
                    origen TEXT NOT NULL,
                    metodo TEXT NOT NULL,
                    distancia REAL,
                    meta TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS objetos_inventario (
                    id TEXT PRIMARY KEY,
                    nombre TEXT NOT NULL,
                    categoria TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS estado_runtime (
                    clave TEXT PRIMARY KEY,
                    valor TEXT NOT NULL,
                    actualizado_t TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS telemetria_shadow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    t TEXT NOT NULL,
                    modo TEXT NOT NULL DEFAULT 'LIS',
                    persona_id TEXT,
                    metodo TEXT NOT NULL,
                    pistas_activas INTEGER NOT NULL DEFAULT 0,
                    dist_primero REAL,
                    dist_segundo REAL,
                    margen REAL,
                    calidad REAL,
                    brillo REAL,
                    yaw REAL,
                    pitch REAL,
                    decision_shadow TEXT NOT NULL,
                    meta TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            # Migración automática si la tabla eventos ya existía sin columna seq
            try:
                filas_cols = conn.execute("PRAGMA table_info(eventos);").fetchall()
                nombres_cols = [f["name"] for f in filas_cols]
                if nombres_cols and "seq" not in nombres_cols:
                    conn.executescript(
                        """
                        ALTER TABLE eventos RENAME TO eventos_antiguos;
                        CREATE TABLE eventos (
                            seq INTEGER PRIMARY KEY AUTOINCREMENT,
                            id TEXT UNIQUE NOT NULL,
                            tipo TEXT NOT NULL,
                            persona_id TEXT,
                            objeto_id TEXT,
                            t TEXT NOT NULL,
                            reloj_confiable INTEGER NOT NULL DEFAULT 1,
                            origen TEXT NOT NULL,
                            metodo TEXT NOT NULL,
                            distancia REAL,
                            meta TEXT NOT NULL DEFAULT '{}'
                        );
                        INSERT OR IGNORE INTO eventos (id, tipo, persona_id, objeto_id, t, reloj_confiable, origen, metodo, distancia, meta)
                        SELECT id, tipo, persona_id, objeto_id, t, reloj_confiable, origen, metodo, distancia, meta FROM eventos_antiguos;
                        DROP TABLE eventos_antiguos;
                        """
                    )
            except Exception:
                pass

            # Migración automática si la tabla personas no tenía pin o consentimiento
            try:
                filas_cols_p = conn.execute("PRAGMA table_info(personas);").fetchall()
                nombres_p = [f["name"] for f in filas_cols_p]
                if nombres_p and "pin" not in nombres_p:
                    conn.execute("ALTER TABLE personas ADD COLUMN pin TEXT NOT NULL DEFAULT '0000';")
                if nombres_p and "consentimiento" not in nombres_p:
                    conn.execute("ALTER TABLE personas ADD COLUMN consentimiento INTEGER NOT NULL DEFAULT 1;")
                if nombres_p and "proyecto" not in nombres_p:
                    conn.execute("ALTER TABLE personas ADD COLUMN proyecto TEXT NOT NULL DEFAULT 'LIS';")
            except Exception:
                pass

            # Migración idempotente de metadatos para slots discretos y TTL en embeddings
            try:
                cols_emb = {f["name"] for f in conn.execute("PRAGMA table_info(embeddings);").fetchall()}
                for col, definicion in (
                    ("tipo", "TEXT NOT NULL DEFAULT 'ancla'"),
                    ("origen", "TEXT NOT NULL DEFAULT 'enrolamiento'"),
                    ("expira_en", "TEXT"),
                    ("ultimo_uso", "TEXT"),
                    ("aciertos", "INTEGER NOT NULL DEFAULT 0"),
                ):
                    if cols_emb and col not in cols_emb:
                        conn.execute(f"ALTER TABLE embeddings ADD COLUMN {col} {definicion};")
            except Exception:
                pass

            if _SQLITE_VEC_MODULO is not None and hasattr(conn, "enable_load_extension"):
                try:
                    conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0("
                        "embedding_id INTEGER PRIMARY KEY, vector float[512]"
                        ");"
                    )
                    self.sqlite_vec_activo = True
                except Exception:
                    logging.warning("sqlite-vec no disponible al crear vec_embeddings; se usará RAM.", exc_info=True)
                    self.sqlite_vec_activo = False

    @staticmethod
    def normalizar_id(nombre: str) -> str:
        return normalizar_slug(nombre)

    def registrar_persona_con_embeddings(
        self,
        nombre: str,
        embeddings: list[np.ndarray],
        rol: str = "integrante",
        consentimiento: bool = True,
        modelo: str = "vggface2",
        sincronizar_csv: bool = True,
        proyecto: str = "LIS",
        pin: str = "0000",
    ) -> str:
        perfil = self.crear_o_actualizar_persona(
            nombre=nombre, rol=rol, proyecto=proyecto, pin=pin, consentimiento=consentimiento
        )
        self.guardar_embeddings_persona(
            nombre=nombre,
            embeddings=embeddings,
            modelo=modelo,
            reemplazar=True,
            sincronizar_csv=sincronizar_csv,
        )
        return perfil["id"]

    def obtener_embeddings_activos(self, modelo: str = "vggface2") -> dict[str, list[np.ndarray]]:
        return self.cargar_diccionario_embeddings(modelo=modelo)

    def _migrar_caras_csv_si_corresponde(self) -> None:
        with self._conectar() as conn:
            total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if total > 0 or not self.ruta_caras_csv.exists():
                return

        try:
            with open(self.ruta_caras_csv, "r", encoding="utf-8", newline="") as f:
                lector = csv.reader(f)
                vectores_por_nombre: dict[str, list[np.ndarray]] = {}
                for fila in lector:
                    if len(fila) < 513:
                        continue
                    nombre = fila[-1].strip()
                    if not nombre or nombre.lower() == "nombre":
                        continue
                    try:
                        vec = np.array([float(x) for x in fila[:512]], dtype=np.float32)
                        vectores_por_nombre.setdefault(nombre, []).append(vec)
                    except ValueError:
                        continue

            for nombre, lista_vec in vectores_por_nombre.items():
                self.crear_o_actualizar_persona(nombre=nombre, rol="Integrante", pin="1111", consentimiento=True)
                self.guardar_embeddings_persona(nombre=nombre, embeddings=lista_vec, reemplazar=True, sincronizar_csv=False)
            logging.info("Migración inicial desde caras.csv completada: %d perfiles.", len(vectores_por_nombre))
        except Exception:
            logging.exception("Error al migrar caras.csv inicial")

    def crear_o_actualizar_persona(
        self,
        nombre: str,
        rol: str = "Integrante",
        proyecto: str = "LIS",
        pin: str = "0000",
        consentimiento: bool = True,
    ) -> dict[str, Any]:
        nombre_limpio = nombre.strip()
        if not nombre_limpio:
            raise ValueError("El nombre no puede estar vacío.")
        pid = self.normalizar_id(nombre_limpio)
        ahora = ahora_iso()

        with self._conectar() as conn:
            existente = conn.execute(
                "SELECT id FROM personas WHERE lower(nombre) = lower(?) OR id = ?",
                (nombre_limpio, pid),
            ).fetchone()

            if existente:
                pid = existente["id"]
                conn.execute(
                    """
                    UPDATE personas
                    SET nombre = ?, rol = ?, proyecto = ?, pin = ?, consentimiento = ?, activo = 1
                    WHERE id = ?
                    """,
                    (nombre_limpio, rol, proyecto, pin, int(consentimiento), pid),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO personas (id, nombre, rol, proyecto, pin, consentimiento, fecha_alta, activo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (pid, nombre_limpio, rol, proyecto, pin, int(consentimiento), ahora),
                )
        return self.obtener_persona(nombre_limpio) or {}

    def guardar_embeddings_persona(
        self,
        nombre: str,
        embeddings: list[np.ndarray],
        modelo: str = "vggface2",
        reemplazar: bool = True,
        sincronizar_csv: bool = True,
        *,
        tipo: str = "ancla",
        origen: str = "enrolamiento",
        expira_en: str | None = None,
    ) -> int:
        perfil = self.obtener_persona(nombre)
        if not perfil:
            perfil = self.crear_o_actualizar_persona(nombre=nombre)
        pid = perfil["id"]
        ahora = ahora_iso()

        with self._conectar() as conn:
            if reemplazar:
                conn.execute("DELETE FROM embeddings WHERE persona_id = ? AND modelo = ?", (pid, modelo))
            for emb in embeddings:
                arr = np.asarray(emb, dtype=np.float32).flatten()
                conn.execute(
                    """
                    INSERT INTO embeddings (persona_id, vector, dimension, modelo, fecha_captura, tipo, origen, expira_en)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pid, arr.tobytes(), int(arr.size), modelo, ahora, tipo, origen, expira_en),
                )
            if modelo == "vggface2" and self.sqlite_vec_activo:
                self._sincronizar_vec_en_conexion(conn, modelo="vggface2")

        if sincronizar_csv:
            self.exportar_caras_csv()
        return len(embeddings)

    def eliminar_persona(self, nombre_o_id: str) -> bool:
        with self._conectar() as conn:
            cur = conn.execute(
                "DELETE FROM personas WHERE id = ? OR lower(nombre) = lower(?)",
                (nombre_o_id, nombre_o_id),
            )
            borrado = cur.rowcount > 0
            if borrado and self.sqlite_vec_activo:
                self._sincronizar_vec_en_conexion(conn, modelo="vggface2")
        if borrado:
            self.exportar_caras_csv()
        return borrado

    def obtener_persona(self, nombre_o_id: str) -> dict[str, Any] | None:
        with self._conectar() as conn:
            fila = conn.execute(
                """
                SELECT p.*, COUNT(e.id) AS cantidad_embeddings
                FROM personas p
                LEFT JOIN embeddings e ON e.persona_id = p.id AND e.modelo = 'vggface2'
                WHERE (p.id = ? OR lower(p.nombre) = lower(?)) AND p.activo = 1
                GROUP BY p.id
                """,
                (nombre_o_id, nombre_o_id),
            ).fetchone()
            return dict(fila) if fila else None

    def verificar_pin(self, nombre_o_id: str, pin: str) -> dict[str, Any] | None:
        perfil = self.obtener_persona(nombre_o_id)
        if perfil and str(perfil.get("pin", "")) == str(pin).strip():
            return perfil
        return None

    def buscar_por_pin(self, pin: str) -> tuple[dict[str, Any] | None, str | None]:
        """Busca un integrante activo únicamente por su PIN personal."""
        pin_limpio = str(pin).strip()
        if not pin_limpio:
            return None, "Ingresá un código PIN."
        with self._conectar() as conn:
            filas = conn.execute(
                """
                SELECT p.*, COUNT(e.id) AS cantidad_embeddings
                FROM personas p
                LEFT JOIN embeddings e ON e.persona_id = p.id AND e.modelo = 'vggface2'
                WHERE p.pin = ? AND p.activo = 1
                GROUP BY p.id
                """,
                (pin_limpio,),
            ).fetchall()
        if not filas:
            return None, "PIN incorrecto."
        if len(filas) > 1:
            return None, "PIN compartido por varios perfiles; asigná un PIN único en Padrón."
        return dict(filas[0]), None

    def listar_personas(self) -> list[dict[str, Any]]:
        with self._conectar() as conn:
            filas = conn.execute(
                """
                SELECT p.id, p.nombre, p.rol, p.proyecto, p.pin, p.consentimiento,
                       p.fecha_alta, COUNT(e.id) AS cantidad_embeddings
                FROM personas p
                LEFT JOIN embeddings e ON e.persona_id = p.id AND e.modelo = 'vggface2'
                WHERE p.activo = 1
                GROUP BY p.id
                ORDER BY p.nombre ASC
                """
            ).fetchall()
            return [dict(f) for f in filas]

    def cargar_diccionario_embeddings(
        self,
        modelo: str = "vggface2",
        con_fallback: bool = True,
        proyecto: str | None = None,
        solo_anclas: bool = False,
    ) -> dict[str, list[np.ndarray]]:
        ahora = ahora_iso()
        filtros = [
            "p.activo = 1",
            "(e.expira_en IS NULL OR e.expira_en > ?)",
        ]
        params: list[Any] = [ahora]
        if proyecto is not None:
            filtros.append("p.proyecto = ?")
            params.append(str(proyecto))
        if solo_anclas:
            filtros.append("COALESCE(e.tipo, 'ancla') = 'ancla'")

        where_sql = " AND ".join(filtros)
        resultado: dict[str, list[np.ndarray]] = {}
        with self._conectar() as conn:
            filas = conn.execute(
                f"""
                SELECT p.nombre, e.vector, e.dimension, e.modelo
                FROM embeddings e
                JOIN personas p ON p.id = e.persona_id
                WHERE {where_sql}
                ORDER BY e.id ASC
                """,
                params,
            ).fetchall()

            por_modelo: dict[str, dict[str, list[np.ndarray]]] = {}
            for f in filas:
                nom = f["nombre"]
                mod = f["modelo"]
                dim = int(f["dimension"] or 512)
                arr = np.frombuffer(f["vector"], dtype=np.float32).copy()
                if arr.size == dim:
                    por_modelo.setdefault(mod, {}).setdefault(nom, []).append(arr)

            resultado = dict(por_modelo.get(modelo, {}))
            if con_fallback and modelo != "vggface2":
                base_ref = por_modelo.get("vggface2", {})
                for nom, lista in base_ref.items():
                    if nom not in resultado or len(resultado[nom]) == 0:
                        resultado[nom] = list(lista)

        return resultado

    def cargar_anclas_persona(self, nombre_o_id: str, modelo: str = "vggface2") -> list[np.ndarray]:
        """Devuelve únicamente las anclas base inmutables de una persona."""
        with self._conectar() as conn:
            filas = conn.execute(
                """
                SELECT e.vector, e.dimension
                FROM embeddings e
                JOIN personas p ON p.id = e.persona_id
                WHERE (p.id = ? OR lower(p.nombre) = lower(?))
                  AND p.activo = 1
                  AND e.modelo = ?
                  AND COALESCE(e.tipo, 'ancla') = 'ancla'
                ORDER BY e.id ASC
                """,
                (nombre_o_id, nombre_o_id, modelo),
            ).fetchall()
        anclas: list[np.ndarray] = []
        for f in filas:
            dim = int(f["dimension"] or 512)
            arr = np.frombuffer(f["vector"], dtype=np.float32).copy()
            if arr.size == dim:
                anclas.append(normalizar_vector(arr).astype(np.float32, copy=False))
        return anclas

    def _sincronizar_vec_en_conexion(self, conn: sqlite3.Connection, modelo: str = "vggface2") -> int:
        if not self.sqlite_vec_activo:
            return 0
        ahora = ahora_iso()
        conn.execute("DELETE FROM vec_embeddings;")
        filas = conn.execute(
            """
            SELECT e.id, e.vector, e.dimension
            FROM embeddings e
            JOIN personas p ON p.id = e.persona_id
            WHERE p.activo = 1
              AND e.modelo = ?
              AND e.dimension = 512
              AND (e.expira_en IS NULL OR e.expira_en > ?)
            ORDER BY e.id ASC
            """,
            (modelo, ahora),
        ).fetchall()
        lote: list[tuple[int, bytes]] = []
        for f in filas:
            arr = np.frombuffer(f["vector"], dtype=np.float32).copy()
            if arr.size == 512:
                arr_n = normalizar_vector(arr).astype(np.float32, copy=False)
                lote.append((int(f["id"]), arr_n.tobytes()))
        if lote:
            conn.executemany(
                "INSERT INTO vec_embeddings (embedding_id, vector) VALUES (?, ?)",
                lote,
            )
        return len(lote)

    def sincronizar_indice_vec(self, modelo: str = "vggface2") -> int:
        """Sincroniza vec_embeddings (sqlite-vec) con embeddings activos y no expirados."""
        if not self.sqlite_vec_activo:
            return 0
        with self._conectar() as conn:
            return self._sincronizar_vec_en_conexion(conn, modelo=modelo)

    def consultar_knn_sqlite_vec(
        self,
        embedding_consulta: np.ndarray,
        *,
        k: int = 10,
        modelo: str = "vggface2",
        proyecto: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ejecuta búsqueda KNN nativa con sqlite-vec (o fallback en RAM si no está disponible)."""
        emb_norm = normalizar_vector(np.asarray(embedding_consulta, dtype=np.float32).flatten()).astype(
            np.float32, copy=False
        )
        limite = max(1, int(k))

        if not self.sqlite_vec_activo or modelo != "vggface2" or emb_norm.size != 512:
            galeria = self.cargar_diccionario_embeddings(
                modelo=modelo, con_fallback=False, proyecto=proyecto
            )
            resultados_ram: list[dict[str, Any]] = []
            for nombre, vecs in galeria.items():
                for v in vecs:
                    v_n = normalizar_vector(np.asarray(v, dtype=np.float32).flatten())
                    if v_n.size == emb_norm.size:
                        dist = float(np.linalg.norm(v_n - emb_norm))
                        resultados_ram.append(
                            {
                                "embedding_id": None,
                                "nombre": nombre,
                                "proyecto": proyecto,
                                "tipo": "ancla",
                                "distancia": round(dist, 6),
                            }
                        )
            resultados_ram.sort(key=lambda item: item["distancia"])
            return resultados_ram[:limite]

        ahora = ahora_iso()
        with self._conectar() as conn:
            total_vec = int(conn.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()[0])
            if total_vec == 0:
                return []
            k_vec = min(total_vec, max(limite * 20, 200) if proyecto is not None else limite)
            filtros = [
                "v.vector MATCH ?",
                "k = ?",
                "p.activo = 1",
                "e.modelo = ?",
                "(e.expira_en IS NULL OR e.expira_en > ?)",
            ]
            params: list[Any] = [emb_norm.tobytes(), int(k_vec), modelo, ahora]
            if proyecto is not None:
                filtros.append("p.proyecto = ?")
                params.append(str(proyecto))
            params.append(limite)

            filas = conn.execute(
                f"""
                SELECT v.embedding_id, v.distance AS distancia, p.nombre, p.proyecto,
                       COALESCE(e.tipo, 'ancla') AS tipo
                FROM vec_embeddings v
                JOIN embeddings e ON e.id = v.embedding_id
                JOIN personas p ON p.id = e.persona_id
                WHERE {' AND '.join(filtros)}
                ORDER BY v.distance ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            {
                "embedding_id": int(f["embedding_id"]),
                "nombre": f["nombre"],
                "proyecto": f["proyecto"],
                "tipo": f["tipo"],
                "distancia": round(float(f["distancia"]), 6),
            }
            for f in filas
        ]

    def agregar_slot_adaptativo(
        self,
        nombre_o_id: str,
        embedding: np.ndarray,
        *,
        modelo: str = "vggface2",
        origen: str = "actualizacion_explicita",
        ttl_dias: float = 120.0,
        max_slots: int = 3,
        expira_en: str | None = None,
    ) -> int:
        """Agrega un slot adaptativo discreto aplicando reemplazo LRU y TTL sin tocar anclas."""
        arr = normalizar_vector(np.asarray(embedding, dtype=np.float32).flatten()).astype(
            np.float32, copy=False
        )
        ahora = ahora_iso()
        if expira_en is None:
            expira_en = (datetime.now(timezone.utc) + timedelta(days=float(ttl_dias))).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        with self._conectar() as conn:
            persona = conn.execute(
                "SELECT id FROM personas WHERE (id = ? OR lower(nombre) = lower(?)) AND activo = 1",
                (nombre_o_id, nombre_o_id),
            ).fetchone()
            if not persona:
                raise ValueError(f"Persona no encontrada o inactiva: {nombre_o_id}")
            pid = persona["id"]

            # Purgar adaptativos vencidos
            conn.execute(
                """
                DELETE FROM embeddings
                WHERE persona_id = ? AND modelo = ? AND COALESCE(tipo, 'ancla') = 'adaptativo'
                  AND expira_en IS NOT NULL AND expira_en <= ?
                """,
                (pid, modelo, ahora),
            )

            # Política LRU sobre slots adaptativos activos (nunca toca anclas)
            existentes = conn.execute(
                """
                SELECT id FROM embeddings
                WHERE persona_id = ? AND modelo = ? AND COALESCE(tipo, 'ancla') = 'adaptativo'
                ORDER BY COALESCE(ultimo_uso, fecha_captura) ASC, id ASC
                """,
                (pid, modelo),
            ).fetchall()
            exceso = len(existentes) - max(1, int(max_slots)) + 1
            if exceso > 0:
                ids_borrar = [int(f["id"]) for f in existentes[:exceso]]
                placeholders = ",".join("?" * len(ids_borrar))
                conn.execute(f"DELETE FROM embeddings WHERE id IN ({placeholders})", ids_borrar)

            cur = conn.execute(
                """
                INSERT INTO embeddings
                (persona_id, vector, dimension, modelo, fecha_captura, tipo, origen, expira_en, ultimo_uso, aciertos)
                VALUES (?, ?, ?, ?, ?, 'adaptativo', ?, ?, ?, 1)
                """,
                (pid, arr.tobytes(), int(arr.size), modelo, ahora, origen, expira_en, ahora),
            )
            nuevo_id = int(cur.lastrowid or 0)
            if modelo == "vggface2" and self.sqlite_vec_activo:
                self._sincronizar_vec_en_conexion(conn, modelo="vggface2")
        return nuevo_id

    def revertir_a_anclas(
        self,
        nombre_o_id: str | None = None,
        modelo: str | None = None,
    ) -> int:
        """Elimina los slots adaptativos dejando intactas las anclas base (rollback sin fotos)."""
        filtros = ["COALESCE(tipo, 'ancla') = 'adaptativo'"]
        params: list[Any] = []
        with self._conectar() as conn:
            if nombre_o_id:
                persona = conn.execute(
                    "SELECT id FROM personas WHERE id = ? OR lower(nombre) = lower(?)",
                    (nombre_o_id, nombre_o_id),
                ).fetchone()
                if not persona:
                    return 0
                filtros.append("persona_id = ?")
                params.append(persona["id"])
            if modelo:
                filtros.append("modelo = ?")
                params.append(modelo)

            cur = conn.execute(f"DELETE FROM embeddings WHERE {' AND '.join(filtros)}", params)
            eliminados = int(cur.rowcount or 0)
            if eliminados > 0 and self.sqlite_vec_activo:
                self._sincronizar_vec_en_conexion(conn, modelo="vggface2")
        return eliminados

    def registrar_telemetria_shadow(
        self,
        *,
        modo: str = "LIS",
        persona_id: str | None = None,
        metodo: str = "cara+gesto",
        pistas_activas: int = 0,
        dist_primero: float | None = None,
        dist_segundo: float | None = None,
        margen: float | None = None,
        calidad: float | None = None,
        brillo: float | None = None,
        yaw: float | None = None,
        pitch: float | None = None,
        decision_shadow: str = "observacion",
        meta: dict[str, Any] | None = None,
        t: str | None = None,
    ) -> int:
        """Registra una observación numérica liviana en telemetria_shadow (sin fotos)."""
        with self._conectar() as conn:
            cur = conn.execute(
                """
                INSERT INTO telemetria_shadow
                (t, modo, persona_id, metodo, pistas_activas, dist_primero, dist_segundo, margen,
                 calidad, brillo, yaw, pitch, decision_shadow, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t or ahora_iso(),
                    modo,
                    persona_id,
                    metodo,
                    int(pistas_activas),
                    float(dist_primero) if dist_primero is not None else None,
                    float(dist_segundo) if dist_segundo is not None else None,
                    float(margen) if margen is not None else None,
                    float(calidad) if calidad is not None else None,
                    float(brillo) if brillo is not None else None,
                    float(yaw) if yaw is not None else None,
                    float(pitch) if pitch is not None else None,
                    decision_shadow,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid or 0)

    def listar_telemetria_shadow(self, limite: int = 100) -> list[dict[str, Any]]:
        with self._conectar() as conn:
            filas = conn.execute(
                "SELECT * FROM telemetria_shadow ORDER BY id DESC LIMIT ?",
                (max(1, int(limite)),),
            ).fetchall()
        res: list[dict[str, Any]] = []
        for f in filas:
            item = dict(f)
            try:
                item["meta"] = json.loads(item.get("meta") or "{}")
            except Exception:
                item["meta"] = {}
            res.append(item)
        return res

    def exportar_caras_csv(self) -> None:
        """Reescribe caras.csv de forma limpia desde SQLite (sin duplicados basura)."""
        base = self.cargar_diccionario_embeddings()
        try:
            with open(self.ruta_caras_csv, "w", encoding="utf-8", newline="") as f:
                escritor = csv.writer(f)
                for nombre, vectores in base.items():
                    for vec in vectores:
                        fila = [f"{ float(x):.6f}" for x in vec] + [nombre]
                        escritor.writerow(fila)
        except Exception:
            logging.exception("Error al exportar espejo caras.csv")

    def registrar_evento(
        self,
        evento: dict[str, Any] | None = None,
        *,
        tipo: str | None = None,
        nombre: str | None = None,
        persona_id: str | None = None,
        objeto_id: str | None = None,
        metodo: str = "cara+gesto",
        origen: str = "pipeline-facial",
        distancia: float | None = None,
        meta: dict[str, Any] | None = None,
        reloj_confiable: bool = True,
        proyecto: str = "LIS",
        **kwargs: Any,
    ) -> bool | str:
        if evento is None:
            tipo_real = tipo or kwargs.get("tipo", "entrada")
            id_persona = persona_id or (normalizar_slug(nombre) if nombre else None)
            meta_dict = dict(meta or {})
            if nombre:
                meta_dict.setdefault("nombre", nombre)
            meta_dict.setdefault("proyecto", proyecto)
            evento = {
                "id": str(uuid.uuid4()),
                "tipo": tipo_real,
                "persona_id": id_persona,
                "objeto_id": objeto_id,
                "t": ahora_iso(),
                "reloj_confiable": bool(reloj_confiable),
                "origen": origen,
                "metodo": metodo,
                "distancia": float(distancia) if distancia is not None else None,
                "meta": meta_dict,
            }
            es_invocacion_legacy = True
        else:
            es_invocacion_legacy = False

        pid = evento.get("persona_id")
        ev_meta = evento.get("meta") or {}
        if pid and not ev_meta.get("nombre"):
            if " " in pid:
                ev_meta["nombre"] = pid
                pid = normalizar_slug(pid)
                evento["persona_id"] = pid
                evento["meta"] = ev_meta

        with self._conectar() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO eventos
                (id, tipo, persona_id, objeto_id, t, reloj_confiable, origen, metodo, distancia, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evento["id"],
                    evento["tipo"],
                    evento.get("persona_id"),
                    evento.get("objeto_id"),
                    evento["t"],
                    int(evento.get("reloj_confiable", True)),
                    evento["origen"],
                    evento["metodo"],
                    evento.get("distancia"),
                    json.dumps(evento.get("meta") or {}, ensure_ascii=False),
                ),
            )
            insertado = cur.rowcount > 0

        if insertado and evento["tipo"] in ("entrada", "salida"):
            self._sincronizar_csv_asistencia_e_ingresos()

        if es_invocacion_legacy:
            return evento["id"]
        return insertado

    def listar_eventos(self, desde_seq: int = 0, limite: int = 500) -> list[dict[str, Any]]:
        with self._conectar() as conn:
            filas = conn.execute(
                "SELECT * FROM eventos WHERE seq > ? ORDER BY seq ASC LIMIT ?",
                (desde_seq, limite),
            ).fetchall()
            eventos = []
            for f in filas:
                eventos.append(
                    {
                        "seq": f["seq"],
                        "id": f["id"],
                        "tipo": f["tipo"],
                        "persona_id": f["persona_id"],
                        "objeto_id": f["objeto_id"],
                        "t": f["t"],
                        "reloj_confiable": bool(f["reloj_confiable"]),
                        "origen": f["origen"],
                        "metodo": f["metodo"],
                        "distancia": f["distancia"],
                        "meta": json.loads(f["meta"] or "{}"),
                    }
                )
            return eventos

    def _sincronizar_csv_asistencia_e_ingresos(self) -> None:
        """Mantiene actualizados asistencia.csv y registro_de_ingresos.csv soportando múltiples reingresos."""
        eventos = self.listar_eventos(desde_seq=0, limite=5000)
        sesiones: list[dict[str, str]] = []
        abiertas_por_usuario: dict[str, dict[str, str]] = {}
        filas_asistencia: list[list[str]] = []

        for ev in eventos:
            tipo = ev["tipo"]
            usuario = (ev.get("meta") or {}).get("nombre") or ev.get("persona_id") or ""
            if not usuario or tipo not in ("entrada", "salida"):
                continue

            ts_str = ev["t"]
            fecha = ts_str[:10]
            hora = ts_str[11:19] if len(ts_str) >= 19 else ts_str
            hora_corta = hora[:5] if len(hora) >= 5 else hora
            proyecto = (ev.get("meta") or {}).get("proyecto", "LIS")

            if tipo == "entrada":
                nueva_sesion = {
                    "fecha": fecha,
                    "usuario": usuario,
                    "proyecto": proyecto,
                    "ingreso": hora_corta,
                    "egreso": "N/A",
                }
                sesiones.append(nueva_sesion)
                abiertas_por_usuario[usuario] = nueva_sesion
                filas_asistencia.append([fecha, usuario, proyecto, hora_corta])
            elif tipo == "salida":
                sesion_abierta = abiertas_por_usuario.pop(usuario, None)
                if sesion_abierta:
                    sesion_abierta["egreso"] = hora_corta
                else:
                    sesiones.append({
                        "fecha": fecha,
                        "usuario": usuario,
                        "proyecto": proyecto,
                        "ingreso": "N/A",
                        "egreso": hora_corta,
                    })

        try:
            with open(self.ruta_ingresos_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                for s in sesiones:
                    w.writerow([s["fecha"], s["usuario"], s["proyecto"], s["ingreso"], s["egreso"]])

            with open(self.ruta_asistencia_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                for fila in filas_asistencia:
                    w.writerow(fila)
        except Exception:
            logging.exception("Error al sincronizar CSVs de asistencia/ingresos")

    def registrar_asistencia_permanencia(
        self, nombre: str, proyecto: str = "LIS", distancia: float | None = None
    ) -> str:
        return str(self.registrar_evento(
            tipo="asistencia_permanencia",
            nombre=nombre,
            metodo="cara_pasiva",
            origen="pipeline-facial",
            distancia=distancia,
            proyecto=proyecto,
        ))

    def obtener_estado_usuario(self, nombre_o_id: str) -> str:
        slug = normalizar_slug(nombre_o_id)
        consulta = """
            SELECT tipo FROM eventos
            WHERE (persona_id = ? OR lower(persona_id) = lower(?) OR json_extract(meta, '$.nombre') = ?) AND tipo IN ('entrada', 'salida')
            ORDER BY seq DESC
            LIMIT 1;
        """
        with self._conectar() as conn:
            fila = conn.execute(consulta, (slug, nombre_o_id, nombre_o_id)).fetchone()
            if fila and fila["tipo"] == "entrada":
                return "DENTRO"
        return "FUERA"

    def esta_adentro(self, nombre_o_id: str) -> bool:
        return self.obtener_estado_usuario(nombre_o_id) == "DENTRO"

    def obtener_presentes(self) -> set[str]:
        eventos = self.listar_eventos(desde_seq=0, limite=5000)
        estado = derivar_estado(eventos)
        presentes_dict = estado.get("presentes", {})
        res: set[str] = set()
        for k, v in presentes_dict.items():
            nombre = v.get("nombre") or k
            res.add(nombre)
            res.add(k)
        return res

    def obtener_conteo_presentes(self) -> int:
        eventos = self.listar_eventos(desde_seq=0, limite=5000)
        estado = derivar_estado(eventos)
        return len(estado.get("presentes", {}))

    def listar_inventario(self) -> list[dict[str, Any]]:
        with self._conectar() as conn:
            filas = conn.execute("SELECT * FROM objetos_inventario ORDER BY id ASC").fetchall()
            return [dict(f) for f in filas]

    def guardar_estado_runtime(self, clave: str, valor: dict[str, Any]) -> None:
        with self._conectar() as conn:
            conn.execute(
                """
                INSERT INTO estado_runtime (clave, valor, actualizado_t)
                VALUES (?, ?, ?)
                ON CONFLICT(clave) DO UPDATE SET valor = excluded.valor, actualizado_t = excluded.actualizado_t
                """,
                (clave, json.dumps(valor, ensure_ascii=False), ahora_iso()),
            )

    def leer_estado_runtime(self, clave: str, defecto: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._conectar() as conn:
            fila = conn.execute("SELECT valor FROM estado_runtime WHERE clave = ?", (clave,)).fetchone()
            if not fila:
                return defecto or {}
            try:
                return json.loads(fila["valor"])
            except Exception:
                return defecto or {}

    def leer_todos_estados_runtime(self) -> dict[str, dict[str, Any]]:
        """Lee todas las claves de estado_runtime en una única consulta SQLite."""
        resultado: dict[str, dict[str, Any]] = {}
        with self._conectar() as conn:
            filas = conn.execute("SELECT clave, valor FROM estado_runtime").fetchall()
            for f in filas:
                try:
                    resultado[f["clave"]] = json.loads(f["valor"])
                except Exception:
                    pass
        return resultado

    def solicitar_enrolamiento_camara(
        self,
        nombre: str,
        muestras: int = 5,
        guiado: bool = False,
        poses: int = 5,
        modo: str = "normal",
    ) -> dict[str, Any]:
        total = int(poses) if guiado else int(muestras)
        orden = {
            "activo": True,
            "nombre": nombre.strip(),
            "muestras_totales": total,
            "muestras_capturadas": 0,
            "estado": "pendiente",
            "mensaje": f"Posicione a {nombre.strip()} frente a la cámara...",
            "solicitado_t": ahora_iso(),
            "guiado": bool(guiado),
            "modo": modo,
        }
        self.guardar_estado_runtime("orden_enrolamiento", orden)
        return orden
