"""Módulo de persistencia y base de datos relacional SQLite para el LIS.

Implementa la arquitectura de almacenamiento SQLite con modo WAL,
esquema de eventos append-only, soporte de múltiples embeddings por persona,
auditoría de accesos y sincronización retrocompatible con archivos CSV.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sqlite3
import unicodedata
import uuid

import numpy as np


logger = logging.getLogger(__name__)


def normalizar_slug(texto: str) -> str:
    """Genera un slug normalizado en minúsculas sin acentos ni caracteres especiales."""
    texto_norm = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", texto_norm).strip("_").lower()
    return slug or f"usuario_{uuid.uuid4().hex[:8]}"


class RepositorioLIS:
    """Gestiona el acceso a datos en SQLite con sincronización bidireccional a CSV."""

    def __init__(
        self,
        ruta_db: str | Path = "datos/lis_maestra.db",
        ruta_caras_csv: str | Path | None = None,
        ruta_asistencia_csv: str | Path | None = None,
        ruta_ingresos_csv: str | Path | None = None,
    ) -> None:
        self.ruta_db = Path(ruta_db)
        self.ruta_caras_csv = Path(ruta_caras_csv) if ruta_caras_csv else None
        self.ruta_asistencia_csv = Path(ruta_asistencia_csv) if ruta_asistencia_csv else None
        self.ruta_ingresos_csv = Path(ruta_ingresos_csv) if ruta_ingresos_csv else None

        # Asegurar que el directorio de la base de datos exista
        if self.ruta_db.parent and str(self.ruta_db.parent) != ".":
            self.ruta_db.parent.mkdir(parents=True, exist_ok=True)

        self._inicializar_esquema()

        if self.ruta_caras_csv and self.ruta_caras_csv.exists():
            self._migrar_desde_caras_csv_si_vacia(self.ruta_caras_csv)

    def _conectar(self) -> sqlite3.Connection:
        """Crea una conexión a SQLite con modo WAL y claves foráneas activadas."""
        conexion = sqlite3.connect(self.ruta_db)
        conexion.row_factory = sqlite3.Row
        conexion.execute("PRAGMA journal_mode = WAL;")
        conexion.execute("PRAGMA foreign_keys = ON;")
        return conexion

    def _inicializar_esquema(self) -> None:
        """Crea las 4 tablas principales y sus índices si no existen."""
        with self._conectar() as conn:
            cursor = conn.cursor()
            # 1. Tabla de personas (cumple Sección 13)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    nombre TEXT UNIQUE NOT NULL,
                    rol TEXT NOT NULL DEFAULT 'integrante',
                    consentimiento_informado INTEGER NOT NULL DEFAULT 1,
                    fecha_alta TEXT NOT NULL,
                    activo INTEGER NOT NULL DEFAULT 1
                );
                """
            )

            # 2. Tabla de embeddings (soporta múltiples muestras por persona)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    modelo TEXT NOT NULL,
                    fecha_captura TEXT NOT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_persona_modelo
                ON embeddings(persona_id, modelo);
                """
            )

            # 3. Tabla de eventos (esquema exacto Sección 6, append-only)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS eventos (
                    id TEXT PRIMARY KEY,
                    tipo TEXT NOT NULL,
                    persona_id TEXT,
                    objeto_id TEXT,
                    t TEXT NOT NULL,
                    reloj_confiable INTEGER NOT NULL,
                    origen TEXT NOT NULL,
                    metodo TEXT NOT NULL,
                    distancia REAL,
                    meta TEXT
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eventos_persona_t
                ON eventos(persona_id, t);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eventos_tipo
                ON eventos(tipo);
                """
            )

            # 4. Tabla de auditoría de accesos (cumple Sección 13)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auditoria_accesos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT NOT NULL,
                    accion TEXT NOT NULL,
                    detalle TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def registrar_auditoria(self, accion: str, detalle: str) -> None:
        """Registra un evento de auditoría en la base de datos."""
        ahora = datetime.now(timezone.utc).isoformat()
        try:
            with self._conectar() as conn:
                conn.execute(
                    "INSERT INTO auditoria_accesos (fecha, accion, detalle) VALUES (?, ?, ?);",
                    (ahora, accion, detalle),
                )
                conn.commit()
        except Exception:
            logger.exception("Error al registrar auditoría de acceso")

    def _migrar_desde_caras_csv_si_vacia(self, ruta_csv: Path) -> None:
        """Importa los embeddings de caras.csv a SQLite si la tabla embeddings está vacía."""
        with self._conectar() as conn:
            cantidad = conn.execute("SELECT COUNT(*) FROM embeddings;").fetchone()[0]
            if cantidad > 0:
                return

        logger.info("Tabla embeddings vacía. Iniciando migración desde %s", ruta_csv)
        total_migrados = 0
        ahora = datetime.now(timezone.utc).isoformat()

        try:
            with open(ruta_csv, mode="r", encoding="utf-8") as archivo:
                lector = csv.reader(archivo)
                encabezado = next(lector, None)
                for fila in lector:
                    if not fila or len(fila) < 2:
                        continue
                    # caras.csv guarda [embedding_csv_str, nombre]
                    valores_str = fila[0].strip().strip('"')
                    nombre = fila[1].strip().strip('"')
                    if not valores_str or not nombre:
                        continue
                    vector = np.array([float(x) for x in valores_str.split(",")], dtype=np.float32)
                    self.registrar_persona_con_embeddings(
                        nombre=nombre,
                        embeddings=[vector],
                        rol="integrante",
                        consentimiento=True,
                        modelo="vggface2",
                        fecha_registro=ahora,
                        sincronizar_csv=False,
                    )
                    total_migrados += 1

            self.registrar_auditoria(
                accion="migracion_csv_inicial",
                detalle=f"Se migraron {total_migrados} personas desde {ruta_csv}",
            )
            logger.info("Migración completada exitosamente: %d personas importadas", total_migrados)
        except Exception:
            logger.exception("Fallo durante la migración automática de %s", ruta_csv)

    def registrar_persona_con_embeddings(
        self,
        nombre: str,
        embeddings: list[np.ndarray],
        rol: str = "integrante",
        consentimiento: bool = True,
        modelo: str = "vggface2",
        fecha_registro: str | None = None,
        sincronizar_csv: bool = True,
    ) -> str:
        """Registra o actualiza una persona y agrega sus vectores de embedding."""
        if not embeddings:
            raise ValueError("Se requiere al menos un embedding para registrar a la persona.")

        nombre_limpio = nombre.strip()
        persona_id = normalizar_slug(nombre_limpio)
        fecha = fecha_registro or datetime.now(timezone.utc).isoformat()

        with self._conectar() as conn:
            cursor = conn.cursor()
            # Insertar o reactivar persona
            cursor.execute(
                """
                INSERT INTO personas (id, nombre, rol, consentimiento_informado, fecha_alta, activo)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    nombre = excluded.nombre,
                    activo = 1,
                    consentimiento_informado = excluded.consentimiento_informado;
                """,
                (persona_id, nombre_limpio, rol, 1 if consentimiento else 0, fecha),
            )

            # Insertar cada embedding como BLOB float32
            for vector in embeddings:
                vector_f32 = np.asarray(vector, dtype=np.float32)
                blob = vector_f32.tobytes()
                cursor.execute(
                    """
                    INSERT INTO embeddings (persona_id, vector, dimension, modelo, fecha_captura)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (persona_id, blob, int(vector_f32.size), modelo, fecha),
                )
            conn.commit()

        self.registrar_auditoria(
            accion="alta_persona",
            detalle=f"Registrada persona '{nombre_limpio}' (id={persona_id}) con {len(embeddings)} muestras (modelo={modelo})",
        )

        if sincronizar_csv and self.ruta_caras_csv:
            self._agregar_a_caras_csv(nombre_limpio, embeddings)

        return persona_id

    def _agregar_a_caras_csv(self, nombre: str, embeddings: list[np.ndarray]) -> None:
        """Agrega los nuevos embeddings al archivo caras.csv para retrocompatibilidad."""
        try:
            if not self.ruta_caras_csv:
                return
            self.ruta_caras_csv.parent.mkdir(parents=True, exist_ok=True)
            archivo_existe = self.ruta_caras_csv.exists()

            with open(self.ruta_caras_csv, mode="a", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f, quoting=csv.QUOTE_ALL)
                if not archivo_existe:
                    escritor.writerow(["embedding", "nombre"])
                for emb in embeddings:
                    vector_f32 = np.asarray(emb, dtype=np.float32)
                    fila = [",".join(map(str, vector_f32.tolist())), nombre]
                    escritor.writerow(fila)
        except Exception:
            logger.exception("Error al sincronizar con caras.csv")

    def obtener_embeddings_activos(self, modelo: str = "vggface2") -> dict[str, list[np.ndarray]]:
        """Devuelve un diccionario {nombre: [vector1, vector2, ...]} con personas activas."""
        consulta = """
            SELECT p.nombre, e.vector, e.dimension
            FROM personas p
            JOIN embeddings e ON p.id = e.persona_id
            WHERE p.activo = 1 AND e.modelo = ?
            ORDER BY p.nombre, e.id;
        """
        resultado: dict[str, list[np.ndarray]] = {}
        with self._conectar() as conn:
            filas = conn.execute(consulta, (modelo,)).fetchall()
            for fila in filas:
                nombre = fila["nombre"]
                dim = fila["dimension"]
                vector = np.frombuffer(fila["vector"], dtype=np.float32)
                if vector.size != dim:
                    vector = vector[:dim]
                resultado.setdefault(nombre, []).append(vector)
        return resultado

    def registrar_evento(
        self,
        tipo: str,
        nombre: str | None = None,
        persona_id: str | None = None,
        objeto_id: str | None = None,
        metodo: str = "cara+gesto",
        origen: str = "pipeline-facial",
        distancia: float | None = None,
        meta: dict | None = None,
        reloj_confiable: bool = True,
        proyecto: str = "LIS",
    ) -> str:
        """Inserta un evento inmutable en SQLite y sincroniza los archivos CSV en espejo."""
        evento_id = str(uuid.uuid4())
        ahora_dt = datetime.now(timezone.utc)
        t_iso = ahora_dt.isoformat()

        # Resolver persona_id si solo se brindó el nombre
        id_persona = persona_id
        if not id_persona and nombre:
            id_persona = normalizar_slug(nombre)

        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        with self._conectar() as conn:
            if nombre and id_persona:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO personas (id, nombre, rol, consentimiento_informado, fecha_alta, activo)
                    VALUES (?, ?, 'integrante', 1, ?, 1);
                    """,
                    (id_persona, nombre.strip(), t_iso),
                )
            conn.execute(
                """
                INSERT INTO eventos (
                    id, tipo, persona_id, objeto_id, t,
                    reloj_confiable, origen, metodo, distancia, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    evento_id,
                    tipo,
                    id_persona,
                    objeto_id,
                    t_iso,
                    1 if reloj_confiable else 0,
                    origen,
                    metodo,
                    float(distancia) if distancia is not None else None,
                    meta_json,
                ),
            )
            conn.commit()

        # Sincronización en espejo con CSV
        self._sincronizar_espejo_evento(
            tipo=tipo,
            nombre=nombre or id_persona or "N/A",
            proyecto=proyecto,
            ahora_dt=datetime.now(),  # hora local para legibilidad humana en los CSV legacy
        )

        return evento_id

    def _sincronizar_espejo_evento(
        self, tipo: str, nombre: str, proyecto: str, ahora_dt: datetime
    ) -> None:
        """Mantiene actualizados registro_de_ingresos.csv y asistencia.csv."""
        fecha_str = ahora_dt.strftime("%Y-%m-%d")
        hora_str = ahora_dt.strftime("%H:%M")

        # 1. Espejo para registro_de_ingresos.csv
        if self.ruta_ingresos_csv and tipo in ("entrada", "salida"):
            try:
                self.ruta_ingresos_csv.parent.mkdir(parents=True, exist_ok=True)
                filas_existentes: list[list[str]] = []
                if self.ruta_ingresos_csv.exists():
                    with open(self.ruta_ingresos_csv, mode="r", newline="", encoding="utf-8") as f:
                        filas_existentes = list(csv.reader(f))

                if tipo == "entrada":
                    filas_existentes.append([fecha_str, nombre, proyecto, hora_str, "N/A"])
                elif tipo == "salida":
                    # Buscar el último ingreso sin salida de esta persona hoy
                    actualizado = False
                    for i in range(len(filas_existentes) - 1, -1, -1):
                        fila = filas_existentes[i]
                        if len(fila) >= 5 and fila[1] == nombre and fila[4] == "N/A":
                            fila[4] = hora_str
                            actualizado = True
                            break
                    if not actualizado:
                        filas_existentes.append([fecha_str, nombre, proyecto, "N/A", hora_str])

                with open(self.ruta_ingresos_csv, mode="w", newline="", encoding="utf-8") as f:
                    escritor = csv.writer(f)
                    escritor.writerows(filas_existentes)
            except Exception:
                logger.exception("Error al sincronizar registro_de_ingresos.csv")

        # 2. Espejo para asistencia.csv
        if self.ruta_asistencia_csv and tipo == "entrada":
            try:
                self.ruta_asistencia_csv.parent.mkdir(parents=True, exist_ok=True)
                archivo_nuevo = not self.ruta_asistencia_csv.exists()
                with open(self.ruta_asistencia_csv, mode="a", newline="", encoding="utf-8") as f:
                    escritor = csv.writer(f)
                    if archivo_nuevo:
                        escritor.writerow(["fecha", "nombre", "proyecto", "hora_llegada"])
                    escritor.writerow([fecha_str, nombre, proyecto, hora_str])
            except Exception:
                logger.exception("Error al sincronizar asistencia.csv")

    def registrar_asistencia_permanencia(
        self, nombre: str, proyecto: str = "LIS", distancia: float | None = None
    ) -> str:
        """Registra un evento de presencia periódica en asistencia.csv y en la tabla eventos."""
        return self.registrar_evento(
            tipo="asistencia_permanencia",
            nombre=nombre,
            metodo="cara_pasiva",
            origen="pipeline-facial",
            distancia=distancia,
            proyecto=proyecto,
        )

    def obtener_estado_usuario(self, nombre_o_id: str) -> str:
        """Deriva si el usuario está 'DENTRO' o 'FUERA' según su último evento de entrada/salida."""
        slug = normalizar_slug(nombre_o_id)
        consulta = """
            SELECT tipo FROM eventos
            WHERE (persona_id = ? OR persona_id = ?) AND tipo IN ('entrada', 'salida')
            ORDER BY t DESC
            LIMIT 1;
        """
        with self._conectar() as conn:
            fila = conn.execute(consulta, (slug, nombre_o_id)).fetchone()
            if fila and fila["tipo"] == "entrada":
                return "DENTRO"
        return "FUERA"

    def esta_adentro(self, nombre_o_id: str) -> bool:
        """Devuelve True si el usuario tiene como último estado 'entrada'."""
        return self.obtener_estado_usuario(nombre_o_id) == "DENTRO"

    def obtener_presentes(self) -> set[str]:
        """Obtiene el conjunto de nombres o IDs de personas que están actualmente adentro."""
        consulta = """
            WITH ultimos_eventos AS (
                SELECT persona_id, tipo, ROW_NUMBER() OVER (
                    PARTITION BY persona_id ORDER BY t DESC
                ) as rn
                FROM eventos
                WHERE tipo IN ('entrada', 'salida') AND persona_id IS NOT NULL
            )
            SELECT p.nombre, ue.persona_id
            FROM ultimos_eventos ue
            LEFT JOIN personas p ON ue.persona_id = p.id
            WHERE ue.rn = 1 AND ue.tipo = 'entrada';
        """
        presentes: set[str] = set()
        with self._conectar() as conn:
            filas = conn.execute(consulta).fetchall()
            for fila in filas:
                nombre = fila["nombre"] or fila["persona_id"]
                presentes.add(nombre)
        return presentes

    def obtener_conteo_presentes(self) -> int:
        """Devuelve la cantidad de personas actualmente adentro del laboratorio."""
        return len(self.obtener_presentes())
