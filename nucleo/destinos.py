"""Módulo de destinos de eventos (patrón Observer / Adapter para persistencia distribuida).

Permite enviar eventos a múltiples sinks desacoplados:
- Base de datos maestra SQLite.
- Archivos CSV compatibles.
- Portero HTTP (Galileo o servidor local) con cola de reintentos JSONL offline-first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
import urllib.request

logger = logging.getLogger(__name__)


class DestinoDeEventos(ABC):
    """Interfaz abstracta para el envío y persistencia de eventos."""

    @abstractmethod
    def enviar_evento(self, evento: dict[str, Any]) -> bool:
        """Procesa y almacena un evento inmutable."""
        pass

    def cerrar(self) -> None:
        """Libera recursos si corresponde."""
        pass


class DestinoSQLite(DestinoDeEventos):
    """Envía eventos a la base de datos relacional SQLite."""

    def __init__(self, repositorio) -> None:
        self.repositorio = repositorio

    def enviar_evento(self, evento: dict[str, Any]) -> bool:
        try:
            self.repositorio.registrar_evento(
                tipo=evento["tipo"],
                persona_id=evento.get("persona_id"),
                objeto_id=evento.get("objeto_id"),
                metodo=evento.get("metodo", "manual"),
                origen=evento.get("origen", "pipeline"),
                distancia=evento.get("distancia"),
                meta=evento.get("meta"),
                reloj_confiable=evento.get("reloj_confiable", True),
            )
            return True
        except Exception:
            logger.exception("Error al persistir evento en SQLite")
            return False


class DestinoCSV(DestinoDeEventos):
    """Mantiene sincronizados en espejo los archivos CSV sin sobrescribir reingresos."""

    def __init__(
        self,
        ruta_ingresos_csv: str | Path,
        ruta_asistencia_csv: str | Path,
        proyecto: str = "LIS",
    ) -> None:
        self.ruta_ingresos = Path(ruta_ingresos_csv)
        self.ruta_asistencia = Path(ruta_asistencia_csv)
        self.proyecto = proyecto

    def enviar_evento(self, evento: dict[str, Any]) -> bool:
        tipo = evento.get("tipo")
        if tipo not in ("entrada", "salida"):
            return True

        pid = evento.get("persona_id") or "N/A"
        t_str = evento.get("t", "")
        try:
            dt = datetime.fromisoformat(t_str)
        except Exception:
            dt = datetime.now()

        fecha_str = dt.strftime("%Y-%m-%d")
        hora_str = dt.strftime("%H:%M")

        # 1. Registro de ingresos
        try:
            self.ruta_ingresos.parent.mkdir(parents=True, exist_ok=True)
            filas: list[list[str]] = []
            if self.ruta_ingresos.exists():
                with open(self.ruta_ingresos, "r", encoding="utf-8") as f:
                    filas = list(csv.reader(f))

            if tipo == "entrada":
                filas.append([fecha_str, pid, self.proyecto, hora_str, "N/A"])
            elif tipo == "salida":
                actualizado = False
                for i in range(len(filas) - 1, -1, -1):
                    if len(filas[i]) >= 5 and filas[i][1] == pid and filas[i][4] == "N/A":
                        filas[i][4] = hora_str
                        actualizado = True
                        break
                if not actualizado:
                    filas.append([fecha_str, pid, self.proyecto, "N/A", hora_str])

            with open(self.ruta_ingresos, "w", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                escritor.writerows(filas)
        except Exception:
            logger.exception("Error al actualizar CSV de ingresos")
            return False

        # 2. Asistencia
        if tipo == "entrada":
            try:
                self.ruta_asistencia.parent.mkdir(parents=True, exist_ok=True)
                es_nuevo = not self.ruta_asistencia.exists()
                with open(self.ruta_asistencia, "a", newline="", encoding="utf-8") as f:
                    escritor = csv.writer(f)
                    if es_nuevo:
                        escritor.writerow(["fecha", "nombre", "proyecto", "hora_llegada"])
                    escritor.writerow([fecha_str, pid, self.proyecto, hora_str])
            except Exception:
                logger.exception("Error al actualizar CSV de asistencia")
                return False

        return True


class DestinoPorteroHTTP(DestinoDeEventos):
    """Envía eventos al Portero HTTP (POST /events) con cola local JSONL offline-first."""

    def __init__(
        self,
        url_portero: str = "http://127.0.0.1:8080/events",
        ruta_cola_pendientes: str | Path = "datos/cola_pendientes.jsonl",
        timeout_segundos: float = 2.0,
    ) -> None:
        self.url_portero = url_portero
        self.ruta_cola = Path(ruta_cola_pendientes)
        self.timeout = timeout_segundos
        self.ruta_cola.parent.mkdir(parents=True, exist_ok=True)

    def enviar_evento(self, evento: dict[str, Any]) -> bool:
        # Intentar reintentar cola pendiente primero
        self.reintentar_pendientes()

        enviado = self._enviar_directo(evento)
        if not enviado:
            self._guardar_en_cola(evento)
        return enviado

    def _enviar_directo(self, evento: dict[str, Any]) -> bool:
        try:
            datos_json = json.dumps([evento]).encode("utf-8")
            req = urllib.request.Request(
                self.url_portero,
                data=datos_json,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status in (200, 201):
                    return True
        except Exception:
            # Fallo silencioso con registro debug para no saturar consola cuando el Portero está apagado
            logger.debug("Portero HTTP no disponible. Encolando evento localmente.")
        return False

    def _guardar_en_cola(self, evento: dict[str, Any]) -> None:
        try:
            with open(self.ruta_cola, "a", encoding="utf-8") as f:
                f.write(json.dumps(evento, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Error al guardar evento en cola de pendientes")

    def reintentar_pendientes(self) -> int:
        """Intenta enviar los eventos acumulados cuando se restablece la conexión."""
        if not self.ruta_cola.exists() or self.ruta_cola.stat().st_size == 0:
            return 0

        eventos_pendientes: list[dict[str, Any]] = []
        try:
            with open(self.ruta_cola, "r", encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if linea:
                        eventos_pendientes.append(json.loads(linea))
        except Exception:
            logger.exception("Error al leer cola de pendientes")
            return 0

        if not eventos_pendientes:
            return 0

        try:
            datos_json = json.dumps(eventos_pendientes).encode("utf-8")
            req = urllib.request.Request(
                self.url_portero,
                data=datos_json,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status in (200, 201):
                    # Limpiar cola de pendientes
                    with open(self.ruta_cola, "w", encoding="utf-8") as f:
                        f.truncate(0)
                    logger.info("Se vació la cola pendiente: %d eventos entregados al Portero.", len(eventos_pendientes))
                    return len(eventos_pendientes)
        except Exception:
            pass

        return 0


class DestinoCompuesto(DestinoDeEventos):
    """Distribuye eventos a una colección de destinos configurados."""

    def __init__(self, destinos: list[DestinoDeEventos] | None = None) -> None:
        self.destinos = destinos or []

    def agregar_destino(self, destino: DestinoDeEventos) -> None:
        self.destinos.append(destino)

    def enviar_evento(self, evento: dict[str, Any]) -> bool:
        exitos = True
        for dest in self.destinos:
            try:
                res = dest.enviar_evento(evento)
                if not res:
                    exitos = False
            except Exception:
                logger.exception("Error al despachar evento en destino compuesto")
                exitos = False
        return exitos

    def cerrar(self) -> None:
        for dest in self.destinos:
            try:
                dest.cerrar()
            except Exception:
                pass
