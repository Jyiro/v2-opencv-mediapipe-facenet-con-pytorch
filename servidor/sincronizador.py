"""Sincronizador por Pull del Servidor Maestro LIS (Fase 3 y Sección 5).

Consulta periódicamente 'GET /events?since=<cursor>' al Portero,
guarda los eventos en la base de datos maestra SQLite (datos/lis_maestra.db)
y en los archivos CSV en espejo, y persiste el cursor confirmado.

Incluye control de pausa/reanudación para simular el servidor apagado/encendido.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import urllib.request

logger = logging.getLogger(__name__)


class SincronizadorServidor:
    """Hilo de sincronización por pull desde el Portero hacia la base SQLite maestra."""

    def __init__(
        self,
        repositorio,
        url_portero: str = "http://127.0.0.1:8080/events",
        ruta_cursor: str | Path = "datos/servidor_cursor.txt",
        intervalo_segundos: float = 2.0,
    ) -> None:
        self.repositorio = repositorio
        self.url_portero = url_portero
        self.ruta_cursor = Path(ruta_cursor)
        self.intervalo = intervalo_segundos

        self.ruta_cursor.parent.mkdir(parents=True, exist_ok=True)
        self.cursor = self._cargar_cursor()

        self.activo = True
        self.pausado = False
        self.hilo: threading.Thread | None = None
        self.lock = threading.Lock()
        self.eventos_sincronizados_total = 0

    def _cargar_cursor(self) -> int:
        if self.ruta_cursor.exists():
            try:
                texto = self.ruta_cursor.read_text(encoding="utf-8").strip()
                return int(texto)
            except Exception:
                pass
        return 0

    def _guardar_cursor(self, cursor: int) -> None:
        try:
            self.ruta_cursor.write_text(str(cursor), encoding="utf-8")
        except Exception:
            logger.exception("Error al guardar cursor del servidor")

    def iniciar(self) -> None:
        """Inicia el hilo de sincronización periódica en segundo plano."""
        self.activo = True
        self.hilo = threading.Thread(target=self._bucle_sincronizacion, daemon=True, name="SincronizadorPull")
        self.hilo.start()
        logger.info("Sincronizador del Servidor iniciado (cursor inicial=%d).", self.cursor)

    def pausar(self) -> None:
        """Simula el apagado del servidor."""
        with self.lock:
            self.pausado = True
        logger.warning("[SIMULACIÓN] Servidor Maestro APAGADO (sincronización pausada).")

    def reanudar(self) -> None:
        """Simula el encendido del servidor."""
        with self.lock:
            self.pausado = False
        logger.info("[SIMULACIÓN] Servidor Maestro ENCENDIDO (reanudando sincronización pull).")

    def esta_pausado(self) -> bool:
        with self.lock:
            return self.pausado

    def sincronizar_ahora(self) -> int:
        """Ejecuta una iteración de pull manual y retorna cantidad de eventos importados."""
        with self.lock:
            if self.pausado:
                return 0

        url = f"{self.url_portero}?since={self.cursor}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    cuerpo = resp.read().decode("utf-8")
                    data = json.loads(cuerpo)
                    nuevo_cursor = data.get("cursor", self.cursor)
                    eventos = data.get("events", [])

                    if eventos:
                        for ev in eventos:
                            self.repositorio.registrar_evento(
                                tipo=ev["tipo"],
                                persona_id=ev.get("persona_id"),
                                objeto_id=ev.get("objeto_id"),
                                metodo=ev.get("metodo", "manual"),
                                origen=ev.get("origen", "portero"),
                                distancia=ev.get("distancia"),
                                meta=ev.get("meta"),
                                reloj_confiable=ev.get("reloj_confiable", True),
                            )
                        self.cursor = nuevo_cursor
                        self._guardar_cursor(self.cursor)
                        self.eventos_sincronizados_total += len(eventos)
                        logger.info("Servidor sincronizó %d nuevos eventos (cursor=%d).", len(eventos), self.cursor)
                        return len(eventos)
        except Exception:
            # Fallo normal si el Portero no está corriendo en ese momento
            pass

        return 0

    def _bucle_sincronizacion(self) -> None:
        while self.activo:
            self.sincronizar_ahora()
            time.sleep(self.intervalo)

    def detener(self) -> None:
        self.activo = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from nucleo.base_datos import RepositorioLIS

    repo = RepositorioLIS()
    sinc = SincronizadorServidor(repositorio=repo)
    sinc.iniciar()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sinc.detener()
