"""Cliente de comunicación serie para el Fichador Arduino UNO y Base Shield Grove.

Implementa el protocolo oficial de la Sección 8 (115200 baudios, delimitado por '|'):
- Comandos a Arduino: MSG, RGB, BEEP, LED, PING.
- Eventos desde Arduino: TOUCH, BTN, PONG, SENS.
- Modo Simulado interactivo cuando no hay placa física conectada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def truncar_16(texto: str) -> str:
    """Trunca el texto a 16 caracteres para compatibilidad con display LCD 16x2."""
    return str(texto)[:16]


@dataclass
class EstadoFichadorVirtual:
    """Estado en memoria del Fichador simulado para la interfaz web."""

    linea1: str = "LIS Control"
    linea2: str = "Esperando..."
    rgb: tuple[int, int, int] = (0, 100, 255)
    leds: dict[str, int] = field(default_factory=lambda: {"azul": 0, "verde": 0, "rojo": 0})
    ultimo_beep: str | None = None
    ultimo_beep_tiempo: float = 0.0


class ClienteFichadorSerie:
    """Gestiona la comunicación con el Arduino del Fichador o emula su comportamiento."""

    def __init__(
        self,
        puerto: str | None = None,
        baudios: int = 115200,
        simulado: bool = False,
        callback_touch: Callable[[], None] | None = None,
        callback_boton: Callable[[], None] | None = None,
    ) -> None:
        self.puerto = puerto
        self.baudios = baudios
        self.simulado = simulado
        self.callback_touch = callback_touch
        self.callback_boton = callback_boton

        self.ser = None
        self.hilo_lector: threading.Thread | None = None
        self.activo = True
        self.lock = threading.Lock()

        self.estado_virtual = EstadoFichadorVirtual()

        if not self.simulado and self.puerto:
            self._conectar_serie()
        else:
            self.simulado = True
            logger.info("Fichador iniciado en MODO SIMULADO (virtual).")

    def _conectar_serie(self) -> None:
        try:
            import serial  # type: ignore

            logger.info("Intentando abrir puerto serie %s a %d baudios...", self.puerto, self.baudios)
            self.ser = serial.Serial(self.puerto, self.baudios, timeout=1.0)
            time.sleep(2.0)  # Esperar reinicio de Arduino al abrir DTR
            self.simulado = False

            # Iniciar hilo lector de respuestas y eventos físicos
            self.hilo_lector = threading.Thread(target=self._bucle_lectura_serie, daemon=True)
            self.hilo_lector.start()
            logger.info("Fichador físico conectado exitosamente en %s.", self.puerto)
        except Exception as e:
            logger.warning("No se pudo conectar al Fichador físico (%s). Fallback a MODO SIMULADO.", e)
            self.simulado = True
            self.ser = None

    def _bucle_lectura_serie(self) -> None:
        while self.activo and self.ser and self.ser.is_open:
            try:
                linea = self.ser.readline().decode("ascii", errors="ignore").strip()
                if not linea:
                    continue

                partes = linea.split("|")
                comando = partes[0].upper()

                if comando == "TOUCH":
                    logger.info("Evento físico recibido: TOUCH")
                    if self.callback_touch:
                        self.callback_touch()
                elif comando == "BTN":
                    logger.info("Evento físico recibido: BTN")
                    if self.callback_boton:
                        self.callback_boton()
                elif comando == "PONG":
                    logger.debug("Respuesta recibida: PONG")
            except Exception:
                time.sleep(0.1)

    def enviar_mensaje_lcd(self, linea1: str, linea2: str = "") -> None:
        """Envía texto al LCD 16x2 truncando a 16 caracteres por línea."""
        l1 = truncar_16(linea1)
        l2 = truncar_16(linea2)
        with self.lock:
            self.estado_virtual.linea1 = l1
            self.estado_virtual.linea2 = l2

        self._transmitir(f"MSG|{l1}|{l2}\n")

    def set_color_lcd(self, r: int, g: int, b: int) -> None:
        """Configura el color de fondo RGB del display Grove (0-255)."""
        r_c = max(0, min(255, int(r)))
        g_c = max(0, min(255, int(g)))
        b_c = max(0, min(255, int(b)))
        with self.lock:
            self.estado_virtual.rgb = (r_c, g_c, b_c)

        self._transmitir(f"RGB|{r_c}|{g_c}|{b_c}\n")

    def beep(self, patron: str = "ok") -> None:
        """Emite un sonido de buzzer ('ok', 'err', 'warn')."""
        p = patron.lower().strip()
        if p not in ("ok", "err", "warn"):
            p = "ok"
        with self.lock:
            self.estado_virtual.ultimo_beep = p
            self.estado_virtual.ultimo_beep_tiempo = time.time()

        self._transmitir(f"BEEP|{p}\n")

    def set_led(self, color: str, estado: int | bool) -> None:
        """Enciende o apaga uno de los LEDs DIP ('azul', 'verde', 'rojo')."""
        c = color.lower().strip()
        e = 1 if estado else 0
        with self.lock:
            if c in self.estado_virtual.leds:
                self.estado_virtual.leds[c] = e

        self._transmitir(f"LED|{c}|{e}\n")

    def ping(self) -> bool:
        """Envía PING al Arduino para verificar presencia física."""
        if self.simulado or not self.ser:
            return True
        try:
            self._transmitir("PING\n")
            return True
        except Exception:
            return False

    def _transmitir(self, comando: str) -> None:
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(comando.encode("ascii"))
                self.ser.flush()
            except Exception:
                logger.exception("Error al transmitir comando serie: %s", comando)

    def simular_touch(self) -> None:
        """Dispara el evento de sensor táctil desde el simulador o tablero web."""
        logger.info("Simulación: Sensor TOUCH accionado.")
        if self.callback_touch:
            self.callback_touch()

    def simular_boton(self) -> None:
        """Dispara el evento de botón desde el simulador o tablero web."""
        logger.info("Simulación: Botón Fichador presionado.")
        if self.callback_boton:
            self.callback_boton()

    def obtener_estado_virtual(self) -> dict[str, Any]:
        """Devuelve el estado serializado para alimentar la interfaz web interactiva."""
        with self.lock:
            return {
                "linea1": self.estado_virtual.linea1,
                "linea2": self.estado_virtual.linea2,
                "rgb": list(self.estado_virtual.rgb),
                "leds": dict(self.estado_virtual.leds),
                "ultimo_beep": self.estado_virtual.ultimo_beep,
                "ultimo_beep_tiempo": self.estado_virtual.ultimo_beep_tiempo,
                "simulado": self.simulado,
            }

    def cerrar(self) -> None:
        self.activo = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
