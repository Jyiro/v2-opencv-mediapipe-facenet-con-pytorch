from __future__ import annotations
import glob
import logging
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from nucleo.base_datos import RepositorioLIS

PROYECTO_DIR = Path(__file__).resolve().parent.parent

_LOCK_SERIE = threading.Lock()
_SERIAL_COMPARTIDO: Any = None
_PUERTO_ACTIVO: str | None = None
_HILO_LECTOR: threading.Thread | None = None
_CALLBACKS_BOTONES: list[Callable[[str], None]] = []


def _cargar_cfg_fichador() -> dict[str, Any]:
    ruta = PROYECTO_DIR / "config.toml"
    if ruta.exists():
        try:
            with open(ruta, "rb") as f:
                return tomllib.load(f).get("fichador", {})
        except Exception:
            pass
    return {}


def detectar_puerto_arduino() -> str | None:
    candidatos = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if candidatos:
        return candidatos[0]
    try:
        from serial.tools import list_ports  # type: ignore

        for p in list_ports.comports():
            if p.device and ("USB" in p.device or "ACM" in p.device or "COM" in p.device):
                return p.device
    except Exception:
        pass
    return None


def truncar_16(texto: str) -> str:
    limpio = str(texto).replace("|", " ").strip()
    return limpio[:16].ljust(16)


class ClienteFichadorSerie:
    """
    Cliente del protocolo serie PC <-> Arduino UNO (Sección 8).
    Trunca cada línea del LCD a 16 caracteres exactos y degrada limpiamente a modo simulado.
    """

    truncar_16 = staticmethod(truncar_16)

    def __init__(
        self,
        repositorio: RepositorioLIS | None = None,
        puerto: str | None = None,
        baudios: int = 115200,
        simular: bool | None = None,
        simulado: bool | None = None,
    ) -> None:
        global _SERIAL_COMPARTIDO, _PUERTO_ACTIVO, _HILO_LECTOR

        self.repositorio = repositorio
        cfg = _cargar_cfg_fichador()

        if simulado is not None:
            usar_simulado = bool(simulado)
        elif simular is not None:
            usar_simulado = bool(simular)
        else:
            usar_simulado = bool(cfg.get("simular", False))

        puerto_cfg = puerto or str(cfg.get("puerto", "auto")).strip()
        if puerto_cfg.lower() in ("auto", ""):
            puerto_resuelto = None if usar_simulado else detectar_puerto_arduino()
        else:
            puerto_resuelto = puerto_cfg

        self._forzar_simulado = usar_simulado
        self._puerto_cfg = puerto_cfg
        self.puerto = puerto_resuelto
        self.baudios = int(cfg.get("baudios", baudios))
        self.modo_simulado = True
        self._serial = None
        self.estado: dict[str, Any] = {
            "conectado_fisico": False,
            "modo_simulado": True,
            "puerto": self.puerto or "simulado",
            "linea1": "UTN FRC - LIS   ",
            "linea2": "TOQUE EL SENSOR ",
            "rgb": [10, 60, 150],
            "leds": {"azul": 0, "verde": 0, "rojo": 0},
            "ultimo_beep": {"tipo": "none", "t": ""},
            "ultimo_comando": "INIT",
        }

        if not self._forzar_simulado:
            if self._intentar_conectar_serie():
                self.resetear()
            if _HILO_LECTOR is None or not _HILO_LECTOR.is_alive():
                _HILO_LECTOR = threading.Thread(target=self._bucle_lectura_serie, daemon=True)
                _HILO_LECTOR.start()

        self._persistir()

    def _intentar_conectar_serie(self) -> bool:
        """Intenta abrir el puerto serie (o reutilizar el compartido) y validar PING/PONG."""
        global _SERIAL_COMPARTIDO, _PUERTO_ACTIVO

        if self._forzar_simulado:
            return False

        puerto_obj = self._puerto_cfg
        if puerto_obj.lower() in ("auto", ""):
            puerto_obj = detectar_puerto_arduino() or ""
        if not puerto_obj:
            return False

        with _LOCK_SERIE:
            if _SERIAL_COMPARTIDO is not None and getattr(_SERIAL_COMPARTIDO, "is_open", False):
                self._serial = _SERIAL_COMPARTIDO
                self.puerto = _PUERTO_ACTIVO
                self.modo_simulado = False
                self.estado["conectado_fisico"] = True
                self.estado["modo_simulado"] = False
                self.estado["puerto"] = self.puerto
                return True

            try:
                import serial  # type: ignore

                ser = serial.Serial(puerto_obj, self.baudios, timeout=0.25)
                # Al abrir el puerto serie en Linux, DTR reinicia el bootloader del Arduino UNO (~1.6s)
                conectado = False
                fin_espera = time.time() + 3.0
                time.sleep(1.7)
                ser.reset_input_buffer()
                while time.time() < fin_espera:
                    ser.write(b"PING\n")
                    ser.flush()
                    resp = ser.readline().decode("ascii", errors="ignore").strip()
                    if resp == "PONG":
                        conectado = True
                        break
                    time.sleep(0.2)

                if conectado:
                    _SERIAL_COMPARTIDO = ser
                    _PUERTO_ACTIVO = puerto_obj
                    self._serial = ser
                    self.puerto = puerto_obj
                    self.modo_simulado = False
                    self.estado["conectado_fisico"] = True
                    self.estado["modo_simulado"] = False
                    self.estado["puerto"] = self.puerto
                    logging.info("Fichador Arduino físico conectado en %s (%d baudios)", self.puerto, self.baudios)
                    return True
                logging.warning("Arduino no respondió PONG en %s; usando modo simulado.", puerto_obj)
                ser.close()
            except Exception as exc:
                if self.estado.get("ultimo_comando") == "INIT":
                    logging.info(
                        "Puerto serie %s no disponible (%s); degradando a modo simulado. "
                        "(Recordá cerrar el Serial Monitor de Arduino IDE si estaba abierto).",
                        puerto_obj,
                        exc,
                    )
                self.modo_simulado = True
                self.estado["conectado_fisico"] = False
                self.estado["modo_simulado"] = True
        return False

    def _marcar_desconexion_serie(self) -> None:
        global _SERIAL_COMPARTIDO, _PUERTO_ACTIVO
        with _LOCK_SERIE:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
            _SERIAL_COMPARTIDO = None
            _PUERTO_ACTIVO = None
            self._serial = None
            self.modo_simulado = True
            self.estado["conectado_fisico"] = False
            self.estado["modo_simulado"] = True
        self._persistir()

    def registrar_callback_boton(self, fn: Callable[[str], None]) -> None:
        if fn not in _CALLBACKS_BOTONES:
            _CALLBACKS_BOTONES.append(fn)

    def _bucle_lectura_serie(self) -> None:
        while not self._forzar_simulado:
            if self._serial is None or not getattr(self._serial, "is_open", False) or self.modo_simulado:
                time.sleep(4.0)
                if self._intentar_conectar_serie():
                    rgb = self.estado.get("rgb", [10, 60, 150])
                    self.cambiar_rgb(rgb[0], rgb[1], rgb[2])
                    self.mostrar_mensaje(self.estado.get("linea1", "UTN FRC - LIS"), self.estado.get("linea2", "TOQUE EL SENSOR"))
                continue
            try:
                if self._serial.in_waiting > 0:
                    with _LOCK_SERIE:
                        linea = self._serial.readline().decode("ascii", errors="ignore").strip().upper()
                    if linea in ("TOUCH", "BTN") or linea.startswith("SENS|"):
                        logging.info("Evento físico recibido desde Arduino: %s", linea)
                        if self.repositorio is not None:
                            self.repositorio.guardar_estado_runtime(
                                "evento_hardware_arduino",
                                {"evento": linea, "t": time.time()},
                            )
                        for cb in list(_CALLBACKS_BOTONES):
                            try:
                                cb(linea)
                            except Exception:
                                pass
                else:
                    time.sleep(0.03)
            except Exception as exc:
                logging.warning("Desconexión serie detectada (%s); reintentando en segundo plano.", exc)
                self._marcar_desconexion_serie()
                time.sleep(1.0)

    @staticmethod
    def truncar_16(texto: str) -> str:
        limpio = texto.replace("|", " ").strip()
        return limpio[:16].ljust(16)

    def enviar_comando(self, comando: str) -> None:
        linea = comando.strip() + "\n"
        self.estado["ultimo_comando"] = comando.strip()
        if self._serial and not self.modo_simulado:
            try:
                with _LOCK_SERIE:
                    self._serial.write(linea.encode("ascii", errors="replace"))
                    self._serial.flush()
            except Exception:
                logging.warning("Fallo escritura serie; pasando a modo simulado (reintento automático activo).")
                self._marcar_desconexion_serie()

    def resetear(self) -> None:
        """Restaura el fichador a su estado original de reposo (Standby)."""
        self.cambiar_rgb(10, 60, 150)
        self.mostrar_mensaje("UTN FRC - LIS", "TOQUE EL SENSOR")

    def mostrar_mensaje(self, linea1: str, linea2: str = "") -> None:
        l1 = self.truncar_16(linea1)
        l2 = self.truncar_16(linea2)
        self.estado["linea1"] = l1
        self.estado["linea2"] = l2
        self.enviar_comando(f"MSG|{l1.strip()}|{l2.strip()}")
        self._persistir()

    def cambiar_rgb(self, r: int, g: int, b: int) -> None:
        r, g, b = max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))
        self.estado["rgb"] = [r, g, b]
        self.enviar_comando(f"RGB|{r}|{g}|{b}")
        self._persistir()

    def sonar_beep(self, patron: str = "ok") -> None:
        patron = patron if patron in ("ok", "in", "out", "wake", "err", "warn") else "ok"
        self.estado["ultimo_beep"] = {
            "tipo": patron,
            "t": datetime.now().strftime("%H:%M:%S.%f"),
        }
        self.enviar_comando(f"BEEP|{patron}")
        self._persistir()

    def cambiar_led(self, color: str, encendido: bool) -> None:
        if color in ("azul", "verde", "rojo"):
            val = 1 if encendido else 0
            self.estado["leds"][color] = val
            self._persistir()

    def notificar_evento_acceso(self, nombre: str, tipo: str, distancia: float | None = None) -> None:
        hora = datetime.now().strftime("%H:%M")
        nombre_corto = nombre.upper()[:12]
        if tipo == "entrada":
            self.cambiar_rgb(16, 230, 80)
            dist_str = f"d:{distancia:.2f}" if distancia is not None else "PIN"
            self.mostrar_mensaje(f"IN: {nombre_corto}", f"{hora} {dist_str}")
            self.sonar_beep("in")
        elif tipo == "salida":
            self.cambiar_rgb(30, 100, 255)
            self.mostrar_mensaje(f"OUT: {nombre_corto}", f"ADIOS! {hora}")
            self.sonar_beep("out")
        else:
            self.cambiar_rgb(239, 40, 40)
            self.mostrar_mensaje("NO RECONOCIDO", "USE PIN EN TABLET")
            self.sonar_beep("err")

    def enviar_mensaje_lcd(self, linea1: str, linea2: str = "") -> None:
        self.mostrar_mensaje(linea1, linea2)

    def set_color_lcd(self, r: int, g: int, b: int) -> None:
        self.cambiar_rgb(r, g, b)

    def set_led(self, color: str, encendido: bool | int) -> None:
        self.cambiar_led(color, bool(encendido))

    def beep(self, patron: str = "ok") -> None:
        self.sonar_beep(patron)

    def obtener_estado_virtual(self) -> dict[str, Any]:
        copia = dict(self.estado)
        if isinstance(self.estado.get("ultimo_beep"), dict):
            copia["ultimo_beep"] = self.estado["ultimo_beep"].get("tipo", "none")
        return copia

    def _persistir(self) -> None:
        if self.repositorio is not None:
            try:
                self.repositorio.guardar_estado_runtime("fichador_lcd", self.estado)
            except Exception:
                pass
