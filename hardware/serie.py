from __future__ import annotations
import logging
from datetime import datetime
from typing import Any

from nucleo.base_datos import RepositorioLIS


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
        simular: bool = True,
        simulado: bool | None = None,
    ) -> None:
        self.repositorio = repositorio
        self.puerto = puerto
        self.baudios = baudios
        if simulado is not None:
            simular = simulado
        self.modo_simulado = simular
        self._serial = None
        self.estado: dict[str, Any] = {
            "conectado_fisico": False,
            "modo_simulado": True,
            "linea1": "UTN FRC - LIS   ",
            "linea2": "EN ESPERA       ",
            "rgb": [0, 180, 120],
            "leds": {"azul": 0, "verde": 1, "rojo": 0},
            "ultimo_beep": {"tipo": "none", "t": ""},
            "ultimo_comando": "INIT",
        }

        if not simular and puerto:
            try:
                import serial  # type: ignore

                self._serial = serial.Serial(puerto, baudios, timeout=0.2)
                self.enviar_comando("PING")
                resp = self._serial.readline().decode("ascii", errors="ignore").strip()
                if resp == "PONG":
                    self.modo_simulado = False
                    self.estado["conectado_fisico"] = True
                    self.estado["modo_simulado"] = False
                    logging.info("Fichador Arduino conectado en %s", puerto)
                else:
                    logging.warning("Arduino no respondió PONG en %s; usando modo simulado.", puerto)
            except Exception as exc:
                logging.info("Puerto serie no disponible (%s); degradando a modo simulado.", exc)
                self.modo_simulado = True

        self._persistir()

    @staticmethod
    def truncar_16(texto: str) -> str:
        limpio = texto.replace("|", " ").strip()
        return limpio[:16].ljust(16)

    def enviar_comando(self, comando: str) -> None:
        linea = comando.strip() + "\n"
        self.estado["ultimo_comando"] = comando.strip()
        if self._serial and not self.modo_simulado:
            try:
                self._serial.write(linea.encode("ascii", errors="replace"))
            except Exception:
                logging.warning("Fallo escritura serie; pasando a modo simulado.")
                self.modo_simulado = True
                self.estado["modo_simulado"] = True

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
        patron = patron if patron in ("ok", "err", "warn") else "ok"
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
            self.enviar_comando(f"LED|{color}|{val}")
            self._persistir()

    def notificar_evento_acceso(self, nombre: str, tipo: str, distancia: float | None = None) -> None:
        hora = datetime.now().strftime("%H:%M")
        nombre_corto = nombre.upper()[:10]
        if tipo == "entrada":
            self.cambiar_rgb(16, 185, 129)
            self.cambiar_led("verde", True)
            self.cambiar_led("azul", False)
            self.cambiar_led("rojo", False)
            dist_str = f"d:{distancia:.2f}" if distancia is not None else "PIN/QR"
            self.mostrar_mensaje(f"IN: {nombre_corto}", f"{hora} {dist_str}")
            self.sonar_beep("ok")
        elif tipo == "salida":
            self.cambiar_rgb(59, 130, 246)
            self.cambiar_led("azul", True)
            self.cambiar_led("verde", False)
            self.cambiar_led("rojo", False)
            self.mostrar_mensaje(f"OUT: {nombre_corto}", f"HASTA LUEGO {hora}")
            self.sonar_beep("ok")
        else:
            self.cambiar_rgb(239, 68, 68)
            self.cambiar_led("rojo", True)
            self.cambiar_led("verde", False)
            self.cambiar_led("azul", False)
            self.mostrar_mensaje("NO RECONOCIDO", "USE PIN O QR")
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
