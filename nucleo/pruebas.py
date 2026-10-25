"""Estadísticas en vivo del modo de pruebas: embedding crudo vs. con filtros de iluminación.

Todo es lógica pura (sin cámara ni modelos) para poder testearla.

Si se indica el sujeto esperado (quién está frente a la cámara), se estiman en vivo:
  - aciertos (lo reconoce como él),
  - falsas aceptaciones (lo reconoce como otra persona),
  - falsos rechazos (no lo reconoce).
Sin sujeto, solo se cuenta cuántas veces cada variante logra reconocer a alguien.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BRAZOS = ("crudo", "filtrado")


@dataclass
class BrazoStats:
    inferencias: int = 0
    reconocidas: int = 0
    aciertos: int = 0
    falsas_aceptaciones: int = 0
    falsos_rechazos: int = 0
    suma_distancia_sujeto: float = 0.0
    n_distancia_sujeto: int = 0
    mejor_distancia: float | None = None

    def registrar(self, *, nombre: str | None, distancia: float, reconocido: bool, sujeto: str | None) -> None:
        self.inferencias += 1
        if reconocido:
            self.reconocidas += 1
        if sujeto:
            if reconocido and nombre and nombre.lower() == sujeto.lower():
                self.aciertos += 1
            elif reconocido:
                self.falsas_aceptaciones += 1
            else:
                self.falsos_rechazos += 1
        if (not sujeto) or (nombre and nombre.lower() == sujeto.lower()):
            if distancia < 9.0:
                self.suma_distancia_sujeto += distancia
                self.n_distancia_sujeto += 1
                if self.mejor_distancia is None or distancia < self.mejor_distancia:
                    self.mejor_distancia = distancia

    def a_dict(self) -> dict[str, Any]:
        n = max(self.inferencias, 1)
        return {
            "inferencias": self.inferencias,
            "reconocidas": self.reconocidas,
            "tasa_reconocimiento": round(self.reconocidas / n, 3),
            "aciertos": self.aciertos,
            "falsas_aceptaciones": self.falsas_aceptaciones,
            "falsos_rechazos": self.falsos_rechazos,
            "distancia_media": (
                round(self.suma_distancia_sujeto / self.n_distancia_sujeto, 3) if self.n_distancia_sujeto else None
            ),
            "mejor_distancia": round(self.mejor_distancia, 3) if self.mejor_distancia is not None else None,
        }


class CarreraPista:
    """Decide qué variante reconoce primero a una cara que aparece en cámara."""

    def __init__(self) -> None:
        self.inferencias = 0
        self.ganador: str | None = None
        self.pasos_hasta_acierto: dict[str, int | None] = {"crudo": None, "filtrado": None}

    def actualizar(self, acierto_crudo: bool, acierto_filtrado: bool) -> str | None:
        """Devuelve 'crudo', 'filtrado' o 'empate' solo la primera vez que alguna variante acierta.

        Sigue registrando los pasos de la variante rezagada, para saber cuánto tardó en acertar.
        """
        self.inferencias += 1
        if acierto_crudo and self.pasos_hasta_acierto["crudo"] is None:
            self.pasos_hasta_acierto["crudo"] = self.inferencias
        if acierto_filtrado and self.pasos_hasta_acierto["filtrado"] is None:
            self.pasos_hasta_acierto["filtrado"] = self.inferencias
        if self.ganador is not None:
            return None
        if acierto_crudo and acierto_filtrado:
            self.ganador = "empate"
        elif acierto_crudo:
            self.ganador = "crudo"
        elif acierto_filtrado:
            self.ganador = "filtrado"
        return self.ganador


@dataclass
class SesionPruebas:
    sujeto: str = ""
    crudo: BrazoStats = field(default_factory=BrazoStats)
    filtrado: BrazoStats = field(default_factory=BrazoStats)
    victorias: dict[str, int] = field(default_factory=lambda: {"crudo": 0, "filtrado": 0, "empate": 0})

    def registrar(
        self,
        res_crudo: dict[str, Any],
        res_filtrado: dict[str, Any],
    ) -> None:
        sujeto = self.sujeto or None
        for brazo, res in ((self.crudo, res_crudo), (self.filtrado, res_filtrado)):
            brazo.registrar(
                nombre=res.get("nombre"),
                distancia=float(res.get("distancia", 9.99)),
                reconocido=bool(res.get("reconocido")),
                sujeto=sujeto,
            )

    def es_acierto(self, res: dict[str, Any]) -> bool:
        """Criterio de 'detectó': con sujeto, debe ser él; sin sujeto, basta con reconocer a alguien."""
        if not res.get("reconocido"):
            return False
        if not self.sujeto:
            return True
        return str(res.get("nombre", "")).lower() == self.sujeto.lower()

    def registrar_victoria(self, ganador: str | None) -> None:
        if ganador in self.victorias:
            self.victorias[ganador] += 1

    def a_dict(self) -> dict[str, Any]:
        c_dict = self.crudo.a_dict()
        f_dict = self.filtrado.a_dict()
        vict = dict(self.victorias)
        return {
            "sujeto": self.sujeto,
            "crudo": c_dict,
            "filtrado": f_dict,
            "victorias": vict,
            "sesion": {
                "crudo": c_dict,
                "filtrado": f_dict,
            },
            "carrera": {
                "victorias_crudo": vict.get("crudo", 0),
                "victorias_filtrado": vict.get("filtrado", 0),
                "empates": vict.get("empate", 0),
            },
        }
