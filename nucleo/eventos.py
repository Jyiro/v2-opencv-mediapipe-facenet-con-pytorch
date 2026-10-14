"""Módulo de modelado de eventos inmutables y derivación de estado (Event Sourcing).

Implementa el esquema oficial de la Sección 6 de proyecto.md y la lógica para
derivar en tiempo real: personas presentes, préstamos de equipos, avisos activos
y estado ambiental del laboratorio.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any
import uuid

logger = logging.getLogger(__name__)

TIPOS_EVENTO_VALIDOS = {
    "entrada",
    "salida",
    "retiro",
    "devolucion",
    "aviso_creado",
    "aviso_cerrado",
    "estado_lab",
    "ambiente",
    "correccion",
    "asistencia_permanencia",
}

ORIGENES_VALIDOS = {
    "pipeline-facial",
    "caja-fuerte",
    "tablet-puerta",
    "torre-estado",
    "armario",
    "web-admin",
}

METODOS_VALIDOS = {
    "cara+gesto",
    "qr",
    "pin",
    "manual",
    "sensor",
    "cara_pasiva",
}


def crear_evento(
    tipo: str,
    persona_id: str | None = None,
    objeto_id: str | None = None,
    origen: str = "pipeline-facial",
    metodo: str = "cara+gesto",
    distancia: float | None = None,
    meta: dict[str, Any] | None = None,
    reloj_confiable: bool = True,
    t: str | None = None,
    evento_id: str | None = None,
) -> dict[str, Any]:
    """Crea y valida un diccionario de evento inmutable según la Sección 6."""
    if tipo not in TIPOS_EVENTO_VALIDOS:
        raise ValueError(f"Tipo de evento inválido: {tipo}. Válidos: {TIPOS_EVENTO_VALIDOS}")

    id_evento = evento_id or str(uuid.uuid4())
    t_evento = t or datetime.now(timezone.utc).isoformat()

    evento = {
        "id": id_evento,
        "tipo": tipo,
        "persona_id": str(persona_id) if persona_id else None,
        "objeto_id": str(objeto_id) if objeto_id else None,
        "t": t_evento,
        "reloj_confiable": bool(reloj_confiable),
        "origen": origen,
        "metodo": metodo,
        "distancia": round(float(distancia), 4) if distancia is not None else None,
        "meta": meta or {},
    }
    return evento


def validar_evento(evento: dict[str, Any]) -> bool:
    """Verifica que un diccionario contenga la estructura y tipos requeridos."""
    campos_requeridos = ["id", "tipo", "t", "reloj_confiable", "origen", "metodo"]
    for campo in campos_requeridos:
        if campo not in evento:
            return False
    if evento["tipo"] not in TIPOS_EVENTO_VALIDOS:
        return False
    return True


def derivar_estado(lista_eventos: list[dict[str, Any]]) -> dict[str, Any]:
    """Procesa una secuencia de eventos inmutables y deriva el estado actual completo.

    Args:
        lista_eventos: Lista de eventos append-only.

    Returns:
        Diccionario con:
        - "presentes": {persona_id: {"hora_ingreso": ..., "metodo": ..., "distancia": ..., "reingresos_hoy": ...}}
        - "prestamos": {objeto_id: {"persona_id": ..., "t": ...}}
        - "avisos": {aviso_id: {"titulo": ..., "dueño": ..., "vencimiento": ..., "t": ...}}
        - "estado_lab": modo actual ("disponible", "reunion", "no_molestar")
        - "ambiente": {"lux": ..., "temperatura": ..., "ruido": ...}
        - "total_eventos": int
    """
    # Identificar IDs de eventos anulados por correcciones
    eventos_anulados: set[str] = set()
    for ev in lista_eventos:
        if ev.get("tipo") == "correccion":
            ref = ev.get("meta", {}).get("corrige_id") or ev.get("objeto_id")
            if ref:
                eventos_anulados.add(ref)

    # Ordenar cronológicamente
    eventos_validos = [ev for ev in lista_eventos if ev.get("id") not in eventos_anulados]
    eventos_validos.sort(key=lambda x: str(x.get("t", "")))

    presentes: dict[str, dict[str, Any]] = {}
    historial_ingresos_dia: dict[str, int] = {}
    prestamos: dict[str, dict[str, Any]] = {}
    avisos: dict[str, dict[str, Any]] = {}
    estado_lab = "disponible"
    ambiente: dict[str, Any] = {"lux": 300, "temperatura": 22.0, "ruido": 45}

    for ev in eventos_validos:
        tipo = ev.get("tipo")
        pid = ev.get("persona_id")
        oid = ev.get("objeto_id")
        t = ev.get("t")
        meta = ev.get("meta", {})

        if tipo == "entrada" and pid:
            historial_ingresos_dia[pid] = historial_ingresos_dia.get(pid, 0) + 1
            presentes[pid] = {
                "persona_id": pid,
                "nombre": meta.get("nombre", pid),
                "hora_ingreso": t,
                "metodo": ev.get("metodo"),
                "distancia": ev.get("distancia"),
                "reingresos_hoy": historial_ingresos_dia[pid],
            }
        elif tipo == "salida" and pid:
            presentes.pop(pid, None)

        elif tipo == "retiro" and oid and pid:
            prestamos[oid] = {
                "objeto_id": oid,
                "persona_id": pid,
                "nombre_persona": meta.get("nombre_persona", pid),
                "descripcion": meta.get("descripcion", oid),
                "hora_retiro": t,
            }
        elif tipo == "devolucion" and oid:
            prestamos.pop(oid, None)

        elif tipo == "aviso_creado" and oid:
            avisos[oid] = {
                "id": oid,
                "titulo": meta.get("titulo", oid),
                "dueño": pid or meta.get("dueño", "LIS"),
                "vencimiento": meta.get("vencimiento"),
                "creado": t,
            }
        elif tipo == "aviso_cerrado" and oid:
            avisos.pop(oid, None)

        elif tipo == "estado_lab":
            estado_lab = meta.get("modo", oid or estado_lab)

        elif tipo == "ambiente":
            if "lux" in meta:
                ambiente["lux"] = meta["lux"]
            if "temperatura" in meta:
                ambiente["temperatura"] = meta["temperatura"]
            if "ruido" in meta:
                ambiente["ruido"] = meta["ruido"]

    return {
        "presentes": presentes,
        "prestamos": prestamos,
        "avisos": avisos,
        "estado_lab": estado_lab,
        "ambiente": ambiente,
        "total_eventos": len(lista_eventos),
    }
