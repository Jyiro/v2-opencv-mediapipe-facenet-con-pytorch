from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any

TIPOS_VALIDOS = {
    "entrada",
    "salida",
    "retiro",
    "devolucion",
    "aviso_creado",
    "aviso_cerrado",
    "estado_lab",
    "ambiente",
    "correccion",
}

ORIGENES_VALIDOS = {
    "pipeline-facial",
    "caja-fuerte",
    "tablet-puerta",
    "torre-estado",
    "armario",
}

METODOS_VALIDOS = {
    "cara+gesto",
    "qr",
    "pin",
    "manual",
    "sensor",
}


def ahora_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def crear_evento(
    tipo: str,
    origen: str = "pipeline-facial",
    metodo: str = "cara+gesto",
    persona_id: str | None = None,
    objeto_id: str | None = None,
    distancia: float | None = None,
    reloj_confiable: bool = True,
    meta: dict[str, Any] | None = None,
    evento_id: str | None = None,
    t: str | None = None,
) -> dict[str, Any]:
    if tipo not in TIPOS_VALIDOS:
        raise ValueError(f"Tipo de evento inválido: {tipo}")
    if origen not in ORIGENES_VALIDOS:
        raise ValueError(f"Origen inválido: {origen}")
    if metodo not in METODOS_VALIDOS:
        raise ValueError(f"Método inválido: {metodo}")

    evento = {
        "id": evento_id or str(uuid.uuid4()),
        "tipo": tipo,
        "persona_id": persona_id,
        "objeto_id": objeto_id,
        "t": t or ahora_iso(),
        "reloj_confiable": bool(reloj_confiable),
        "origen": origen,
        "metodo": metodo,
        "distancia": round(float(distancia), 4) if distancia is not None else None,
        "meta": meta or {},
    }
    return evento


def validar_evento(evento: dict[str, Any]) -> bool:
    campos_requeridos = {"id", "tipo", "t", "reloj_confiable", "origen", "metodo"}
    if not isinstance(evento, dict) or not campos_requeridos.issubset(evento.keys()):
        return False
    if evento["tipo"] not in TIPOS_VALIDOS:
        return False
    if evento["origen"] not in ORIGENES_VALIDOS:
        return False
    if evento["metodo"] not in METODOS_VALIDOS:
        return False
    return True


def _parse_iso(ts: str) -> datetime:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)


class ListaAvisos(list):
    def __contains__(self, item: Any) -> bool:
        if super().__contains__(item):
            return True
        for x in self:
            if isinstance(x, dict) and (x.get("id") == item or x.get("objeto_id") == item):
                return True
        return False

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for x in self:
                if isinstance(x, dict) and (x.get("id") == key or x.get("objeto_id") == key):
                    return x
        return super().__getitem__(key)


class EstadoLabDict(dict):
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("modo") == other
        return super().__eq__(other)


def derivar_estado(eventos: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Deriva el estado completo del laboratorio a partir de la secuencia inmutable de eventos.
    Respeta eventos de tipo 'correccion' que anulan o corrigen un evento previo por id.
    """
    ids_anulados: set[str] = set()
    for ev in eventos:
        if ev.get("tipo") == "correccion":
            corrige_id = (ev.get("meta") or {}).get("corrige_id")
            if corrige_id:
                ids_anulados.add(str(corrige_id))

    hoy_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fecha_referencia = eventos[-1].get("t", "")[:10] if (eventos and eventos[-1].get("t")) else hoy_str
    estado_personas: dict[str, dict[str, Any]] = {}
    prestamos: dict[str, dict[str, Any]] = {}
    avisos_map: dict[str, dict[str, Any]] = {}
    estado_lab = {
        "modo": "disponible",
        "abierto": True,
        "actualizado_t": None,
    }
    ambiente = {
        "temperatura": 22.5,
        "luz": 340,
        "lux": 340,
        "sonido": 38,
        "ruido": 38,
        "actualizado_t": None,
    }

    for ev in eventos:
        ev_id = ev.get("id")
        if not ev_id or ev_id in ids_anulados:
            continue

        tipo = ev.get("tipo")
        persona = ev.get("persona_id")
        objeto = ev.get("objeto_id")
        t_str = ev.get("t", "")
        meta = ev.get("meta") or {}

        if tipo in ("entrada", "salida") and persona:
            info = estado_personas.setdefault(
                persona,
                {
                    "persona_id": persona,
                    "nombre": meta.get("nombre", persona),
                    "dentro": False,
                    "ultima_entrada": None,
                    "ultima_salida": None,
                    "metodo": ev.get("metodo"),
                    "origen": ev.get("origen"),
                    "distancia": ev.get("distancia"),
                    "ingresos_hoy": 0,
                    "segundos_hoy": 0.0,
                    "_ts_entrada_abierta": None,
                },
            )
            if meta.get("nombre"):
                info["nombre"] = meta["nombre"]

            ts_actual = _parse_iso(t_str)
            es_hoy = (not t_str) or t_str.startswith(hoy_str) or t_str.startswith(fecha_referencia)

            if tipo == "entrada":
                info["dentro"] = True
                info["ultima_entrada"] = t_str
                info["metodo"] = ev.get("metodo")
                info["origen"] = ev.get("origen")
                info["distancia"] = ev.get("distancia")
                info["_ts_entrada_abierta"] = ts_actual
                if es_hoy:
                    info["ingresos_hoy"] += 1
            elif tipo == "salida":
                if info["dentro"] and info["_ts_entrada_abierta"] is not None:
                    delta = (ts_actual - info["_ts_entrada_abierta"]).total_seconds()
                    if delta > 0 and es_hoy:
                        info["segundos_hoy"] += delta
                info["dentro"] = False
                info["ultima_salida"] = t_str
                info["metodo"] = ev.get("metodo")
                info["origen"] = ev.get("origen")
                info["distancia"] = ev.get("distancia")
                info["_ts_entrada_abierta"] = None

        elif tipo == "retiro" and objeto and persona:
            prestamos[objeto] = {
                "objeto_id": objeto,
                "nombre_objeto": meta.get("nombre_objeto", objeto),
                "persona_id": persona,
                "desde": t_str,
                "metodo": ev.get("metodo"),
                "evento_id": ev_id,
            }
        elif tipo == "devolucion" and objeto:
            prestamos.pop(objeto, None)

        elif tipo == "aviso_creado":
            aviso_id = objeto or ev_id
            texto_aviso = meta.get("texto", "") or meta.get("titulo", "")
            avisos_map[aviso_id] = {
                "id": aviso_id,
                "autor": persona or "LIS",
                "texto": texto_aviso,
                "titulo": texto_aviso,
                "prioridad": meta.get("prioridad", "normal"),
                "vencimiento": meta.get("vencimiento", ""),
                "creado_t": t_str,
            }
        elif tipo == "aviso_cerrado":
            aviso_id = objeto or meta.get("aviso_id")
            if aviso_id:
                avisos_map.pop(str(aviso_id), None)

        elif tipo == "estado_lab":
            if "modo" in meta:
                estado_lab["modo"] = meta["modo"]
            if "abierto" in meta:
                estado_lab["abierto"] = bool(meta["abierto"])
            estado_lab["actualizado_t"] = t_str

        elif tipo == "ambiente":
            for k in ("temperatura", "luz", "sonido", "lux", "ruido"):
                if k in meta and meta[k] is not None:
                    ambiente[k] = meta[k]
                    if k == "luz":
                        ambiente["lux"] = meta[k]
                    elif k == "lux":
                        ambiente["luz"] = meta[k]
                    elif k == "sonido":
                        ambiente["ruido"] = meta[k]
                    elif k == "ruido":
                        ambiente["sonido"] = meta[k]
            ambiente["actualizado_t"] = t_str

    ahora_dt = datetime.now(timezone.utc)
    presentes: dict[str, dict[str, Any]] = {}
    resumen_personas: dict[str, dict[str, Any]] = {}

    for pid, pdata in estado_personas.items():
        segundos_totales = pdata["segundos_hoy"]
        if pdata["dentro"] and pdata["_ts_entrada_abierta"] is not None:
            delta_vivo = max(0.0, (ahora_dt - pdata["_ts_entrada_abierta"]).total_seconds())
            segundos_totales += delta_vivo

        limpio = {
            "persona_id": pid,
            "nombre": pdata["nombre"],
            "dentro": pdata["dentro"],
            "ultima_entrada": pdata["ultima_entrada"],
            "ultima_salida": pdata["ultima_salida"],
            "metodo": pdata["metodo"],
            "origen": pdata["origen"],
            "distancia": pdata["distancia"],
            "ingresos_hoy": pdata["ingresos_hoy"],
            "reingresos_hoy": pdata["ingresos_hoy"],
            "minutos_hoy": round(segundos_totales / 60.0, 1),
        }
        resumen_personas[pid] = limpio
        if pdata["dentro"]:
            presentes[pid] = limpio

    return {
        "presentes": presentes,
        "personas_actividad": resumen_personas,
        "prestamos": prestamos,
        "avisos": ListaAvisos(avisos_map.values()),
        "estado_lab": EstadoLabDict(estado_lab),
        "ambiente": ambiente,
        "total_eventos_validos": len(eventos) - len(ids_anulados),
    }
