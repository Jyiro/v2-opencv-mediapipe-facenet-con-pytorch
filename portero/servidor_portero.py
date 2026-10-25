from __future__ import annotations
import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROYECTO_DIR = Path(__file__).resolve().parent.parent
if str(PROYECTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROYECTO_DIR))

from hardware.serie import ClienteFichadorSerie
from hardware.camara import (
    guardar_frame_disco_seguro,
    guardar_frame_subido_jpeg,
    obtener_frame_jpeg,
    obtener_preview_pruebas_jpeg,
)
from nucleo.base_datos import RepositorioLIS
from nucleo.eventos import ahora_iso, crear_evento, derivar_estado, validar_evento

RUTA_JSONL = PROYECTO_DIR / "datos" / "portero_eventos.jsonl"
RUTA_TABLERO = PROYECTO_DIR / "tablero" / "index.html"
RUTA_FRAME = PROYECTO_DIR / "datos" / "ultimo_frame.jpg"

repositorio = RepositorioLIS()
fichador = ClienteFichadorSerie(repositorio=repositorio, simular=True)


class GestorEventosPortero:
    def __init__(
        self,
        ruta_jsonl: Path | str | None = None,
        repositorio_db: RepositorioLIS | None = None,
    ) -> None:
        self.ruta_jsonl = Path(ruta_jsonl) if ruta_jsonl else RUTA_JSONL
        self.repo = repositorio_db
        self.lock = threading.Lock()
        self.ruta_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if not self.ruta_jsonl.exists():
            self.ruta_jsonl.touch()

    def registrar_eventos(self, lista_eventos: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        with self.lock:
            ids_existentes = set()
            if self.ruta_jsonl.exists():
                with open(self.ruta_jsonl, "r", encoding="utf-8") as f:
                    for linea in f:
                        l = linea.strip()
                        if l:
                            try:
                                ev = json.loads(l)
                                if "id" in ev:
                                    ids_existentes.add(ev["id"])
                            except Exception:
                                pass
            aceptados: list[str] = []
            duplicados: list[str] = []

            with open(self.ruta_jsonl, "a", encoding="utf-8") as f:
                for item in lista_eventos:
                    ev_id = item.get("id")
                    if not ev_id or not validar_evento(item):
                        continue
                    if ev_id in ids_existentes:
                        duplicados.append(ev_id)
                        continue
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    aceptados.append(ev_id)
                    ids_existentes.add(ev_id)
                    repo_dest = self.repo or (repositorio if self.ruta_jsonl == RUTA_JSONL else None)
                    if repo_dest is not None:
                        try:
                            repo_dest.registrar_evento(item)
                        except Exception:
                            pass
                f.flush()
                os.fsync(f.fileno())
            return aceptados, duplicados

    def todos_los_eventos(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.ruta_jsonl == RUTA_JSONL:
                return asegurar_jsonl_sincronizado()
            eventos: list[dict[str, Any]] = []
            ids: set[str] = set()
            if self.ruta_jsonl.exists():
                with open(self.ruta_jsonl, "r", encoding="utf-8") as f:
                    for linea in f:
                        l = linea.strip()
                        if l:
                            try:
                                ev = json.loads(l)
                                if validar_evento(ev) and ev["id"] not in ids:
                                    eventos.append(ev)
                                    ids.add(ev["id"])
                            except Exception:
                                pass
            return eventos

    def total_eventos(self) -> int:
        return len(self.todos_los_eventos())

    def cursor_actual(self) -> int:
        return self.total_eventos()

    def obtener_desde(self, cursor: int) -> tuple[int, list[dict[str, Any]]]:
        evs = self.todos_los_eventos()
        recorte = evs[cursor:] if cursor < len(evs) else []
        return len(evs), recorte


_gestor_portero = GestorEventosPortero()


class ServidorPortero:
    def __init__(self, host: str = "0.0.0.0", puerto: int = 8080) -> None:
        self.host = host
        self.puerto = puerto
        self.httpd: ThreadingHTTPServer | None = None
        self.hilo: threading.Thread | None = None

    def iniciar(self) -> None:
        asegurar_jsonl_sincronizado()
        self.httpd = ThreadingHTTPServer((self.host, self.puerto), ManejadorPortero)
        self.hilo = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.hilo.start()

    def detener(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def asegurar_jsonl_sincronizado() -> list[dict[str, Any]]:
    """
    Lee portero_eventos.jsonl y garantiza sincronía bidireccional con SQLite
    para que tanto eventos de cámara como del Portero estén siempre unificados.
    """
    RUTA_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if not RUTA_JSONL.exists():
        RUTA_JSONL.touch()

    eventos_jsonl: list[dict[str, Any]] = []
    ids_jsonl: set[str] = set()

    with open(RUTA_JSONL, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                ev = json.loads(linea)
                if validar_evento(ev) and ev["id"] not in ids_jsonl:
                    eventos_jsonl.append(ev)
                    ids_jsonl.add(ev["id"])
                    repositorio.registrar_evento(ev)
            except Exception:
                continue

    eventos_db = repositorio.listar_eventos(desde_seq=0, limite=5000)
    nuevos_para_jsonl = [ev for ev in eventos_db if ev["id"] not in ids_jsonl]

    if nuevos_para_jsonl:
        with open(RUTA_JSONL, "a", encoding="utf-8") as f:
            for ev in nuevos_para_jsonl:
                copia = {k: v for k, v in ev.items() if k != "seq"}
                f.write(json.dumps(copia, ensure_ascii=False) + "\n")
                eventos_jsonl.append(copia)
            f.flush()
            os.fsync(f.fileno())

    return eventos_jsonl


class ManejadorPortero(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _responder_json(self, codigo: int, payload: Any) -> None:
        cuerpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _leer_body_json(self) -> Any:
        largo = int(self.headers.get("Content-Length", "0"))
        if largo <= 0:
            return {}
        crudo = self.rfile.read(largo).decode("utf-8")
        return json.loads(crudo) if crudo else {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Device-Token")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        ruta = parsed.path
        qs = parse_qs(parsed.query)

        if ruta in ("/", "/tablero"):
            if not RUTA_TABLERO.exists():
                self._responder_json(404, {"error": "No se encontró tablero/index.html"})
                return
            contenido = RUTA_TABLERO.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(contenido)))
            self.end_headers()
            self.wfile.write(contenido)
            return

        if ruta == "/health":
            eventos = asegurar_jsonl_sincronizado()
            self._responder_json(
                200,
                {
                    "hora": ahora_iso(),
                    "version": "2.0.0-lis",
                    "cantidad_eventos": len(eventos),
                    "reloj_confiable": True,
                },
            )
            return

        if ruta == "/events":
            cursor = int(qs.get("since", ["0"])[0])
            total_cursor, recorte = _gestor_portero.obtener_desde(cursor)
            self._responder_json(
                200,
                {
                    "eventos": recorte,
                    "events": recorte,
                    "proximo_cursor": total_cursor,
                    "cursor": total_cursor,
                    "total": total_cursor,
                },
            )
            return

        if ruta == "/state":
            eventos = asegurar_jsonl_sincronizado()
            estado = derivar_estado(eventos)
            estado["perfiles"] = repositorio.listar_personas()
            estado["catalogo_inventario"] = repositorio.listar_inventario()
            estado["fichador"] = repositorio.leer_estado_runtime("fichador_lcd", fichador.estado)
            estado["enrolamiento"] = repositorio.leer_estado_runtime(
                "orden_enrolamiento", {"activo": False, "estado": "inactivo"}
            )
            estado["camara_runtime"] = repositorio.leer_estado_runtime(
                "camara_estado", {"activa": False, "fps": 0.0, "rostros_en_cuadro": 0, "fuente": "0", "resolucion": 160}
            )
            estado["orden_camara"] = repositorio.leer_estado_runtime(
                "orden_camara", {"fuente": "0"}
            )
            estado["pruebas"] = repositorio.leer_estado_runtime("pruebas_estado", {})
            estado["pruebas_config"] = repositorio.leer_estado_runtime("pruebas_config", {})
            estado["config_reconocimiento"] = repositorio.leer_estado_runtime(
                "config_reconocimiento", {"resolucion": 160, "resoluciones_disponibles": [112, 128, 160, 224]}
            )
            estado["ultimos_eventos"] = list(reversed(eventos[-40:]))
            self._responder_json(200, estado)
            return

        if ruta == "/pruebas/estado":
            st = repositorio.leer_estado_runtime("pruebas_estado", {})
            cfg = repositorio.leer_estado_runtime("pruebas_config", {})
            self._responder_json(200, {"estado": st, "config": cfg})
            return

        if ruta == "/pruebas/preview.jpg":
            jpg = obtener_preview_pruebas_jpeg()
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(204)
                self.end_headers()
            return

        if ruta == "/frame.jpg":
            jpg = obtener_frame_jpeg()
            if jpg is None and RUTA_FRAME.exists():
                try:
                    jpg = RUTA_FRAME.read_bytes()
                except (PermissionError, OSError):
                    jpg = None
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(204)
                self.end_headers()
            return

        if ruta == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = obtener_frame_jpeg()
                    if jpg is None and RUTA_FRAME.exists():
                        try:
                            jpg = RUTA_FRAME.read_bytes()
                        except (PermissionError, OSError):
                            jpg = None
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                return

        self._responder_json(404, {"error": f"Ruta no encontrada: {ruta}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        ruta = parsed.path

        if ruta == "/camara/frame_upload":
            largo = int(self.headers.get("Content-Length", "0"))
            if largo > 0:
                datos_jpeg = self.rfile.read(largo)
                guardar_frame_subido_jpeg(datos_jpeg)
                ruta_subido = PROYECTO_DIR / "datos" / "frame_subido.jpg"
                guardar_frame_disco_seguro(datos_jpeg, ruta_subido)
                repositorio.guardar_estado_runtime("orden_camara", {"fuente": "navegador", "t": ahora_iso()})
                self._responder_json(200, {"ok": True})
            else:
                self._responder_json(400, {"error": "Cuerpo vacío"})
            return

        try:
            datos = self._leer_body_json()
        except Exception as exc:
            self._responder_json(400, {"error": f"JSON inválido: {exc}"})
            return

        if ruta == "/camara/configurar":
            fuente = str(datos.get("fuente", "0")).strip()
            repositorio.guardar_estado_runtime("orden_camara", {"fuente": fuente, "t": ahora_iso()})
            self._responder_json(200, {"ok": True, "fuente": fuente})
            return

        if ruta == "/reconocimiento/resolucion":
            resolucion = int(datos.get("resolucion", 160))
            if resolucion not in (112, 128, 160, 224):
                self._responder_json(400, {"ok": False, "error": "Resolución no soportada. Use 112, 128, 160 o 224."})
                return
            nueva_cfg = {
                "resolucion": resolucion,
                "resoluciones_disponibles": [112, 128, 160, 224],
                "t": ahora_iso(),
            }
            repositorio.guardar_estado_runtime("config_reconocimiento", nueva_cfg)
            self._responder_json(200, {"ok": True, "config": nueva_cfg})
            return

        if ruta == "/events":
            lista = datos if isinstance(datos, list) else datos.get("events", [datos])
            eventos_actuales = asegurar_jsonl_sincronizado()
            ids_existentes = {e["id"] for e in eventos_actuales}
            aceptados: list[str] = []
            duplicados: list[str] = []

            for item in lista:
                if "id" not in item or "t" not in item:
                    ev = crear_evento(
                        tipo=item["tipo"],
                        origen=item.get("origen", "tablet-puerta"),
                        metodo=item.get("metodo", "manual"),
                        persona_id=item.get("persona_id"),
                        objeto_id=item.get("objeto_id"),
                        distancia=item.get("distancia"),
                        meta=item.get("meta") or {},
                    )
                else:
                    ev = item

                if not validar_evento(ev):
                    continue

                if ev["id"] in ids_existentes:
                    duplicados.append(ev["id"])
                    continue

                repositorio.registrar_evento(ev)
                ids_existentes.add(ev["id"])
                aceptados.append(ev["id"])

                if ev["tipo"] in ("entrada", "salida") and ev.get("persona_id"):
                    nombre_mostrar = (ev.get("meta") or {}).get("nombre") or ev["persona_id"]
                    fichador.notificar_evento_acceso(nombre_mostrar, ev["tipo"], ev.get("distancia"))

            asegurar_jsonl_sincronizado()
            self._responder_json(200, {"aceptados": aceptados, "duplicados": duplicados})
            return

        if ruta == "/auth/pin":
            usuario = str(datos.get("persona_id", "")).strip()
            pin = str(datos.get("pin", "")).strip()
            tipo_accion = datos.get("tipo", "entrada")
            origen = datos.get("origen", "tablet-puerta")
            metodo = datos.get("metodo", "pin")

            perfil = repositorio.verificar_pin(usuario, pin)
            if not perfil:
                fichador.notificar_evento_acceso("PIN INVALIDO", "error")
                self._responder_json(401, {"ok": False, "error": "PIN incorrecto o usuario inexistente."})
                return

            ev = crear_evento(
                tipo=tipo_accion,
                origen=origen,
                metodo=metodo,
                persona_id=perfil["nombre"],
                meta={"nombre": perfil["nombre"], "proyecto": perfil["proyecto"], "rol": perfil["rol"]},
            )
            repositorio.registrar_evento(ev)
            asegurar_jsonl_sincronizado()
            fichador.notificar_evento_acceso(perfil["nombre"], tipo_accion, None)
            self._responder_json(200, {"ok": True, "evento": ev, "perfil": perfil})
            return

        if ruta == "/perfiles":
            nombre = str(datos.get("nombre", "")).strip()
            rol = str(datos.get("rol", "Integrante")).strip()
            proyecto = str(datos.get("proyecto", "LIS")).strip()
            pin = str(datos.get("pin", "1111")).strip()
            consentimiento = bool(datos.get("consentimiento", True))
            enrolar_ahora = bool(datos.get("enrolar_en_camara", False))

            if not consentimiento:
                self._responder_json(
                    400,
                    {"ok": False, "error": "El consentimiento informado es obligatorio (Sección 13)."},
                )
                return

            perfil = repositorio.crear_o_actualizar_persona(
                nombre=nombre, rol=rol, proyecto=proyecto, pin=pin, consentimiento=consentimiento
            )
            orden = None
            if enrolar_ahora:
                orden = repositorio.solicitar_enrolamiento_camara(nombre=perfil["nombre"], muestras=5)
            self._responder_json(200, {"ok": True, "perfil": perfil, "orden_enrolamiento": orden})
            return

        if ruta == "/pruebas/configurar":
            cfg_actual = repositorio.leer_estado_runtime("pruebas_config", {})
            activo = bool(datos.get("activo", cfg_actual.get("activo", False)))
            sujeto = str(datos.get("sujeto", cfg_actual.get("sujeto", ""))).strip()
            usar_en_rec = bool(datos.get("usar_en_reconocimiento", cfg_actual.get("usar_en_reconocimiento", False)))
            reiniciar = bool(datos.get("reiniciar", False))
            ilum = datos.get("iluminacion") if "iluminacion" in datos else cfg_actual.get("iluminacion", {})
            resolucion = int(datos.get("resolucion", cfg_actual.get("resolucion", 160)))

            token = cfg_actual.get("token_reinicio", "")
            if reiniciar or not token:
                token = uuid.uuid4().hex[:8]
                repositorio.guardar_estado_runtime("pruebas_estado", {})

            nueva_cfg = {
                "activo": activo,
                "sujeto": sujeto,
                "usar_en_reconocimiento": usar_en_rec,
                "token_reinicio": token,
                "iluminacion": ilum,
                "resolucion": resolucion,
                "t": ahora_iso(),
            }
            repositorio.guardar_estado_runtime("pruebas_config", nueva_cfg)
            self._responder_json(200, {"ok": True, "config": nueva_cfg})
            return

        if ruta == "/perfiles/alta_guiada":
            nombre = str(datos.get("nombre", "")).strip()
            rol = str(datos.get("rol", "Integrante")).strip()
            proyecto = str(datos.get("proyecto", "LIS")).strip()
            pin = str(datos.get("pin", "1111")).strip()
            consentimiento = bool(datos.get("consentimiento", True))
            poses = int(datos.get("poses", 5))
            modo = str(datos.get("modo", "normal")).strip()

            if not consentimiento:
                self._responder_json(
                    400,
                    {"ok": False, "error": "El consentimiento informado es obligatorio (Sección 13)."},
                )
                return

            if not nombre:
                self._responder_json(400, {"ok": False, "error": "Falta indicar el nombre del perfil."})
                return

            perfil = repositorio.crear_o_actualizar_persona(
                nombre=nombre, rol=rol, proyecto=proyecto, pin=pin, consentimiento=consentimiento
            )
            orden = repositorio.solicitar_enrolamiento_camara(
                nombre=perfil["nombre"],
                guiado=True,
                poses=poses,
                modo=modo,
            )
            fichador.cambiar_rgb(245, 158, 11)
            fichador.mostrar_mensaje("ALTA GUIADA", perfil["nombre"][:16])
            self._responder_json(200, {"ok": True, "perfil": perfil, "orden_enrolamiento": orden})
            return

        if ruta == "/perfiles/enrolar":
            nombre = str(datos.get("nombre", "")).strip()
            if not nombre:
                self._responder_json(400, {"ok": False, "error": "Falta indicar el nombre del perfil."})
                return
            perfil = repositorio.obtener_persona(nombre)
            if not perfil:
                perfil = repositorio.crear_o_actualizar_persona(nombre=nombre)
            guiado = bool(datos.get("guiado", False))
            poses = int(datos.get("poses", 5))
            modo = str(datos.get("modo", "normal"))
            orden = repositorio.solicitar_enrolamiento_camara(
                nombre=perfil["nombre"],
                muestras=poses if not guiado else 5,
                guiado=guiado,
                poses=poses,
                modo=modo,
            )
            fichador.cambiar_rgb(245, 158, 11)
            fichador.mostrar_mensaje("ALTA GUIADA" if guiado else "MODO ALTA CARA", perfil["nombre"][:16])
            self._responder_json(200, {"ok": True, "orden": orden})
            return

        if ruta == "/perfiles/eliminar":
            nombre = str(datos.get("nombre", "")).strip()
            ok = repositorio.eliminar_persona(nombre)
            self._responder_json(200, {"ok": ok})
            return

        if ruta in ("/perfiles/cancelar_enrolamiento", "/perfiles/cancelar"):
            repositorio.guardar_estado_runtime(
                "orden_enrolamiento",
                {
                    "activo": False,
                    "estado": "cancelado",
                    "mensaje": "Captura biométrica cancelada por el usuario.",
                    "t": ahora_iso(),
                },
            )
            fichador.cambiar_rgb(56, 189, 248)
            fichador.mostrar_mensaje("ENROLAMIENTO", "CANCELADO")
            self._responder_json(200, {"ok": True, "mensaje": "Captura biométrica cancelada exitosamente."})
            return

        if ruta == "/hardware/touch":
            repositorio.guardar_estado_runtime("touch_wake", {"activo": True, "t": ahora_iso()})
            fichador.cambiar_rgb(16, 185, 129)
            fichador.mostrar_mensaje("CAMARA ACTIVA", "HAGA SU GESTO")
            fichador.sonar_beep("ok")
            self._responder_json(200, {"ok": True})
            return

        self._responder_json(404, {"error": f"Endpoint POST desconocido: {ruta}"})


def iniciar_servidor(puerto: int = 8080) -> ThreadingHTTPServer:
    asegurar_jsonl_sincronizado()
    servidor = ThreadingHTTPServer(("0.0.0.0", puerto), ManejadorPortero)
    logging.info("Portero HTTP y Tablero escuchando en http://0.0.0.0:%d/tablero", puerto)
    return servidor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    srv = iniciar_servidor(8080)
    srv.serve_forever()
