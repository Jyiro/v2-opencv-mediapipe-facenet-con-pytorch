from __future__ import annotations
import logging
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import cv2 as cv
import mediapipe as mp
import numpy as np

PROYECTO_DIR = Path(__file__).resolve().parent.parent
MEDIAPIPE_DIR = Path(__file__).resolve().parent
if str(PROYECTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROYECTO_DIR))

from hardware.camara import (
    guardar_frame_disco_seguro,
    guardar_frame_jpeg,
    normalizar_origen_camara,
    obtener_frame_subido_jpeg,
)
from hardware.serie import ClienteFichadorSerie
from nucleo.base_datos import RepositorioLIS
from nucleo.comparacion import consultar_embeddings
from nucleo.eventos import crear_evento, derivar_estado
from nucleo.iluminacion import evaluar_calidad
from nucleo.pose import EstimadorPose, GuiaDePoses, PoseCabeza, pasos_para
from nucleo.reconocimiento import (
    RESOLUCIONES_SOPORTADAS,
    ExtractorFaceNetRapido,
    nombre_modelo_resolucion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def cargar_configuracion() -> dict[str, Any]:
    ruta_cfg = PROYECTO_DIR / "config.toml"
    if ruta_cfg.exists():
        with open(ruta_cfg, "rb") as f:
            return tomllib.load(f)
    return {}


def _ascii_seguro(texto: str) -> str:
    """Convierte acentos a ASCII básico para renderizado seguro en fuentes Hershey de OpenCV."""
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N", "ü": "u", "Ü": "U", "…": "...",
    }
    res = str(texto or "")
    for orig, dest in reemplazos.items():
        res = res.replace(orig, dest)
    return "".join(c for c in res if ord(c) < 128)


class PistaRostro:
    def __init__(self, bbox: tuple[int, int, int, int], centro: tuple[int, int]) -> None:
        self.bbox = bbox
        self.centro = centro
        self.frames_estables = 1
        self.frames_desde_inferencia = 999
        self.reconocido = False
        self.nombre = "Estabilizando..."
        self.distancia = 9.99
        self.ultimo_embedding: np.ndarray | None = None
        self.gesto_actual: str | None = None
        self.frames_gesto = 0


class PipelineReconocimientoLIS:
    def __init__(self, fuente_camara: int | str | None = None, mostrar_ventana: bool = True) -> None:
        self.config = cargar_configuracion()
        params = self.config.get("parametros", {})
        modelos_cfg = self.config.get("modelos", {})
        cam_cfg = self.config.get("camara", {})

        self.umbral: float = float(params.get("umbral_reconocimiento", 0.68))
        self.metodo_comp: str = str(params.get("metodo_comparacion", "minimo"))
        self.margen_separacion: float = float(params.get("margen_separacion", 0.04))
        self.pad_crop_cara: float = float(params.get("pad_crop_cara", 0.06))
        res_cfg = int(params.get("resolucion_recorte", 160))
        self.resolucion_recorte: int = res_cfg if res_cfg in RESOLUCIONES_SOPORTADAS else 160
        self.frames_estable_req: int = int(params.get("frames_cara_estable", 3))
        self.tolerancia_px: int = int(params.get("tolerancia_movimiento_px", 45))
        self.frames_gesto_req: int = int(params.get("frames_confirmacion_gesto", 5))
        self.cooldown_seg: float = float(params.get("cooldown_gesto_segundos", 4.0))
        self.sostener_pose_seg: float = float(params.get("sostener_pose_segundos", 0.5))
        self.proyecto: str = str(params.get("proyecto", "LIS"))

        self.mostrar_ventana = mostrar_ventana and bool(cam_cfg.get("mostrar_ventana_local", True))
        self.espejo = bool(cam_cfg.get("espejo", True))

        url_stream = str(cam_cfg.get("url_stream", "")).strip()
        self.fuente_camara = fuente_camara if fuente_camara is not None else (url_stream or int(cam_cfg.get("indice", 0)))

        self.repositorio = RepositorioLIS()
        self.fichador = ClienteFichadorSerie(repositorio=self.repositorio, simular=True)
        self.extractor = ExtractorFaceNetRapido(resolucion=self.resolucion_recorte)

        # Diccionario Hot-Swap: {112: {...}, 128: {...}, 160: {...}, 224: {...}}
        self.galerias_por_resolucion: dict[int, dict[str, list[np.ndarray]]] = {}
        self._recargar_galerias()
        self.repositorio.guardar_estado_runtime(
            "config_reconocimiento",
            {"resolucion": self.resolucion_recorte, "resoluciones_disponibles": list(RESOLUCIONES_SOPORTADAS)},
        )

        self.ultimo_disparo_por_persona: dict[str, float] = {}
        self.banner_feedback: tuple[str, tuple[int, int, int], float] | None = None

        self.ruta_modelo_caras = MEDIAPIPE_DIR / modelos_cfg.get("detector_caras", "blaze_face_short_range.tflite")
        self.ruta_modelo_manos = MEDIAPIPE_DIR / modelos_cfg.get("reconocedor_manos", "manos/gesture_recognizer.task")
        self.ruta_frame_compartido = PROYECTO_DIR / "datos" / "ultimo_frame.jpg"
        self.ruta_frame_compartido.parent.mkdir(parents=True, exist_ok=True)

        self.ultimas_caras_mp: Any = None
        self.ultimas_manos_mp: Any = None
        self.pistas: list[PistaRostro] = []

        # Buffers multi-resolución para el alta: {112: [emb1, ...], 128: [...], 160: [...], 224: [...]}
        self.buffers_alta_multires: dict[int, list[np.ndarray]] = {r: [] for r in RESOLUCIONES_SOPORTADAS}
        self.nombre_enrolando: str | None = None
        self.ultimo_frame_captura_alta: int = 0

        # Alta guiada por poses
        self.ruta_landmarks = MEDIAPIPE_DIR / modelos_cfg.get("landmarks_caras", "face_landmarker.task")
        self._landmarker: Any = None
        self.estimador_pose = EstimadorPose(espejo=self.espejo)
        self.guia: GuiaDePoses | None = None
        self.hud_guia: dict[str, Any] | None = None

    def _recargar_galerias(self) -> None:
        """Carga en RAM los bancos de embeddings de las 4 resoluciones para permitir Hot-Swap instantáneo."""
        for r in RESOLUCIONES_SOPORTADAS:
            clave_mod = nombre_modelo_resolucion(r)
            # con_fallback=True permite que perfiles antiguos (solo en 160) sigan funcionando
            self.galerias_por_resolucion[r] = self.repositorio.cargar_diccionario_embeddings(
                modelo=clave_mod, con_fallback=True
            )

    @property
    def base_embeddings_activa(self) -> dict[str, list[np.ndarray]]:
        return self.galerias_por_resolucion.get(self.resolucion_recorte, {})

    def _sincronizar_config_reconocimiento(self) -> None:
        """Verifica si desde la interfaz web se solicitó un Hot-Swap de resolución."""
        cfg = self.repositorio.leer_estado_runtime("config_reconocimiento", {})
        if not cfg:
            return
        res_solicitada = int(cfg.get("resolucion", self.resolucion_recorte))
        if res_solicitada in RESOLUCIONES_SOPORTADAS and res_solicitada != self.resolucion_recorte:
            self.resolucion_recorte = res_solicitada
            self.extractor.fijar_resolucion(res_solicitada)
            # Forzar re-inferencia inmediata de todos los rostros en cámara con la nueva resolución
            for p in self.pistas:
                p.frames_desde_inferencia = 999
            self.banner_feedback = (
                f"HOT-SWAP ACTIVO: Resolucion cambiada a {res_solicitada}x{res_solicitada}px",
                (15, 110, 210),
                time.time() + 3.0,
            )
            logging.info("Hot-Swap de resolución aplicado: %dx%d px", res_solicitada, res_solicitada)

    def _cb_caras(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        self.ultimas_caras_mp = result

    def _cb_manos(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        self.ultimas_manos_mp = result

    def _abrir_camara(self) -> cv.VideoCapture:
        fuente_norm = normalizar_origen_camara(self.fuente_camara)
        if isinstance(fuente_norm, str) and not fuente_norm.isdigit():
            if fuente_norm.lower() in ("simulada", "simulado", "virtual", "mock"):
                logging.info("Iniciando en modo cámara simulada (sin hardware).")
                return cv.VideoCapture()
            logging.info("Abriendo stream de cámara en %s", fuente_norm)
            try:
                cap = cv.VideoCapture(fuente_norm)
                return cap
            except Exception as exc:
                logging.warning("Excepción al conectar con stream %s: %s", fuente_norm, exc)
                return cv.VideoCapture()
        try:
            idx = int(fuente_norm)
            backend = str(self.config.get("camara", {}).get("backend", "auto")).lower()
            if backend == "dshow" and sys.platform.startswith("win"):
                cap = cv.VideoCapture(idx, cv.CAP_DSHOW)
            elif backend == "v4l2" and sys.platform.startswith("linux"):
                cap = cv.VideoCapture(idx, cv.CAP_V4L2)
            else:
                cap = cv.VideoCapture(idx)
            if not cap.isOpened():
                cap = cv.VideoCapture(idx)
            return cap
        except Exception as exc:
            logging.warning("Excepción al abrir cámara %s: %s.", self.fuente_camara, exc)
            return cv.VideoCapture()

    def _obtener_landmarker(self) -> Any:
        if self._landmarker is not None:
            return self._landmarker
        if not self.ruta_landmarks.exists():
            logging.warning("No se encontró el modelo FaceLandmarker en %s", self.ruta_landmarks)
            self._landmarker = False
            return False
        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.ruta_landmarks)),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            return self._landmarker
        except Exception:
            logging.exception("Fallo al inicializar FaceLandmarker")
            self._landmarker = False
            return False

    def _cerrar_landmarker(self) -> None:
        if self._landmarker and hasattr(self._landmarker, "close"):
            try:
                self._landmarker.close()
            except Exception:
                pass
        self._landmarker = None

    def _actualizar_pistas(self, frame_bgr: np.ndarray, cajas_detectadas: list[tuple[int, int, int, int]]) -> list[PistaRostro]:
        nuevas_pistas: list[PistaRostro] = []
        cajas_asignadas = set()

        for pista in self.pistas:
            mejor_idx = -1
            menor_dist = 999999.0
            cx, cy = pista.centro
            for idx_caja, caja in enumerate(cajas_detectadas):
                if idx_caja in cajas_asignadas:
                    continue
                x, y, w, h = caja
                c_caja = (x + w // 2, y + h // 2)
                dist = ((cx - c_caja[0]) ** 2 + (cy - c_caja[1]) ** 2) ** 0.5
                if dist < menor_dist and dist <= self.tolerancia_px:
                    menor_dist = dist
                    mejor_idx = idx_caja

            if mejor_idx != -1:
                cajas_asignadas.add(mejor_idx)
                x, y, w, h = cajas_detectadas[mejor_idx]
                cx, cy = x + w // 2, y + h // 2
                pista.bbox = (x, y, w, h)
                pista.centro = (cx, cy)
                pista.frames_estables += 1
                pista.frames_desde_inferencia += 1
                nuevas_pistas.append(pista)

        for idx_caja, caja in enumerate(cajas_detectadas):
            if idx_caja not in cajas_asignadas:
                x, y, w, h = caja
                nuevas_pistas.append(PistaRostro((x, y, w, h), (x + w // 2, y + h // 2)))

        for pista in nuevas_pistas:
            x, y, w, h = pista.bbox
            debe_inferir = (
                pista.frames_estables == self.frames_estable_req
                or (pista.frames_estables > self.frames_estable_req and pista.frames_desde_inferencia >= 15)
                or (self.nombre_enrolando is not None and pista.frames_estables >= 2)
            )

            if debe_inferir:
                fh, fw = frame_bgr.shape[:2]
                pad_x, pad_y = int(w * self.pad_crop_cara), int(h * self.pad_crop_cara)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
                recorte = frame_bgr[y1:y2, x1:x2]

                if recorte is not None and recorte.size > 0:
                    emb = self.extractor.generar_embedding(recorte, resolucion=self.resolucion_recorte)
                    if emb is not None:
                        pista.ultimo_embedding = emb
                        pista.frames_desde_inferencia = 0
                        res = consultar_embeddings(
                            emb,
                            self.base_embeddings_activa,
                            self.umbral,
                            self.metodo_comp,
                            margen_separacion=self.margen_separacion,
                        )
                        pista.reconocido = res["reconocido"]
                        pista.nombre = res["nombre"]
                        pista.distancia = res["distancia"]

        self.pistas = nuevas_pistas
        return self.pistas

    def _limpiar_buffers_alta(self) -> None:
        for r in RESOLUCIONES_SOPORTADAS:
            self.buffers_alta_multires[r].clear()

    def _persistir_alta_multiresolucion(self, nombre_objetivo: str) -> int:
        """Guarda las muestras capturadas en las 4 resoluciones dentro de SQLite y recarga las galerías."""
        total_muestras = len(self.buffers_alta_multires.get(160, []))
        for r in RESOLUCIONES_SOPORTADAS:
            vectores_r = self.buffers_alta_multires.get(r, [])
            if vectores_r:
                self.repositorio.guardar_embeddings_persona(
                    nombre=nombre_objetivo,
                    embeddings=vectores_r,
                    modelo=nombre_modelo_resolucion(r),
                    reemplazar=True,
                    sincronizar_csv=(r == 160),
                )
        self._recargar_galerias()
        # Forzar re-evaluación inmediata en el próximo frame
        for p in self.pistas:
            p.frames_desde_inferencia = 999
        return total_muestras

    def _procesar_enrolamiento_activo(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        orden = self.repositorio.leer_estado_runtime("orden_enrolamiento", {"activo": False})
        if not orden.get("activo"):
            if self.nombre_enrolando is not None:
                self.nombre_enrolando = None
                self._limpiar_buffers_alta()
                self.guia = None
                self.hud_guia = None
                self._cerrar_landmarker()
            return

        nombre_objetivo = str(orden.get("nombre", "")).strip()
        es_guiado = bool(orden.get("guiado", False))
        muestras_totales = int(orden.get("muestras_totales", 5))

        if self.nombre_enrolando != nombre_objetivo:
            self.nombre_enrolando = nombre_objetivo
            self._limpiar_buffers_alta()
            self._recargar_galerias()
            if es_guiado:
                self.guia = GuiaDePoses(pasos=pasos_para(muestras_totales), sostener_seg=self.sostener_pose_seg)
                self.estimador_pose.reiniciar()

        if not self.pistas:
            orden["mensaje"] = f"Buscando rostro de {nombre_objetivo} en cámara..."
            self.hud_guia = None
            self.repositorio.guardar_estado_runtime("orden_enrolamiento", orden)
            return

        pista_principal = max(self.pistas, key=lambda p: p.bbox[2] * p.bbox[3])
        x, y, w, h = pista_principal.bbox
        fh, fw = frame_bgr.shape[:2]
        pad_x, pad_y = int(w * self.pad_crop_cara), int(h * self.pad_crop_cara)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
        recorte_bgr = frame_bgr[y1:y2, x1:x2]

        if recorte_bgr is None or recorte_bgr.size == 0:
            return

        calidad = evaluar_calidad(recorte_bgr)
        muestras_actuales = len(self.buffers_alta_multires[160])

        if es_guiado:
            landmarker = self._obtener_landmarker()
            pose: PoseCabeza | None = None
            landmarks = None
            if landmarker:
                try:
                    rgb_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    res_lm = landmarker.detect(mp_img)
                    if res_lm and res_lm.face_landmarks:
                        landmarks = res_lm.face_landmarks[0]
                        pose = self.estimador_pose.estimar(landmarks)
                except Exception:
                    pass

            if self.guia is None:
                self.guia = GuiaDePoses(pasos=pasos_para(muestras_totales), sostener_seg=self.sostener_pose_seg)

            res_guia = self.guia.actualizar(pose, calidad_ok=calidad["ok"], motivo_calidad=calidad["motivo"])

            self.hud_guia = {
                "bbox": pista_principal.bbox,
                "instruccion": res_guia.instruccion,
                "mensaje": res_guia.mensaje,
                "coincide": res_guia.coincide,
                "progreso": res_guia.progreso_sostener,
                "indice": res_guia.indice,
                "total": res_guia.total,
                "hechas": res_guia.hechas,
            }

            orden.update({
                "estado": "guiado",
                "instruccion": res_guia.instruccion,
                "mensaje": res_guia.mensaje,
                "coincide": res_guia.coincide,
                "progreso_sostener": round(res_guia.progreso_sostener, 2),
                "muestras_capturadas": muestras_actuales,
                "muestras_totales": res_guia.total,
                "indice_paso": res_guia.indice,
                "total_pasos": res_guia.total,
                "hechas": res_guia.hechas,
            })
            self.repositorio.guardar_estado_runtime("orden_enrolamiento", orden)
            self.fichador.mostrar_mensaje(f"P{res_guia.indice+1}/{res_guia.total} {res_guia.instruccion[:10]}", res_guia.mensaje[:16])

            if res_guia.capturar:
                if res_guia.clave == "frente" and landmarks:
                    self.estimador_pose.fijar_referencia_frontal(landmarks)

                embs_multi = self.extractor.generar_embeddings_multiresolucion(recorte_bgr)
                emb_160 = embs_multi.get(160)

                if emb_160 is not None:
                    if muestras_actuales == 0:
                        otras_caras = {
                            k: v for k, v in self.galerias_por_resolucion.get(160, {}).items()
                            if k.lower() != nombre_objetivo.lower()
                        }
                        chequeo = consultar_embeddings(emb_160, otras_caras, self.umbral, self.metodo_comp)
                        if chequeo["reconocido"]:
                            duplicado_de = chequeo["nombre"]
                            msg_err = f"Rostro ya registrado bajo '{duplicado_de}' (d={chequeo['distancia']:.2f})."
                            logging.warning(msg_err)
                            self.repositorio.guardar_estado_runtime(
                                "orden_enrolamiento",
                                {
                                    "activo": False,
                                    "nombre": nombre_objetivo,
                                    "estado": "error_duplicado",
                                    "mensaje": msg_err,
                                },
                            )
                            self.banner_feedback = (msg_err, (0, 0, 230), time.time() + 4.0)
                            self.fichador.notificar_evento_acceso("DUPLICADO", "error")
                            self.nombre_enrolando = None
                            self._limpiar_buffers_alta()
                            self.guia = None
                            self.hud_guia = None
                            self._cerrar_landmarker()
                            return

                    for r, vec in embs_multi.items():
                        self.buffers_alta_multires[r].append(vec)
                    self.fichador.sonar_beep("ok")

            if res_guia.completado:
                guardadas = self._persistir_alta_multiresolucion(nombre_objetivo)
                msg_ok = f"Alta de {nombre_objetivo} lista ({guardadas} poses guardadas en 112, 128, 160 y 224px)."
                self.repositorio.guardar_estado_runtime(
                    "orden_enrolamiento",
                    {
                        "activo": False,
                        "nombre": nombre_objetivo,
                        "muestras_totales": res_guia.total,
                        "muestras_capturadas": guardadas,
                        "estado": "completado",
                        "mensaje": msg_ok,
                    },
                )
                self.fichador.cambiar_rgb(16, 185, 129)
                self.fichador.mostrar_mensaje("ALTA GUIADA OK", nombre_objetivo[:16])
                self.fichador.sonar_beep("ok")
                self.banner_feedback = (msg_ok, (16, 185, 129), time.time() + 4.0)
                self.nombre_enrolando = None
                self._limpiar_buffers_alta()
                self.guia = None
                self.hud_guia = None
                self._cerrar_landmarker()
        else:
            if frame_idx - self.ultimo_frame_captura_alta < 8:
                return

            embs_multi = self.extractor.generar_embeddings_multiresolucion(recorte_bgr)
            emb_160 = embs_multi.get(160)
            if emb_160 is None:
                return

            if muestras_actuales == 0:
                otras_caras = {
                    k: v for k, v in self.galerias_por_resolucion.get(160, {}).items()
                    if k.lower() != nombre_objetivo.lower()
                }
                chequeo = consultar_embeddings(emb_160, otras_caras, self.umbral, self.metodo_comp)
                if chequeo["reconocido"]:
                    duplicado_de = chequeo["nombre"]
                    msg_err = f"Rostro ya registrado bajo '{duplicado_de}' (d={chequeo['distancia']:.2f})."
                    logging.warning(msg_err)
                    self.repositorio.guardar_estado_runtime(
                        "orden_enrolamiento",
                        {
                            "activo": False,
                            "nombre": nombre_objetivo,
                            "muestras_totales": muestras_totales,
                            "muestras_capturadas": 0,
                            "estado": "error_duplicado",
                            "mensaje": msg_err,
                        },
                    )
                    self.banner_feedback = (msg_err, (0, 0, 230), time.time() + 4.0)
                    self.fichador.notificar_evento_acceso("DUPLICADO", "error")
                    self.nombre_enrolando = None
                    self._limpiar_buffers_alta()
                    return

            for r, vec in embs_multi.items():
                self.buffers_alta_multires[r].append(vec)
            self.ultimo_frame_captura_alta = frame_idx
            capturadas = len(self.buffers_alta_multires[160])

            if capturadas < muestras_totales:
                orden["muestras_capturadas"] = capturadas
                orden["estado"] = "capturando"
                orden["mensaje"] = f"Capturando {nombre_objetivo}: muestra {capturadas}/{muestras_totales} (Multi-Res: 112/128/160/224px)"
                self.repositorio.guardar_estado_runtime("orden_enrolamiento", orden)
            else:
                guardadas = self._persistir_alta_multiresolucion(nombre_objetivo)
                msg_ok = f"Perfil de {nombre_objetivo} guardado ({guardadas} muestras en 112, 128, 160 y 224px)."
                self.repositorio.guardar_estado_runtime(
                    "orden_enrolamiento",
                    {
                        "activo": False,
                        "nombre": nombre_objetivo,
                        "muestras_totales": muestras_totales,
                        "muestras_capturadas": guardadas,
                        "estado": "completado",
                        "mensaje": msg_ok,
                    },
                )
                self.fichador.cambiar_rgb(16, 185, 129)
                self.fichador.mostrar_mensaje("ALTA COMPLETA", nombre_objetivo[:16])
                self.fichador.sonar_beep("ok")
                self.banner_feedback = (msg_ok, (16, 185, 129), time.time() + 4.0)
                self.nombre_enrolando = None
                self._limpiar_buffers_alta()

    def _evaluar_gestos_y_eventos(self, frame_bgr: np.ndarray, estados_presentes: dict[str, Any]) -> None:
        if not self.ultimas_manos_mp or not self.ultimas_manos_mp.gestures:
            for p in self.pistas:
                p.frames_gesto = max(0, p.frames_gesto - 1)
            return

        fh, fw = frame_bgr.shape[:2]
        for idx_mano, lista_gestos in enumerate(self.ultimas_manos_mp.gestures):
            if not lista_gestos or idx_mano >= len(self.ultimas_manos_mp.hand_landmarks):
                continue
            nombre_gesto = lista_gestos[0].category_name
            landmarks = self.ultimas_manos_mp.hand_landmarks[idx_mano]
            muneca = landmarks[0]
            mx, my = int(muneca.x * fw), int(muneca.y * fh)

            cv.circle(frame_bgr, (mx, my), 6, (255, 200, 0), -1)

            if nombre_gesto not in ("Open_Palm", "Thumb_Up"):
                continue

            for pista in self.pistas:
                x, y, w, h = pista.bbox
                cx, cy = pista.centro
                geometria_ok = (abs(mx - cx) < w * 2.4) and (y - int(h * 0.8) <= my <= y + int(h * 2.5))

                if not geometria_ok:
                    continue

                cv.line(frame_bgr, (mx, my), (cx, cy), (16, 220, 130), 2, cv.LINE_AA)
                etiqueta_g = "ENTRADA (Palma)" if nombre_gesto == "Open_Palm" else "SALIDA (Pulgar)"
                cv.putText(
                    frame_bgr,
                    etiqueta_g,
                    (mx - 40, max(25, my - 12)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (16, 240, 150),
                    2,
                    cv.LINE_AA,
                )

                if not pista.reconocido:
                    continue

                if pista.gesto_actual == nombre_gesto:
                    pista.frames_gesto += 1
                else:
                    pista.gesto_actual = nombre_gesto
                    pista.frames_gesto = 1

                progreso = min(1.0, pista.frames_gesto / float(self.frames_gesto_req))
                ancho_barra = int(w * progreso)
                cv.rectangle(frame_bgr, (x, y - 12), (x + w, y - 4), (40, 40, 40), -1)
                cv.rectangle(frame_bgr, (x, y - 12), (x + ancho_barra, y - 4), (0, 230, 140), -1)

                if pista.frames_gesto >= self.frames_gesto_req:
                    pista.frames_gesto = 0
                    ahora_ts = time.time()
                    ult = self.ultimo_disparo_por_persona.get(pista.nombre, 0.0)
                    if ahora_ts - ult < self.cooldown_seg:
                        continue

                    esta_dentro = pista.nombre in estados_presentes
                    tipo_ev = "entrada" if nombre_gesto == "Open_Palm" else "salida"

                    if tipo_ev == "entrada" and esta_dentro:
                        self.banner_feedback = (f"{pista.nombre} ya figura DENTRO del LIS", (230, 160, 20), ahora_ts + 2.5)
                        self.ultimo_disparo_por_persona[pista.nombre] = ahora_ts
                        continue
                    if tipo_ev == "salida" and not esta_dentro:
                        self.banner_feedback = (f"{pista.nombre} ya figura FUERA del LIS", (230, 160, 20), ahora_ts + 2.5)
                        self.ultimo_disparo_por_persona[pista.nombre] = ahora_ts
                        continue

                    evento = crear_evento(
                        tipo=tipo_ev,
                        origen="pipeline-facial",
                        metodo="cara+gesto",
                        persona_id=pista.nombre,
                        distancia=pista.distancia,
                        meta={
                            "nombre": pista.nombre,
                            "proyecto": self.proyecto,
                            "umbral": self.umbral,
                            "modelo": nombre_modelo_resolucion(self.resolucion_recorte),
                            "resolucion": self.resolucion_recorte,
                        },
                    )
                    self.repositorio.registrar_evento(evento)
                    self.fichador.notificar_evento_acceso(pista.nombre, tipo_ev, pista.distancia)
                    self.ultimo_disparo_por_persona[pista.nombre] = ahora_ts
                    color_b = (16, 185, 129) if tipo_ev == "entrada" else (235, 130, 50)
                    self.banner_feedback = (
                        f"{tipo_ev.upper()} REGISTRADA: {pista.nombre} (d={pista.distancia:.2f})",
                        color_b,
                        ahora_ts + 3.5,
                    )
                    logging.info("Evento %s registrado para %s (d=%.3f, res=%dpx)", tipo_ev, pista.nombre, pista.distancia, self.resolucion_recorte)

    def _dibujar_hud(self, frame_bgr: np.ndarray, fps: float, presentes: dict[str, Any]) -> None:
        fh, fw = frame_bgr.shape[:2]

        overlay = frame_bgr.copy()
        cv.rectangle(overlay, (0, 0), (fw, 44), (18, 24, 38), -1)
        cv.addWeighted(overlay, 0.82, frame_bgr, 0.18, 0, frame_bgr)

        fuente_corta = str(self.fuente_camara)[:12]
        texto_hdr = (
            f"LIS Vision | FPS: {fps:.1f} | Res: {self.resolucion_recorte}px | "
            f"Cam: {fuente_corta} | Presentes: {len(presentes)} | d<{self.umbral:.2f}"
        )
        cv.putText(frame_bgr, texto_hdr, (14, 28), cv.FONT_HERSHEY_SIMPLEX, 0.50, (235, 240, 245), 1, cv.LINE_AA)

        for pista in self.pistas:
            x, y, w, h = pista.bbox
            if pista.frames_estables < self.frames_estable_req:
                color = (40, 200, 250)
                etiqueta_sup = f"Estabilizando ({pista.frames_estables}/{self.frames_estable_req})"
                etiqueta_inf = "Calculando embedding..."
            elif pista.reconocido:
                color = (16, 200, 120)
                estado_str = "DENTRO" if pista.nombre in presentes else "FUERA"
                etiqueta_sup = f"{pista.nombre} [{estado_str}]"
                etiqueta_inf = f"d: {pista.distancia:.2f} / umbral: {self.umbral:.2f} ({self.resolucion_recorte}px)"
            else:
                color = (60, 65, 235)
                etiqueta_sup = "Desconocido"
                etiqueta_inf = f"d: {pista.distancia:.2f} (umbral: {self.umbral:.2f} | {self.resolucion_recorte}px)"

            ancho_badge = max(w, 210)
            cv.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(frame_bgr, (x, max(45, y - 30)), (x + ancho_badge, y), color, -1)
            cv.putText(
                frame_bgr,
                _ascii_seguro(etiqueta_sup),
                (x + 6, max(62, y - 9)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame_bgr,
                _ascii_seguro(etiqueta_inf),
                (x, min(fh - 10, y + h + 20)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv.LINE_AA,
            )

        if self.hud_guia:
            bx, by, bw, bh = self.hud_guia["bbox"]
            cx, cy = bx + bw // 2, by + bh // 2
            col_g = (16, 225, 130) if self.hud_guia.get("coincide") else (225, 180, 20)
            cv.ellipse(frame_bgr, (cx, cy), (int(bw * 0.65), int(bh * 0.85)), 0, 0, 360, col_g, 2, cv.LINE_AA)
            prog = float(self.hud_guia.get("progreso", 0.0))
            if prog > 0:
                cv.ellipse(frame_bgr, (cx, cy), (int(bw * 0.65) + 6, int(bh * 0.85) + 6), 0, -90, -90 + int(360 * prog), (0, 240, 160), 4, cv.LINE_AA)
            inst = _ascii_seguro(str(self.hud_guia.get("instruccion", "")))
            mens = _ascii_seguro(str(self.hud_guia.get("mensaje", "")))
            idx_p = int(self.hud_guia.get("indice", 0)) + 1
            tot_p = int(self.hud_guia.get("total", 5))
            cv.putText(frame_bgr, f"[{idx_p}/{tot_p}] {inst}", (max(16, cx - 120), max(60, by - 40)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame_bgr, mens, (max(16, cx - 120), max(80, by - 16)), cv.FONT_HERSHEY_SIMPLEX, 0.48, col_g, 1, cv.LINE_AA)

        if self.nombre_enrolando:
            orden = self.repositorio.leer_estado_runtime("orden_enrolamiento", {})
            msg = _ascii_seguro(orden.get("mensaje", f"Captura biometrica: {self.nombre_enrolando}..."))
            cv.rectangle(frame_bgr, (0, fh - 36), (fw, fh), (15, 110, 210), -1)
            cv.putText(frame_bgr, msg, (16, fh - 12), cv.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv.LINE_AA)
        elif self.banner_feedback and time.time() < self.banner_feedback[2]:
            msg, col, _ = self.banner_feedback
            cv.rectangle(frame_bgr, (0, fh - 36), (fw, fh), col, -1)
            cv.putText(frame_bgr, _ascii_seguro(msg), (16, fh - 12), cv.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv.LINE_AA)

    def ejecutar(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        opciones_caras = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(self.ruta_modelo_caras)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._cb_caras,
        )
        opciones_manos = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(self.ruta_modelo_manos)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._cb_manos,
        )

        cap = self._abrir_camara()
        fps_actual = 25.0
        t_prev = time.perf_counter()
        frame_idx = 0
        ruta_frame_subido = PROYECTO_DIR / "datos" / "frame_subido.jpg"

        with (
            FaceDetector.create_from_options(opciones_caras) as detector_caras,
            GestureRecognizer.create_from_options(opciones_manos) as reconocedor_manos,
        ):
            while True:
                frame_idx += 1

                if frame_idx % 10 == 0:
                    self._sincronizar_config_reconocimiento()
                    orden_cam = self.repositorio.leer_estado_runtime("orden_camara", {})
                    fuente_orden = orden_cam.get("fuente")
                    if fuente_orden is not None and str(fuente_orden).strip() != str(self.fuente_camara).strip():
                        logging.info("Cambio de cámara solicitado vía Web: %s -> %s", self.fuente_camara, fuente_orden)
                        self.fuente_camara = fuente_orden
                        if cap and cap.isOpened():
                            cap.release()
                        cap = self._abrir_camara()

                es_navegador = str(self.fuente_camara).lower() == "navegador"

                if es_navegador:
                    raw_subido = obtener_frame_subido_jpeg()
                    if raw_subido:
                        try:
                            arr = np.frombuffer(raw_subido, dtype=np.uint8)
                            frame = cv.imdecode(arr, cv.IMREAD_COLOR)
                            ok = frame is not None
                        except Exception:
                            ok, frame = False, None
                    elif ruta_frame_subido.exists() and ruta_frame_subido.stat().st_size > 100:
                        try:
                            frame = cv.imread(str(ruta_frame_subido))
                            ok = frame is not None
                        except Exception:
                            ok, frame = False, None
                    else:
                        ok, frame = False, None
                else:
                    ok, frame = cap.read() if (cap and cap.isOpened()) else (False, None)

                if not ok or frame is None:
                    frame = np.full((480, 640, 3), (18, 20, 26), dtype=np.uint8)
                    cv.rectangle(frame, (48, 68), (592, 432), (38, 43, 56), 1)
                    cv.putText(
                        frame,
                        "SIN SENAL DE VIDEO EN FUENTE SELECCIONADA",
                        (72, 235),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (210, 218, 230),
                        1,
                        cv.LINE_AA,
                    )
                    time.sleep(0.05)
                elif self.espejo and not es_navegador:
                    frame = cv.flip(frame, 1)

                t_now = time.perf_counter()
                dt = max(1e-4, t_now - t_prev)
                t_prev = t_now
                fps_actual = 0.9 * fps_actual + 0.1 * (1.0 / dt)

                if frame_idx % 30 == 0:
                    self._recargar_galerias()

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(time.time() * 1000)

                detector_caras.detect_async(mp_img, ts_ms)
                reconocedor_manos.recognize_async(mp_img, ts_ms)

                cajas: list[tuple[int, int, int, int]] = []
                if self.ultimas_caras_mp and self.ultimas_caras_mp.detections:
                    for det in self.ultimas_caras_mp.detections:
                        bb = det.bounding_box
                        cajas.append((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)))

                self._actualizar_pistas(frame, cajas)
                self._procesar_enrolamiento_activo(frame, frame_idx)

                eventos_recientes = self.repositorio.listar_eventos(desde_seq=0, limite=2000)
                estado_derivado = derivar_estado(eventos_recientes)
                presentes = estado_derivado["presentes"]

                self._evaluar_gestos_y_eventos(frame, presentes)
                self._dibujar_hud(frame, fps_actual, presentes)

                if frame_idx % 2 == 0:
                    try:
                        guardar_frame_jpeg(frame, calidad=82)
                        ok_jpg, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 82])
                        if ok_jpg:
                            guardar_frame_disco_seguro(buf.tobytes(), self.ruta_frame_compartido)
                    except Exception:
                        pass

                if frame_idx % 15 == 0:
                    self.repositorio.guardar_estado_runtime(
                        "camara_estado",
                        {
                            "activa": True,
                            "fps": round(fps_actual, 1),
                            "rostros_en_cuadro": len(self.pistas),
                            "fuente": str(self.fuente_camara),
                            "resolucion": self.resolucion_recorte,
                        },
                    )

                if self.mostrar_ventana:
                    cv.imshow("LIS - Reconocimiento Facial y Gestos", frame)
                    tecla = cv.waitKey(1) & 0xFF
                    if tecla == 27:
                        break
                else:
                    time.sleep(0.01)

        self._cerrar_landmarker()
        cap.release()
        if self.mostrar_ventana:
            cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = PipelineReconocimientoLIS()
    pipeline.ejecutar()