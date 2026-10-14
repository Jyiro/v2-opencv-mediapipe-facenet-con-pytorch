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
import torch
from facenet_pytorch import InceptionResnetV1

PROYECTO_DIR = Path(__file__).resolve().parent.parent
MEDIAPIPE_DIR = Path(__file__).resolve().parent
if str(PROYECTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROYECTO_DIR))

from hardware.serie import ClienteFichadorSerie
from nucleo.base_datos import RepositorioLIS
from nucleo.comparacion import consultar_embeddings
from nucleo.eventos import crear_evento, derivar_estado

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def cargar_configuracion() -> dict[str, Any]:
    ruta_cfg = PROYECTO_DIR / "config.toml"
    if ruta_cfg.exists():
        with open(ruta_cfg, "rb") as f:
            return tomllib.load(f)
    return {}


class ExtractorFaceNetRapido:
    """
    Extrae embeddings de 512 dimensiones con InceptionResnetV1 (vggface2)
    directamente sobre el recorte de BlazeFace, eliminando la doble detección de MTCNN en vivo.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        logging.info("Extractor InceptionResnetV1 (vggface2) inicializado en %s", self.device)

    def generar_embedding(self, recorte_bgr: np.ndarray) -> np.ndarray | None:
        try:
            if recorte_bgr is None or recorte_bgr.size == 0:
                return None
            h, w = recorte_bgr.shape[:2]
            if h < 20 or w < 20:
                return None
            img_160 = cv.resize(recorte_bgr, (160, 160), interpolation=cv.INTER_AREA)
            img_rgb = cv.cvtColor(img_160, cv.COLOR_BGR2RGB).astype(np.float32)
            tensor = torch.from_numpy((img_rgb - 127.5) / 128.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = self.resnet(tensor).detach().cpu().numpy().flatten()
            return emb.astype(np.float32)
        except Exception:
            logging.exception("Error al generar embedding facial")
            return None


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

        self.umbral: float = float(params.get("umbral_reconocimiento", 0.80))
        self.metodo_comp: str = str(params.get("metodo_comparacion", "minimo"))
        self.frames_estable_req: int = int(params.get("frames_cara_estable", 3))
        self.tolerancia_px: int = int(params.get("tolerancia_movimiento_px", 45))
        self.frames_gesto_req: int = int(params.get("frames_confirmacion_gesto", 5))
        self.cooldown_seg: float = float(params.get("cooldown_gesto_segundos", 4.0))
        self.proyecto: str = str(params.get("proyecto", "LIS"))

        self.mostrar_ventana = mostrar_ventana and bool(cam_cfg.get("mostrar_ventana_local", True))
        self.espejo = bool(cam_cfg.get("espejo", True))

        url_stream = str(cam_cfg.get("url_stream", "")).strip()
        self.fuente_camara = fuente_camara if fuente_camara is not None else (url_stream or int(cam_cfg.get("indice", 0)))

        self.repositorio = RepositorioLIS()
        self.fichador = ClienteFichadorSerie(repositorio=self.repositorio, simular=True)
        self.extractor = ExtractorFaceNetRapido()

        self.base_embeddings = self.repositorio.cargar_diccionario_embeddings()
        self.ultimo_disparo_por_persona: dict[str, float] = {}
        self.banner_feedback: tuple[str, tuple[int, int, int], float] | None = None

        self.ruta_modelo_caras = MEDIAPIPE_DIR / modelos_cfg.get("detector_caras", "blaze_face_short_range.tflite")
        self.ruta_modelo_manos = MEDIAPIPE_DIR / modelos_cfg.get("reconocedor_manos", "manos/gesture_recognizer.task")
        self.ruta_frame_compartido = PROYECTO_DIR / "datos" / "ultimo_frame.jpg"
        self.ruta_frame_compartido.parent.mkdir(parents=True, exist_ok=True)

        self.ultimas_caras_mp: Any = None
        self.ultimas_manos_mp: Any = None
        self.pistas: list[PistaRostro] = []

        self.buffer_enrolamiento: list[np.ndarray] = []
        self.nombre_enrolando: str | None = None
        self.ultimo_frame_captura_alta: int = 0

    def _cb_caras(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        self.ultimas_caras_mp = result

    def _cb_manos(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        self.ultimas_manos_mp = result

    def _abrir_camara(self) -> cv.VideoCapture:
        if isinstance(self.fuente_camara, str) and not self.fuente_camara.isdigit():
            if self.fuente_camara.lower() in ("simulada", "simulado", "virtual", "mock"):
                logging.info("Iniciando en modo cámara simulada (sin hardware).")
                return cv.VideoCapture()
            logging.info("Abriendo stream de cámara en %s", self.fuente_camara)
            return cv.VideoCapture(self.fuente_camara)
        try:
            idx = int(self.fuente_camara)
            backend = str(self.config.get("camara", {}).get("backend", "auto")).lower()
            if backend == "dshow" and sys.platform.startswith("win"):
                cap = cv.VideoCapture(idx, cv.CAP_DSHOW)
            elif backend == "v4l2" and sys.platform.startswith("linux"):
                cap = cv.VideoCapture(idx, cv.CAP_V4L2)
            else:
                cap = cv.VideoCapture(idx)
            if not cap.isOpened():
                logging.info("Reintentando apertura de cámara %d con backend por defecto...", idx)
                cap = cv.VideoCapture(idx)
            if cap.isOpened():
                logging.info("Cámara física conectada exitosamente en índice %d.", idx)
            else:
                logging.warning("No se detectó cámara física en índice %d. Activando modo simulado (sin señal).", idx)
            return cap
        except Exception as exc:
            logging.warning("Excepción al abrir cámara %s: %s. Usando modo simulado.", self.fuente_camara, exc)
            return cv.VideoCapture()

    def _actualizar_pistas(self, frame_bgr: np.ndarray, detecciones: list[tuple[int, int, int, int]]) -> list[PistaRostro]:
        nuevas_pistas: list[PistaRostro] = []
        usadas: set[int] = set()

        for (x, y, w, h) in detecciones:
            cx, cy = x + w // 2, y + h // 2
            mejor_idx = -1
            menor_dist = float("inf")

            for idx, pista_ant in enumerate(self.pistas):
                if idx in usadas:
                    continue
                dist_px = float(np.hypot(cx - pista_ant.centro[0], cy - pista_ant.centro[1]))
                if dist_px < self.tolerancia_px and dist_px < menor_dist:
                    menor_dist = dist_px
                    mejor_idx = idx

            if mejor_idx >= 0:
                pista = self.pistas[mejor_idx]
                usadas.add(mejor_idx)
                pista.bbox = (x, y, w, h)
                pista.centro = (cx, cy)
                pista.frames_estables += 1
                pista.frames_desde_inferencia += 1
            else:
                pista = PistaRostro((x, y, w, h), (cx, cy))

            debe_inferir = (
                pista.frames_estables == self.frames_estable_req
                or (pista.frames_estables > self.frames_estable_req and pista.frames_desde_inferencia >= 45)
                or (self.nombre_enrolando is not None and pista.frames_estables >= 2)
            )

            if debe_inferir:
                fh, fw = frame_bgr.shape[:2]
                pad_x, pad_y = int(w * 0.08), int(h * 0.08)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
                recorte = frame_bgr[y1:y2, x1:x2]
                emb = self.extractor.generar_embedding(recorte)
                if emb is not None:
                    pista.ultimo_embedding = emb
                    pista.frames_desde_inferencia = 0
                    res = consultar_embeddings(emb, self.base_embeddings, self.umbral, self.metodo_comp)
                    pista.reconocido = res["reconocido"]
                    pista.nombre = res["nombre"]
                    pista.distancia = res["distancia"]

            nuevas_pistas.append(pista)

        self.pistas = nuevas_pistas
        return self.pistas

    def _procesar_enrolamiento_activo(self, frame_idx: int) -> None:
        orden = self.repositorio.leer_estado_runtime("orden_enrolamiento", {"activo": False})
        if not orden.get("activo"):
            self.nombre_enrolando = None
            self.buffer_enrolamiento.clear()
            return

        nombre_objetivo = str(orden.get("nombre", "")).strip()
        muestras_totales = int(orden.get("muestras_totales", 5))

        if self.nombre_enrolando != nombre_objetivo:
            self.nombre_enrolando = nombre_objetivo
            self.buffer_enrolamiento.clear()
            self.base_embeddings = self.repositorio.cargar_diccionario_embeddings()

        if not self.pistas:
            orden["mensaje"] = f"Buscando rostro de {nombre_objetivo} en cámara..."
            self.repositorio.guardar_estado_runtime("orden_enrolamiento", orden)
            return

        pista_principal = max(self.pistas, key=lambda p: p.bbox[2] * p.bbox[3])
        if pista_principal.ultimo_embedding is None:
            return

        if frame_idx - self.ultimo_frame_captura_alta < 8:
            return

        # Verificación anti-duplicado biométrico en la primera muestra
        if len(self.buffer_enrolamiento) == 0:
            otras_caras = {k: v for k, v in self.base_embeddings.items() if k.lower() != nombre_objetivo.lower()}
            chequeo = consultar_embeddings(pista_principal.ultimo_embedding, otras_caras, self.umbral, self.metodo_comp)
            if chequeo["reconocido"]:
                duplicado_de = chequeo["nombre"]
                msg_err = f"Rostro ya registrado bajo el perfil '{duplicado_de}' (d={chequeo['distancia']:.2f})."
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
                return

        self.buffer_enrolamiento.append(pista_principal.ultimo_embedding.copy())
        self.ultimo_frame_captura_alta = frame_idx
        capturadas = len(self.buffer_enrolamiento)

        if capturadas < muestras_totales:
            orden["muestras_capturadas"] = capturadas
            orden["estado"] = "capturando"
            orden["mensaje"] = f"Capturando {nombre_objetivo}: muestra {capturadas}/{muestras_totales} (gire levemente el rostro)"
            self.repositorio.guardar_estado_runtime("orden_enrolamiento", orden)
        else:
            self.repositorio.guardar_embeddings_persona(
                nombre=nombre_objetivo,
                embeddings=self.buffer_enrolamiento,
                reemplazar=True,
                sincronizar_csv=True,
            )
            self.base_embeddings = self.repositorio.cargar_diccionario_embeddings()
            msg_ok = f"Perfil biométrico de {nombre_objetivo} guardado ({muestras_totales} muestras)."
            self.repositorio.guardar_estado_runtime(
                "orden_enrolamiento",
                {
                    "activo": False,
                    "nombre": nombre_objetivo,
                    "muestras_totales": muestras_totales,
                    "muestras_capturadas": muestras_totales,
                    "estado": "completado",
                    "mensaje": msg_ok,
                },
            )
            self.fichador.cambiar_rgb(16, 185, 129)
            self.fichador.mostrar_mensaje("ALTA COMPLETA", nombre_objetivo[:16])
            self.fichador.sonar_beep("ok")
            self.banner_feedback = (msg_ok, (16, 185, 129), time.time() + 4.0)
            self.nombre_enrolando = None
            self.buffer_enrolamiento.clear()

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

            # Buscar el rostro reconocido geométricamente vinculado a esta mano
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

                # Barra de progreso de confirmación del gesto sobre la cara
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
                            "modelo": "vggface2",
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
                    logging.info("Evento %s registrado para %s (d=%.3f)", tipo_ev, pista.nombre, pista.distancia)

    def _dibujar_hud(self, frame_bgr: np.ndarray, fps: float, presentes: dict[str, Any]) -> None:
        fh, fw = frame_bgr.shape[:2]

        # Header superior oscuro
        overlay = frame_bgr.copy()
        cv.rectangle(overlay, (0, 0), (fw, 44), (18, 24, 38), -1)
        cv.addWeighted(overlay, 0.82, frame_bgr, 0.18, 0, frame_bgr)

        fuente_corta = str(self.fuente_camara)[:14]
        texto_hdr = (
            f"LIS Vision | FPS: {fps:.1f} | Cam: {fuente_corta} | "
            f"Presentes: {len(presentes)} | d<{self.umbral:.2f}"
        )
        cv.putText(frame_bgr, texto_hdr, (14, 28), cv.FONT_HERSHEY_SIMPLEX, 0.54, (235, 240, 245), 1, cv.LINE_AA)

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
                etiqueta_inf = f"d: {pista.distancia:.2f} / umbral: {self.umbral:.2f}"
            else:
                color = (60, 65, 235)
                etiqueta_sup = "Desconocido"
                etiqueta_inf = f"d: {pista.distancia:.2f} (umbral: {self.umbral:.2f})"

            cv.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(frame_bgr, (x, max(45, y - 30)), (x + max(w, 190), y), color, -1)
            cv.putText(
                frame_bgr,
                etiqueta_sup,
                (x + 6, max(62, y - 9)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame_bgr,
                etiqueta_inf,
                (x, min(fh - 10, y + h + 20)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
                cv.LINE_AA,
            )

        # Banner de enrolamiento activo
        if self.nombre_enrolando:
            orden = self.repositorio.leer_estado_runtime("orden_enrolamiento", {})
            msg = orden.get("mensaje", f"Enrolando a {self.nombre_enrolando}...")
            cv.rectangle(frame_bgr, (0, fh - 52), (fw, fh), (15, 110, 210), -1)
            cv.putText(frame_bgr, msg, (16, fh - 18), cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv.LINE_AA)
        elif self.banner_feedback and time.time() < self.banner_feedback[2]:
            msg, col, _ = self.banner_feedback
            cv.rectangle(frame_bgr, (0, fh - 48), (fw, fh), col, -1)
            cv.putText(frame_bgr, msg, (16, fh - 16), cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv.LINE_AA)

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

                # Comprobar si se solicitó un cambio de cámara desde el Tablero Web
                if frame_idx % 12 == 0:
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
                    if ruta_frame_subido.exists() and ruta_frame_subido.stat().st_size > 100:
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
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    if es_navegador:
                        cv.putText(
                            frame,
                            "Esperando video desde el navegador...",
                            (40, 230),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 220, 255),
                            2,
                        )
                        cv.putText(
                            frame,
                            "Active la camara en el Tablero Web",
                            (80, 270),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 180),
                            1,
                        )
                    else:
                        cv.putText(
                            frame,
                            "Camara sin senal - Modo Simulado Activo",
                            (60, 230),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.68,
                            (180, 180, 180),
                            2,
                        )
                        cv.putText(
                            frame,
                            f"Fuente: {self.fuente_camara} (Cambie camara en la Web)",
                            (60, 270),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (100, 180, 255),
                            1,
                        )
                    time.sleep(0.05)
                elif self.espejo and not es_navegador:
                    frame = cv.flip(frame, 1)

                t_now = time.perf_counter()
                dt = max(1e-4, t_now - t_prev)
                t_prev = t_now
                fps_actual = 0.9 * fps_actual + 0.1 * (1.0 / dt)

                if frame_idx % 20 == 0:
                    self.base_embeddings = self.repositorio.cargar_diccionario_embeddings()

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
                self._procesar_enrolamiento_activo(frame_idx)

                eventos_recientes = self.repositorio.listar_eventos(desde_seq=0, limite=2000)
                estado_derivado = derivar_estado(eventos_recientes)
                presentes = estado_derivado["presentes"]

                self._evaluar_gestos_y_eventos(frame, presentes)
                self._dibujar_hud(frame, fps_actual, presentes)

                if frame_idx % 2 == 0:
                    ok_jpg, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 82])
                    if ok_jpg:
                        tmp_path = self.ruta_frame_compartido.with_suffix(".tmp")
                        tmp_path.write_bytes(buf.tobytes())
                        tmp_path.replace(self.ruta_frame_compartido)

                if frame_idx % 15 == 0:
                    self.repositorio.guardar_estado_runtime(
                        "camara_estado",
                        {
                            "activa": True,
                            "fps": round(fps_actual, 1),
                            "rostros_en_cuadro": len(self.pistas),
                            "fuente": str(self.fuente_camara),
                        },
                    )

                if self.mostrar_ventana:
                    cv.imshow("LIS - Reconocimiento Facial y Gestos", frame)
                    tecla = cv.waitKey(1) & 0xFF
                    if tecla == 27:
                        break
                else:
                    time.sleep(0.01)

        cap.release()
        if self.mostrar_ventana:
            cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = PipelineReconocimientoLIS()
    pipeline.ejecutar()