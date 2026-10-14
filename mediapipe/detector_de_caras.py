"""Detector de caras y gestos con reconocimiento facial y control de asistencia LIS.

Versión optimizada y modularizada:
- Integra FuenteDeVideo universal (Webcam, DroidCam por Wi-Fi/IP, y cámara simulada).
- Transmisión en tiempo real de cuadros con HUD al tablero web vía streaming MJPEG.
- Integra SQLite nativo (datos/lis_maestra.db) con modo WAL y retrocompatibilidad CSV.
- Extractor FaceNet rápido sin doble pasada por MTCNN.
- Estabilizador de cara para triplicar los FPS del sistema.
- Event Sourcing para entradas, salidas y reingresos múltiples en el mismo día.
- Alta segura con PIN de administrador, consentimiento informado y ráfaga de 5 muestras.
- Comunicación con el Fichador Arduino UNO (hardware/serie.py) y modo simulado interactivo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import Any

import cv2 as cv
import mediapipe as mp
import numpy as np

# Configurar logging en español
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detector_de_caras")

# Soporte para carga de TOML
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

CARPETA_MEDIAPIPE = Path(__file__).resolve().parent
CARPETA_RAIZ = CARPETA_MEDIAPIPE.parent
if str(CARPETA_RAIZ) not in sys.path:
    sys.path.insert(0, str(CARPETA_RAIZ))

from hardware.camara import FuenteDeVideo, guardar_frame_jpeg
from hardware.serie import ClienteFichadorSerie
from nucleo.base_datos import RepositorioLIS
from nucleo.comparacion import consultar_embeddings
from nucleo.destinos import DestinoCompuesto, DestinoPorteroHTTP, DestinoSQLite
from nucleo.estabilizador import EstabilizadorDeCaras
from nucleo.reconocimiento import obtener_extractor, recortar_con_margen

RUTA_CONFIG = CARPETA_RAIZ / "config.toml"
if not RUTA_CONFIG.exists():
    RUTA_CONFIG = CARPETA_MEDIAPIPE / "config.toml"


def cargar_configuracion(ruta: Path) -> dict[str, Any]:
    if ruta.exists():
        with open(ruta, "rb") as f:
            return tomllib.load(f)
    return {
        "camara": {"indice": 0, "url_stream": "", "backend": "auto", "espejo": True, "mostrar_ventana_local": True},
        "modelos": {"detector_caras": "blaze_face_short_range.tflite", "reconocedor_manos": "manos/gesture_recognizer.task", "motor_embeddings": "facenet_rapido"},
        "archivos": {"base_datos_sqlite": "datos/lis_maestra.db", "caras_csv": "caras/caras.csv", "asistencia_csv": "asistencia/asistencia.csv", "ingresos_csv": "manos/registro_de_ingresos.csv"},
        "parametros": {"umbral_reconocimiento": 0.8, "metodo_comparacion": "minimo", "intervalo_frames_asistencia": 100, "frames_cara_estable": 4, "tolerancia_movimiento_px": 35, "muestras_alta": 5, "cooldown_gesto_segundos": 3.0, "pin_admin": "1234", "proyecto": "LIS"},
    }


config = cargar_configuracion(RUTA_CONFIG)


def resolver_ruta(ruta_relativa: str, base: Path = CARPETA_MEDIAPIPE) -> Path:
    p = Path(ruta_relativa)
    if p.is_absolute():
        return p
    candidato_mediapipe = (base / p).resolve()
    if candidato_mediapipe.exists():
        return candidato_mediapipe
    return (CARPETA_RAIZ / p).resolve()


# Rutas de modelos y almacenamiento
ruta_modelo_caras = resolver_ruta(config["modelos"].get("detector_caras", "blaze_face_short_range.tflite"))
ruta_modelo_manos = resolver_ruta(config["modelos"].get("reconocedor_manos", "manos/gesture_recognizer.task"))
ruta_db_sqlite = resolver_ruta(config["archivos"].get("base_datos_sqlite", "datos/lis_maestra.db"), CARPETA_RAIZ)
ruta_caras_csv = resolver_ruta(config["archivos"].get("caras_csv", "caras/caras.csv"))
ruta_asistencia_csv = resolver_ruta(config["archivos"].get("asistencia_csv", "asistencia/asistencia.csv"))
ruta_ingresos_csv = resolver_ruta(config["archivos"].get("ingresos_csv", "manos/registro_de_ingresos.csv"))

# Inicializar motor relacional
repositorio = RepositorioLIS(
    ruta_db=ruta_db_sqlite,
    ruta_caras_csv=ruta_caras_csv,
    ruta_asistencia_csv=ruta_asistencia_csv,
    ruta_ingresos_csv=ruta_ingresos_csv,
)

# Inicializar extractores de visión
motor_embeddings_cfg = config["modelos"].get("motor_embeddings", "facenet_rapido")
ruta_onnx_cfg = resolver_ruta(config["modelos"].get("ruta_modelo_onnx", "modelos_onnx/w600k_mbf.onnx"), CARPETA_RAIZ)
extractor = obtener_extractor(tipo_motor=motor_embeddings_cfg, ruta_modelo_onnx=ruta_onnx_cfg)

estabilizador = EstabilizadorDeCaras(
    frames_cara_estable=int(config["parametros"].get("frames_cara_estable", 4)),
    tolerancia_movimiento_px=float(config["parametros"].get("tolerancia_movimiento_px", 35)),
)

# Destinos combinados (SQLite + Portero HTTP con cola offline)
destino_portero = DestinoPorteroHTTP()
destino_sqlite = DestinoSQLite(repositorio)
destinos = DestinoCompuesto([destino_sqlite, destino_portero])

# Variables compartidas
resultados_caras = None
manos_detectadas = None
ultimo_cuadro_bgr = None
solicitud_alta_click: dict[str, Any] | None = None
solicitud_alta_remota: dict[str, Any] | None = None
fuente_video_global: FuenteDeVideo | None = None
cliente_fichador_global: ClienteFichadorSerie | None = None


def callback_buscar_caras(result: mp.tasks.vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global resultados_caras
    resultados_caras = result


def callback_registrar_manos(result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global manos_detectadas
    manos_detectadas = result


def manejar_clic_raton(event, x, y, flags, param):
    global solicitud_alta_click
    if event in (cv.EVENT_LBUTTONDBLCLK, cv.EVENT_RBUTTONDOWN):
        logger.info("Clic detectado en (%d, %d) para iniciar alta.", x, y)
        solicitud_alta_click = {"x": x, "y": y}


def solicitar_alta_remota(nombre: str) -> bool:
    global solicitud_alta_remota
    if not nombre or not nombre.strip():
        return False
    solicitud_alta_remota = {"nombre": nombre.strip()}
    logger.info("Solicitud de alta remota encolada para: %s", nombre.strip())
    return True


def iniciar_alta_segura(punto_clic: dict[str, int], detecciones_recientes: list[Any], pin_admin_esperado: str) -> dict[str, Any] | None:
    if not detecciones_recientes:
        logger.warning("No se detectó ningún rostro para iniciar el alta.")
        return None

    x, y = punto_clic.get("x", -1), punto_clic.get("y", -1)
    bbox_seleccionada = None

    if x >= 0 and y >= 0:
        # 1. Comprobar si el clic cayó dentro de algún recuadro
        for det in detecciones_recientes:
            bb = det.bounding_box
            x1, y1 = bb.origin_x, bb.origin_y
            x2, y2 = x1 + bb.width, y1 + bb.height
            if x1 <= x <= x2 and y1 <= y <= y2:
                bbox_seleccionada = (x1, y1, bb.width, bb.height)
                break

        # 2. Si hay 1 sola cara en pantalla, tomarla directamente (tolerante a clics en bordes)
        if bbox_seleccionada is None and len(detecciones_recientes) == 1:
            bb = detecciones_recientes[0].bounding_box
            bbox_seleccionada = (bb.origin_x, bb.origin_y, bb.width, bb.height)

        # 3. Si hay varias caras y no cayó exacto, tomar la más cercana al punto de clic
        if bbox_seleccionada is None and len(detecciones_recientes) > 1:
            mejor_dist_sq = float("inf")
            for det in detecciones_recientes:
                bb = det.bounding_box
                cx = bb.origin_x + bb.width / 2.0
                cy = bb.origin_y + bb.height / 2.0
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                if dist_sq < mejor_dist_sq:
                    mejor_dist_sq = dist_sq
                    bbox_seleccionada = (bb.origin_x, bb.origin_y, bb.width, bb.height)
    else:
        # Activación por teclado: tomar la primera cara detectada
        bb = detecciones_recientes[0].bounding_box
        bbox_seleccionada = (bb.origin_x, bb.origin_y, bb.width, bb.height)

    if bbox_seleccionada is None:
        return None

    logger.info("Abriendo diálogo de alta biométrica (PIN requerido)...")
    raiz_tk = tk.Tk()
    raiz_tk.withdraw()
    raiz_tk.attributes("-topmost", True)
    raiz_tk.lift()
    raiz_tk.focus_force()
    try:
        pin_ingresado = simpledialog.askstring(
            "PIN Requerido",
            "Introduce el PIN de Administrador (por defecto: 1234):",
            show="*",
            parent=raiz_tk,
        )
        if not pin_ingresado or pin_ingresado.strip() != pin_admin_esperado:
            if pin_ingresado is not None:
                messagebox.showerror("Denegado", "PIN de administrador incorrecto.", parent=raiz_tk)
            return None

        nombre = simpledialog.askstring(
            "Registro de Perfil",
            "Introduce tu nombre completo:",
            parent=raiz_tk,
        )
        if not nombre or not nombre.strip():
            return None
        nombre = nombre.strip()

        mensaje_consentimiento = (
            f"CONSENTIMIENTO INFORMADO (LIS - UTN FRC)\n\n"
            f"Persona: {nombre}\n\n"
            f"• Se capturarán 5 muestras de vectores biométricos (embeddings FaceNet).\n"
            f"• Los datos residen exclusivamente en la base de datos local SQLite del laboratorio.\n"
            f"• Por privacidad y diseño (Sección 13), NUNCA se guardan fotos ni recortes en disco.\n\n"
            f"¿Aceptas el consentimiento informado para continuar?"
        )
        if not messagebox.askyesno("Consentimiento Informado", mensaje_consentimiento, parent=raiz_tk):
            return None

        logger.info("Alta confirmada para '%s'. Iniciando ráfaga de 5 capturas...", nombre)
        return {"activo": True, "nombre": nombre, "bbox": bbox_seleccionada, "muestras": [], "contador_frames": 0}
    finally:
        raiz_tk.destroy()


def ejecutar_pipeline(
    fuente_camara: int | str | None = None,
    mostrar_ventana: bool | None = None,
    puerto_serie: str | None = None,
) -> None:
    global resultados_caras, manos_detectadas, ultimo_cuadro_bgr, solicitud_alta_click, solicitud_alta_remota, fuente_video_global, cliente_fichador_global

    # Determinar fuente de cámara
    if fuente_camara is not None:
        origen_cam = fuente_camara
    else:
        url_cfg = config["camara"].get("url_stream", "").strip()
        origen_cam = url_cfg if url_cfg else config["camara"].get("indice", 0)

    ventana_activa = mostrar_ventana if mostrar_ventana is not None else bool(config["camara"].get("mostrar_ventana_local", True))
    espejo_cfg = bool(config["camara"].get("espejo", True))
    backend_cfg = str(config["camara"].get("backend", "auto"))

    fuente_video_global = FuenteDeVideo(origen=origen_cam, backend=backend_cfg, espejo=espejo_cfg)

    # Inicializar cliente serie del Fichador
    cliente_fichador_global = ClienteFichadorSerie(puerto=puerto_serie, simulado=(puerto_serie is None))
    cliente_fichador_global.enviar_mensaje_lcd("LIS Control", "Sistema Listo")
    cliente_fichador_global.set_color_lcd(0, 100, 255)
    cliente_fichador_global.set_led("azul", 1)

    umbral = float(config["parametros"].get("umbral_reconocimiento", 0.8))
    metodo_comp = config["parametros"].get("metodo_comparacion", "minimo")
    proyecto = config["parametros"].get("proyecto", "LIS")
    pin_admin = str(config["parametros"].get("pin_admin", "1234"))
    muestras_alta_objetivo = int(config["parametros"].get("muestras_alta", 5))
    cooldown_gesto = float(config["parametros"].get("cooldown_gesto_segundos", 3.0))

    caras_conocidas = repositorio.obtener_embeddings_activos(modelo=extractor.nombre_modelo)

    opciones_caras = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ruta_modelo_caras)),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=callback_buscar_caras,
    )

    opciones_manos = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ruta_modelo_manos)),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=callback_registrar_manos,
    )

    if ventana_activa:
        cv.namedWindow("LIS Control")
        cv.setMouseCallback("LIS Control", manejar_clic_raton)

    ultimo_gesto_tiempo: dict[str, float] = {}
    mensaje_notificacion: str | None = None
    tiempo_fin_notificacion: float = 0.0
    estado_alta: dict[str, Any] | None = None

    with mp.tasks.vision.FaceDetector.create_from_options(opciones_caras) as detector_caras, \
         mp.tasks.vision.GestureRecognizer.create_from_options(opciones_manos) as reconocedor_manos:

        try:
            while fuente_video_global.isOpened():
                ret, cuadro = fuente_video_global.read()
                if not ret or cuadro is None:
                    time.sleep(0.01)
                    continue

                ultimo_cuadro_bgr = cuadro.copy()
                alto, ancho, _ = cuadro.shape

                cuadro_rgb = cv.cvtColor(cuadro, cv.COLOR_BGR2RGB)
                mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=cuadro_rgb)

                frame_timestamp_ms = int(time.perf_counter() * 1000)
                detector_caras.detect_async(mp_imagen, frame_timestamp_ms)
                reconocedor_manos.recognize_async(mp_imagen, frame_timestamp_ms)

                if solicitud_alta_click is not None:
                    detecciones = resultados_caras.detections if resultados_caras else []
                    estado_alta = iniciar_alta_segura(solicitud_alta_click, detecciones, pin_admin)
                    solicitud_alta_click = None

                bboxes_actuales: list[tuple[int, int, int, int]] = []
                if resultados_caras is not None and resultados_caras.detections:
                    for det in resultados_caras.detections:
                        bb = det.bounding_box
                        bboxes_actuales.append((bb.origin_x, bb.origin_y, bb.width, bb.height))

                if solicitud_alta_remota is not None:
                    nom_remoto = solicitud_alta_remota["nombre"]
                    solicitud_alta_remota = None
                    if bboxes_actuales:
                        estado_alta = {
                            "activo": True,
                            "nombre": nom_remoto,
                            "bbox": bboxes_actuales[0],
                            "muestras": [],
                            "contador_frames": 0,
                        }
                        logger.info("Alta remota iniciada para: %s. Mirá a la cámara...", nom_remoto)
                        mensaje_notificacion = f"ALTA EN PROCESO: Mirá a la cámara, {nom_remoto}..."
                        tiempo_fin_notificacion = time.time() + 4.0
                        if cliente_fichador_global:
                            cliente_fichador_global.enviar_mensaje_lcd("Alta Iniciada", nom_remoto[:16])
                            cliente_fichador_global.set_color_lcd(255, 165, 0)
                    else:
                        logger.warning("No hay rostros frente a la cámara para el alta de %s", nom_remoto)
                        mensaje_notificacion = "ALTA RECHAZADA: No se detecta rostro en cámara"
                        tiempo_fin_notificacion = time.time() + 3.0

                pistas_visibles = estabilizador.actualizar_pistas(bboxes_actuales)

                for p_info in pistas_visibles:
                    if p_info["requiere_extraccion"]:
                        x_o, y_o, w_c, h_c = p_info["bbox"]
                        recorte = recortar_con_margen(cuadro, x_o, y_o, w_c, h_c, margen_pct=0.10)
                        if recorte is not None:
                            emb = extractor.generar_embedding(recorte)
                            if emb is not None:
                                res_comp = consultar_embeddings(emb, caras_conocidas, umbral, metodo_comp)
                                estabilizador.registrar_evaluacion(
                                    id_pista=p_info["id_pista"],
                                    nombre=res_comp["nombre"],
                                    distancia=res_comp["distancia"],
                                    umbral=umbral,
                                    reconocido=res_comp["reconocido"],
                                )
                                p_info["evaluado"] = True
                                p_info["reconocido"] = res_comp["reconocido"]
                                p_info["nombre"] = res_comp["nombre"]
                                p_info["distancia"] = res_comp["distancia"]

                # Ráfaga de alta
                if estado_alta is not None and estado_alta.get("activo"):
                    nombre_alta = estado_alta["nombre"]
                    muestras_reunidas = estado_alta["muestras"]
                    estado_alta["contador_frames"] += 1

                    if estado_alta["contador_frames"] % 5 == 0 and bboxes_actuales:
                        x_o, y_o, w_c, h_c = bboxes_actuales[0]
                        recorte_alta = recortar_con_margen(cuadro, x_o, y_o, w_c, h_c, margen_pct=0.10)
                        if recorte_alta is not None:
                            emb_a = extractor.generar_embedding(recorte_alta)
                            if emb_a is not None:
                                muestras_reunidas.append(emb_a)
                                logger.info("Muestra biométrica %d/%d capturada para %s", len(muestras_reunidas), muestras_alta_objetivo, nombre_alta)

                    if len(muestras_reunidas) >= muestras_alta_objetivo:
                        repositorio.registrar_persona_con_embeddings(
                            nombre=nombre_alta,
                            embeddings=muestras_reunidas,
                            rol="integrante",
                            consentimiento=True,
                            modelo=extractor.nombre_modelo,
                        )
                        caras_conocidas = repositorio.obtener_embeddings_activos(modelo=extractor.nombre_modelo)
                        estabilizador.reiniciar()
                        mensaje_notificacion = f"ALTA EXITOSA: {nombre_alta} ({muestras_alta_objetivo} muestras)"
                        tiempo_fin_notificacion = time.time() + 4.0
                        if cliente_fichador_global:
                            cliente_fichador_global.enviar_mensaje_lcd("Alta Exitosa", nombre_alta[:16])
                            cliente_fichador_global.set_color_lcd(0, 255, 100)
                            cliente_fichador_global.beep("ok")
                        estado_alta = None

                # Reconocimiento de gestos y eventos
                gesto_detectado_texto = None
                geometria_valida = False

                if manos_detectadas and manos_detectadas.gestures and manos_detectadas.hand_landmarks:
                    categoria_gesto = manos_detectadas.gestures[0][0].category_name
                    muñeca = manos_detectadas.hand_landmarks[0][0]
                    x_muñeca = int(muñeca.x * ancho)
                    y_muñeca = int(muñeca.y * alto)

                    for p_info in pistas_visibles:
                        if not p_info["reconocido"]:
                            continue
                        usuario = p_info["nombre"]
                        distancia_u = p_info["distancia"]
                        x_o, y_o, w_c, h_c = p_info["bbox"]
                        centro_x = x_o + (w_c / 2.0)
                        geometria_ok = (y_o <= y_muñeca <= y_o + h_c) and (abs(x_muñeca - centro_x) < (w_c * 2.5))

                        if geometria_ok:
                            geometria_valida = True
                            gesto_detectado_texto = f"{categoria_gesto} (Geometría OK)"
                            ahora = time.time()
                            if ahora - ultimo_gesto_tiempo.get(usuario, 0.0) >= cooldown_gesto:
                                estado_usuario = repositorio.obtener_estado_usuario(usuario)

                                if categoria_gesto == "Open_Palm" and estado_usuario == "FUERA":
                                    repositorio.registrar_evento("entrada", nombre=usuario, metodo="cara+gesto", distancia=distancia_u, proyecto=proyecto)
                                    ultimo_gesto_tiempo[usuario] = ahora
                                    mensaje_notificacion = f"ENTRADA REGISTRADA: {usuario} (d={distancia_u:.2f})"
                                    tiempo_fin_notificacion = ahora + 3.0
                                    cliente_fichador_global.enviar_mensaje_lcd(usuario, "Ingreso OK")
                                    cliente_fichador_global.set_color_lcd(0, 255, 0)
                                    cliente_fichador_global.set_led("verde", 1)
                                    cliente_fichador_global.beep("ok")

                                elif categoria_gesto == "Thumb_Up" and estado_usuario == "DENTRO":
                                    repositorio.registrar_evento("salida", nombre=usuario, metodo="cara+gesto", distancia=distancia_u, proyecto=proyecto)
                                    ultimo_gesto_tiempo[usuario] = ahora
                                    mensaje_notificacion = f"SALIDA REGISTRADA: {usuario}"
                                    tiempo_fin_notificacion = ahora + 3.0
                                    cliente_fichador_global.enviar_mensaje_lcd(usuario, "Egreso OK")
                                    cliente_fichador_global.set_color_lcd(0, 150, 255)
                                    cliente_fichador_global.set_led("verde", 1)
                                    cliente_fichador_global.beep("ok")
                        else:
                            gesto_detectado_texto = f"{categoria_gesto} (Lejos del rostro)"

                # HUD
                for p_info in pistas_visibles:
                    x_o, y_o, w_c, h_c = p_info["bbox"]
                    x_d, y_d = x_o + w_c, y_o + h_c
                    if not p_info["evaluado"]:
                        cv.rectangle(cuadro, (x_o, y_o), (x_d, y_d), (0, 255, 255), 2)
                        cv.putText(cuadro, f"Estabilizando {p_info['contador_estable']}/{p_info['meta_estable']}...", (x_o, max(20, y_o - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    elif p_info["reconocido"]:
                        cv.rectangle(cuadro, (x_o, y_o), (x_d, y_d), (0, 255, 0), 2)
                        estado_u = repositorio.obtener_estado_usuario(p_info["nombre"])
                        cv.putText(cuadro, f"{p_info['nombre']} [{estado_u}]", (x_o, max(20, y_o - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        cv.putText(cuadro, f"d: {p_info['distancia']:.2f} / umbral: {umbral:.2f}", (x_o, y_d + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv.rectangle(cuadro, (x_o, y_o), (x_d, y_d), (0, 0, 255), 2)
                        cv.putText(cuadro, f"Desconocido (d: {p_info['distancia']:.2f} / {umbral:.2f})", (x_o, max(20, y_o - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                fps_actual = estabilizador.medidor_fps.fps_actual
                total_presentes = repositorio.obtener_conteo_presentes()
                cv.rectangle(cuadro, (0, 0), (ancho, 36), (30, 30, 30), -1)
                encabezado = f"LIS Control | Motor: {extractor.nombre_modelo} | FPS: {fps_actual:.1f} | Presentes: {total_presentes}"
                cv.putText(cuadro, encabezado, (15, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)

                if estado_alta is not None and estado_alta.get("activo"):
                    cant = len(estado_alta["muestras"])
                    cv.rectangle(cuadro, (0, 40), (ancho, 75), (0, 140, 255), -1)
                    cv.putText(cuadro, f"ALTA EN CURSO: Muestra {cant + 1}/{muestras_alta_objetivo} de {estado_alta['nombre']}...", (15, 65), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if gesto_detectado_texto:
                    color_g = (0, 255, 0) if geometria_valida else (0, 165, 255)
                    cv.putText(cuadro, f"Mano: {gesto_detectado_texto}", (15, alto - 50), cv.FONT_HERSHEY_SIMPLEX, 0.55, color_g, 2)

                if mensaje_notificacion and time.time() < tiempo_fin_notificacion:
                    cv.rectangle(cuadro, (0, alto - 40), (ancho, alto), (0, 120, 0), -1)
                    cv.putText(cuadro, mensaje_notificacion, (20, alto - 12), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                else:
                    cv.putText(
                        cuadro,
                        "[Doble Clic / Clic Derecho / Tecla 'A']: Registrar Perfil",
                        (15, alto - 15),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1,
                        cv.LINE_AA,
                    )

                # Exportar cuadro para clientes web (streaming MJPEG a celulares y tablets)
                guardar_frame_jpeg(cuadro)

                # Visualización local opcional en ventana OpenCV
                if ventana_activa:
                    cv.imshow("LIS Control", cuadro)
                    tecla = cv.waitKey(1) & 0xFF
                    if tecla == 27:
                        break
                    elif tecla in (ord('a'), ord('A'), ord('r'), ord('R')):
                        logger.info("Tecla [A/R]: Iniciar alta con el rostro enfocado.")
                        solicitud_alta_click = {"x": -1, "y": -1}
                    elif tecla in (ord('i'), ord('I')):
                        # Disparo simulado por teclado
                        logger.info("Tecla [I]: Simular ingreso.")
                        if pistas_visibles:
                            repositorio.registrar_evento("entrada", nombre=pistas_visibles[0]["nombre"], metodo="manual", distancia=0.45)
                    elif tecla in (ord('e'), ord('E')):
                        logger.info("Tecla [E]: Simular egreso.")
                        if pistas_visibles:
                            repositorio.registrar_evento("salida", nombre=pistas_visibles[0]["nombre"], metodo="manual", distancia=0.45)
                    elif tecla in (ord('t'), ord('T')):
                        cliente_fichador_global.simular_touch()

        finally:
            if cliente_fichador_global:
                cliente_fichador_global.cerrar()
            if fuente_video_global:
                fuente_video_global.liberar()
            if ventana_activa:
                cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Reconocimiento y Asistencia LIS")
    parser.add_argument("--camara", type=str, default=None, help="Índice (0, 1), URL DroidCam (http://...) o 'simulada'")
    parser.add_argument("--sin-ventana", action="store_true", help="Desactiva cv.imshow (para ver exclusivamente desde la tablet web)")
    parser.add_argument("--serie", type=str, default=None, help="Puerto COM / tty del Arduino Fichador")
    args = parser.parse_args()

    ejecutar_pipeline(
        fuente_camara=args.camara,
        mostrar_ventana=not args.sin_ventana,
        puerto_serie=args.serie,
    )


if __name__ == "__main__":
    main()