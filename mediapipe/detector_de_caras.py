import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import time
import tkinter as tk
from tkinter import simpledialog, messagebox
import csv
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from datetime import datetime
import json

#Variables
#Reconocimiento de caras
ruta_del_modelo = "C:\\Users\\nicol\\OneDrive\\Desktop\\mediapipe\\blaze_face_short_range.tflite"
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions  = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
resultados = None
ultima_cara = None
ruta_a_las_caras = "C:\\Users\\nicol\\OneDrive\\Desktop\\mediapipe\\caras\\caras.csv"
mtcnn = None
resnet = None
#video (opencv)
contador_de_frames = 0
ruta_del_archivo_de_asistencias = f"C:\\Users\\nicol\\OneDrive\\Desktop\\mediapipe\\asistencia\\asistencia.csv"
realizar_la_deteccion = False
VisionRunningMode = mp.tasks.vision.RunningMode
#Reconocimiento de manos
ruta_del_modelo_de_manos = f"C:\\Users\\nicol\\OneDrive\\Desktop\\mediapipe\\manos\\gesture_recognizer.task"
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
manos_detectadas = None
ruta_del_archivo_de_registros_de_ingreso = f"C:\\Users\\nicol\\OneDrive\\Desktop\\mediapipe\\manos\\registro_de_ingresos.csv"
asistencias = {}
ingresos_y_egresos = {}
personas_que_saludaron = set()

#Metodos
def buscar_caras(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global resultados
    resultados = result

def guardar(event, x, y, flags, param):
    """
    Metodo que guarda la cara seleccionada

    Args:
        event: Evento
        x: cordenada x
        y: cordenada y
        flags: banderas
        params: parametros
    
    Return: None
    """
    global ultima_cara, nombre, caras_conocidas
    try:
        if event == cv.EVENT_LBUTTONDBLCLK:
            if resultados is None:
                return
            for cara_detectada in resultados.detections:
                bbox_a_guardar = cara_detectada.bounding_box
                x_origen_a_guardar = bbox_a_guardar.origin_x
                y_origen_a_guardar = bbox_a_guardar.origin_y
                x_destino_a_guardar = bbox_a_guardar.origin_x + bbox_a_guardar.width
                y_destino_a_guardar = bbox_a_guardar.origin_y + bbox_a_guardar.height
                if x_origen_a_guardar <= x <= x_destino_a_guardar and y_origen_a_guardar <= y <= y_destino_a_guardar:
                    cara_a_guardar = ultima_cara[y_origen_a_guardar:y_destino_a_guardar, x_origen_a_guardar:x_destino_a_guardar]
                    aplicacion_de_windows = tk.Tk()
                    aplicacion_de_windows.withdraw()
                    nombre = simpledialog.askstring("Ingresar nombre", "Introduce el nombre de la persona:", parent=aplicacion_de_windows)
                    aplicacion_de_windows.destroy()
                    cara_a_guardar = cv.resize(cara_a_guardar, (160,160))
                    embedding = generar_embedding(cara_a_guardar)
                    if embedding is not None:
                        usuario_registrado = compara_embeddings(embedding, caras_conocidas, 0.8)
                        if not usuario_registrado:
                            with open(ruta_a_las_caras, 'a', newline='') as archivo:
                                nueva_fila = [",".join(map(str, embedding)), nombre]
                                escritor = csv.writer(archivo, quoting=csv.QUOTE_ALL)
                                escritor.writerow(nueva_fila)
                                caras_conocidas = consultar_embeddings()
                        else:
                            aplicacion_de_windows = tk.Tk()
                            aplicacion_de_windows.withdraw()
                            messagebox.showinfo("Accion incompleta", "No se pudo registrar al usuario por que ya existe")
                            aplicacion_de_windows.destroy()
                    else:
                        aplicacion_de_windows = tk.Tk()
                        aplicacion_de_windows.withdraw()
                        messagebox.showinfo("Accion incompleta", "No se pudo registrar al usuario")
                        aplicacion_de_windows.destroy()

    except Exception as e:
        print(f"{e}")

def generar_embedding(cara_recortada):
        global mtcnn, resnet
        try:
            cara_a_guardar_RGB = cv.cvtColor(cara_recortada, cv.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(cara_a_guardar_RGB)
            cara_tensor = mtcnn(imagen_pil)
            embedding_de_la_imagen = resnet(cara_tensor.unsqueeze(0))
            return embedding_de_la_imagen.detach().cpu().numpy().flatten()
        except Exception as e:
            return None

def consultar_embeddings():
    caras_conocidas = {}
    with open(ruta_a_las_caras, 'r') as archivo:
        next(archivo)
        for fila in archivo:
            fila = fila.strip()
            if not fila:
                continue
            valores, nombre = fila.rsplit(',', 1)
            valores = valores.strip('"')
            embedding = np.array([float(x) for x in valores.split(',')], dtype=np.float32)
            caras_conocidas[nombre] = embedding
    return caras_conocidas

def comparar_caras(cara_a_comparar, caras_conocidas, umbral=0.5):
    embedding = generar_embedding(cara_a_comparar)
    minima_distancia = float('inf')
    nombre = None
    if embedding is not None:
        for nombre_conocido, embedding_conocido in caras_conocidas.items():
            distancia = np.linalg.norm(embedding - embedding_conocido)
            if distancia < minima_distancia:
                minima_distancia = distancia
                nombre = nombre_conocido.strip('"')
        if minima_distancia < umbral:
            return nombre
        else:
            return None
    return None

def registrar_ingreso(result: GestureRecognizerResult, out_image: mp.Image, timestamp_ms: int): # type: ignore
    global manos_detectadas
    manos_detectadas = result

def guardar_asistencias(asistencias):
    with open(ruta_del_archivo_de_asistencias, 'a', newline='') as archivo:
        escritor = csv.writer(archivo)
        for i in asistencias:
            escritor.writerow(asistencias[i])
            print(f"Se registró {asistencias[i][1]}")

def guardar_ingresos(ingresos):
     with open(ruta_del_archivo_de_registros_de_ingreso, 'a', newline='') as archivo_registro:
        escritor = csv.writer(archivo_registro)
        for i in ingresos:
            escritor.writerow(ingresos[i])
            print(f"Se registró {ingresos[i][1]}")

def compara_embeddings(embedding, caras_conocidas, umbral=0.5):
    minima_distancia = float('inf')
    nombre = None
    for nombre_conocido, embedding_conocido in caras_conocidas.items():
        distancia = np.linalg.norm(embedding - embedding_conocido)
        if distancia < minima_distancia:
            minima_distancia = distancia
    if minima_distancia < umbral:
        return True
    else:
        return False

#Cuerpo principal del programa
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=ruta_del_modelo),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback=buscar_caras)

opc_de_rec_de_manos = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=ruta_del_modelo_de_manos),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=registrar_ingreso)

with FaceDetector.create_from_options(options) as detector:
    with GestureRecognizer.create_from_options(opc_de_rec_de_manos) as reconocedor:
        try:
            #Activo la camara
            cv.namedWindow("guardar")
            cv.setMouseCallback("guardar", guardar)
            camara = cv.VideoCapture(0, cv.CAP_DSHOW)
            #inicializo el modelo de analisis de caras
            mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device='cpu')
            resnet = InceptionResnetV1(pretrained='vggface2').eval()
            #Cargo por primera vez las caras ya detectadas
            caras_conocidas = consultar_embeddings()
            #Ejecuto un ciclo mientras la camara este disponible
            while camara.isOpened():
                #Leo el cuadro
                ret, cuadro = camara.read()
                cuadro_RGB = cv.cvtColor(cuadro, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cuadro_RGB)
                start = cv.getTickCount() / cv.getTickFrequency()
                frame_timestamp_ms = int(time.time() * 1000)
                frame_timestamp_ms += 33
                #Detecto las caras
                detector.detect_async(mp_image, frame_timestamp_ms)
                reconocedor.recognize_async(mp_image,frame_timestamp_ms)
                alto, ancho, _ = cuadro.shape
                fecha = datetime.now()
                if resultados is not None:
                    #Reviso si se alcanzaron los 3 frames
                    if contador_de_frames >= 100:
                        realizar_la_deteccion = True
                    #Por cada cara dibujo un rectangulo
                    for cara_detectada in resultados.detections:
                        bbox = cara_detectada.bounding_box
                        x_origen = bbox.origin_x
                        y_origen = bbox.origin_y
                        x_destino = bbox.origin_x + bbox.width
                        y_destino = bbox.origin_y + bbox.height
                        ultima_cara = cuadro
                        cuadro = cv.rectangle(cuadro, (x_origen, y_origen), (x_destino, y_destino),(255,0,0),1)
                        cara_a_comparar = cuadro[y_origen:y_destino,x_origen:x_destino]
                        cara_a_comparar = cv.resize(cara_a_comparar, (160,160))
                        usuario = comparar_caras(cara_a_comparar, caras_conocidas, 0.8)
                        if usuario is not None:
                            cuadro = cv.putText(cuadro, usuario, (x_origen, y_origen-10), cv.FONT_ITALIC, 1.0, (255,255,0), 1, cv.LINE_AA)
                            proyecto = "LIS"
                            if manos_detectadas and manos_detectadas.gestures and manos_detectadas.hand_landmarks:
                                #Tomo la ubicación de la muñeca
                                muñeca = manos_detectadas.hand_landmarks[0][0]
                                #Cordenadas de la muñeca procesadas para comparar
                                x_muñeca = int(muñeca.x * ancho)
                                y_muñeca = int(muñeca.y * alto)
                                centro_de_la_cara = x_origen + (bbox.width/2)
                                geometria_ok = y_origen <= y_muñeca <= y_destino and abs(x_muñeca - centro_de_la_cara) < (bbox.width *2.5)
                                if geometria_ok:
                                    if manos_detectadas.gestures[0][0].category_name == "Open_Palm":
                                        if usuario not in personas_que_saludaron:
                                            hora_de_llegada = fecha.strftime('%H:%M')
                                            ingreso = [fecha.date().strftime("%Y-%m-%d"), usuario, proyecto, hora_de_llegada, "N/A"]
                                            ingresos_y_egresos[usuario] = ingreso
                                            personas_que_saludaron.add(usuario)
                                    if manos_detectadas.gestures[0][0].category_name == "Thumb_Up":
                                        if usuario in personas_que_saludaron:
                                            hora_de_salida = fecha.strftime('%H:%M')
                                            ingreso = ingresos_y_egresos[usuario]
                                            ingreso[4] = hora_de_salida
                                            personas_que_saludaron.remove(usuario)
                            if (realizar_la_deteccion):#Era para comparar cada tres frames
                                hora_actual = fecha.strftime('%H:%M')
                                asistencia = [fecha.date().strftime("%Y-%m-%d"), usuario, proyecto, hora_actual]
                                asistencias[usuario] = asistencia
                    if (realizar_la_deteccion):
                        contador_de_frames = 0
                        realizar_la_deteccion = False
                    cuadro = cv.putText(cuadro, f"FPS: {contador_de_frames}", (15, 15), cv.FONT_ITALIC, 1.0, (255,255,0), 1, cv.LINE_AA)
                    cv.imshow('guardar', cuadro)
                    contador_de_frames+=1
                    if cv.waitKey(1) & 0xFF == 27:
                        break
        except Exception as e:
            print(f"{e}")
        finally:
            guardar_asistencias(asistencias)
            guardar_ingresos(ingresos_y_egresos)
            if camara is not None:
                camara.release()
                cv.destroyAllWindows() 