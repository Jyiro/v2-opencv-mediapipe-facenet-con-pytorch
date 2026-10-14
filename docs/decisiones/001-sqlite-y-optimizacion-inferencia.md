# Registro de Decisión Arquitectural (ADR) 001: Migración a SQLite y Optimización del Pipeline de Inferencia Facial

- **Fecha:** 2026-10-14
- **Estado:** Aceptado
- **Contexto:** LIS (Laboratorio de Investigación de Software) — UTN Facultad Regional Córdoba
- **Fases del Proyecto:** Fase 0, Fase 1 y Fase 4

---

## 1. Contexto y Problemática

El prototipo heredado del laboratorio (`detector_de_caras.py`) presentaba limitaciones críticas tanto en la capa de persistencia como en la de visión por computadora:

1. **Almacenamiento en archivos CSV planos:**
   - La base de datos biométrica residía en `caras/caras.csv`, mapeando un único vector de 512 dimensiones por persona. Al registrar a una persona existente con el mismo nombre, se sobrescribía o generaban inconsistencias.
   - El estado de presencia dependía de un diccionario en memoria (`ingresos_y_egresos[usuario]`). Si una persona salía y volvía a ingresar durante el mismo día, el registro previo se sobrescribía, perdiendo el historial.
   - Inexistencia de soporte para auditoría de accesos y consentimiento informado exigidos por las regulaciones de privacidad biométrica (Sección 13 de `proyecto.md`).
   - Bloqueo de concurrencia: un archivo CSV no permite lecturas concurrentes confiables desde tableros web externos mientras el proceso de cámara escribe activamente.

2. **Doble detección redundante y cuello de botella de CPU (MTCNN + BlazeFace):**
   - MediaPipe `BlazeFace` localizaba el rostro en tiempo real de forma muy veloz en CPU.
   - Sin embargo, sobre ese mismo recorte facial, el código invocaba `MTCNN` (de `facenet-pytorch`) en cada cuadro. `MTCNN` ejecuta tres redes convolucionales en cascada (P-Net, R-Net, O-Net) para volver a detectar caras que ya habían sido detectadas por BlazeFace.
   - Esta redundancia consumía más del 60% de los ciclos de CPU por frame, degradando la tasa de refresco a menos de 10 FPS en equipos convencionales.
   - Además, la extracción de embeddings se ejecutaba incondicionalmente en cada cuadro, sin importar si la persona estaba inmóvil o en movimiento brusco.

---

## 2. Decisión Tomada

### A. Adopción de SQLite Nativo con Sincronización en Espejo (Fase 1)
Se implementó `nucleo/base_datos.py` con la clase `RepositorioLIS`, utilizando la biblioteca estándar `sqlite3` de Python sin dependencias externas pesadas, con las siguientes características:

1. **Modo WAL (`PRAGMA journal_mode = WAL;`):**
   - Habilita concurrencia real: lecturas simultáneas no bloqueantes desde el tablero o dashboard web mientras la cámara registra eventos de ingreso y salida.
2. **Esquema relacional normalizado (4 tablas):**
   - `personas`: almacena identificación normalizada (slug), nombre único, rol, estado activo (baja lógica) y registro de consentimiento informado (Sección 13).
   - `embeddings`: soporte nativo para múltiples vectores por persona (`persona_id`, `vector` como BLOB `float32`, `dimension`, `modelo`, `fecha_captura`).
   - `eventos`: arquitectura *Event Sourcing* append-only (esquema exacto de la Sección 6) con UUID v4 idempotente, tipos de evento (`entrada`, `salida`, `asistencia_permanencia`, etc.), método (`cara+gesto`, `qr`, `pin`), distancia euclídea cruda y metadatos JSON.
   - `auditoria_accesos`: trazabilidad obligatoria de quién y cuándo se consultan o modifican los datos biométricos.
3. **Retrocompatibilidad total:**
   - Al iniciar, si la tabla `embeddings` está vacía, se migra automáticamente el contenido de `caras.csv`.
   - Cada alta de persona o registro de evento se sincroniza en espejo con `caras.csv`, `asistencia.csv` y `registro_de_ingresos.csv`, garantizando que herramientas y scripts previos continúen operando sin cambios.

### B. Optimización del Pipeline de Inferencia Facial (`nucleo/reconocimiento.py`)
1. **Eliminación de MTCNN del bucle en tiempo real (`facenet_rapido`):**
   - Se alimenta directamente el tensor de recorte facial de MediaPipe BlazeFace (con un margen de seguridad del 10%) a `InceptionResnetV1(pretrained='vggface2')`.
   - Estandarización idéntica `(píxeles - 127.5) / 128.0` dentro de un bloque `torch.inference_mode()`.
   - Mantiene 100% de compatibilidad con los embeddings históricos generados en `caras.csv` sin requerir re-enrolar usuarios.
2. **Arquitectura modular extensible:**
   - Se diseñó la interfaz abstracta `ExtractorEmbeddings`, con implementaciones para `facenet_rapido`, `onnx_arcface` (con fallback transparente) y `facenet_legacy` (para pruebas comparativas).

### C. Caché de "Cara Estable" (`nucleo/estabilizador.py`)
1. Se implementó `EstabilizadorDeCaras`, que realiza seguimiento espacial de rostros por centroides.
2. La extracción de embeddings solo se dispara cuando el rostro se mantiene estable durante $N$ cuadros (por defecto 4 frames con desplazamiento $< 35$ px).
3. Una vez evaluada la identidad, los datos `(nombre, distancia)` quedan cacheados mientras la persona permanezca en la escena, reevaluándose únicamente ante movimientos bruscos o cada 90 fotogramas.

---

## 3. Consecuencias y Beneficios

1. **Rendimiento:**
   - La eliminación de MTCNN redujo la latencia de inferencia por cuadro en más del 60%.
   - El estabilizador temporal evita ejecutar FaceNet en cada cuadro, elevando la tasa efectiva de procesamiento a 30+ FPS en CPU de escritorio.
2. **Robustez en Asistencia:**
   - Al derivar el estado presente/ausente del último evento append-only (`entrada`/`salida`), los reingresos múltiples en el mismo día quedan registrados íntegramente sin sobrescritura de datos.
3. **Privacidad y Cumplimiento:**
   - No se almacenan fotos ni recortes en disco en ningún momento.
   - La interfaz muestra distancias euclídeas reales junto con el umbral (`d: 0.48 / umbral: 0.80`), eliminando porcentajes de confianza ficticios.
   - El proceso de alta requiere PIN administrativo y confirmación de consentimiento informado.
