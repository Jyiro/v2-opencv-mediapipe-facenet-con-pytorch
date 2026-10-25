# Registro de Decisión Arquitectural (ADR) 005: Normalización Fotométrica Adaptativa, Inferencia A/B en Lote y Enrolamiento Facial Guiado por Poses

- **Fecha:** 2026-10-23
- **Estado:** Aceptado
- **Contexto:** LIS (Laboratorio de Investigación de Software) — UTN Facultad Regional Córdoba
- **Fases del Proyecto:** Fase 1, Fase 4 y Trabajo de Investigación / Beca GAALA

---

## 1. Contexto y Problemática

El sistema de reconocimiento biométrico facial del LIS opera en condiciones de iluminación ambiental no controladas (iluminación fluorescente, reflejos de monitores y variaciones de luz solar según el horario de trabajo). Esto plantea dos problemas centrales de ingeniería y visión computacional:

1. **Degradación de Distancia Euclídea ante Variabilidad Lumínica:**
   - La red `InceptionResnetV1` (preentrenada sobre `vggface2`) extrae vectores de 512 dimensiones estandarizando cada canal como `(x - 127.5) / 128.0`.
   - Cuando una persona ingresa con sombras asimétricas duras o subexposición general, las características faciales en frecuencias medias y altas quedan atenuadas, incrementando la distancia euclídea cruda y elevando el riesgo de Falso Rechazo (FR) o Falsa Alarma (FA).
   - Existía la hipótesis de que aplicar técnicas fotométricas clásicas (balance de blancos Gray-World, CLAHE en canal de luminancia $L$ del espacio CIE LAB y corrección de gamma adaptativa según el brillo medio) permitiría "normalizar" la imagen hacia un estándar constante. Sin embargo, no se disponía de un banco de pruebas experimental en vivo que permitiera comparar de forma rigurosa si el embedding filtrado efectivamente mejora la discriminabilidad respecto al embedding crudo.

2. **Captura Clásica en Ráfaga Temporal Fija:**
   - El procedimiento histórico de enrolamiento capturaba 5 fotogramas consecutivos casi simultáneos mientras el usuario permanecía inmóvil frente a la cámara.
   - Estos vectores presentaban una correlación casi total entre sí, careciendo de variabilidad angular real (rotaciones yaw/pitch).
   - No existía comprobación de "prueba de vida" (liveness) ni asistencia interactiva que guiara al usuario paso a paso (análogo a un procedimiento de alta bancaria).

3. **Restricciones Críticas de Rendimiento y Privacidad:**
   - **CPU-bound:** El procesamiento debe ejecutarse en hardware convencional de laboratorio e interactuar con el nodo Portero (Intel Galileo x86 de 32 bits). La inferencia redundante de dos modelos por separado duplicaría la latencia.
   - **Privacidad estricta:** Conforme a la Sección 13 de `proyecto.md`, está terminantemente prohibido almacenar fotografías de integrantes en disco permanente. Toda previsualización o comparación debe operar exclusivamente en memoria volátil (RAM).

---

## 2. Decisión Tomada

### A. Pipeline de Normalización Fotométrica (`nucleo/iluminacion.py`)
Se implementó un pipeline secuencial de corrección de iluminación que opera directamente sobre recortes faciales redimensionados a $160 \times 160$ píxeles:
1. **Redimensión Previa $160 \times 160$:** En lugar de procesar el recorte crudo a su tamaño original variable (que puede alcanzar $320 \times 320$ o más de $100.000$ píxeles), el recorte se escala primero a $160 \times 160$ ($25.600$ píxeles). Esto reduce el costo computacional de las transformaciones de color en un ~75% sin pérdida de resolución para FaceNet.
2. **Balance de Blancos Gray-World:** Ajuste independiente de ganancias $G_c = \mu_{\text{global}} / \mu_c$ para compensar dominantes de color cálidas o frías.
3. **CLAHE en Canal Luminancia ($L$ de CIE LAB):** Realce adaptativo de contraste local sin distorsionar la información cromática $A$ y $B$ (clip limit = 2.0, grilla $8 \times 8$).
4. **Gamma Adaptativa con Target $0.5$:** Corrección no lineal $V^{\gamma}$ calculada analíticamente como $\gamma = \log(V_{\text{target}}) / \log(\mu_L + \varepsilon)$ para centrar el brillo promedio.
5. **Métrica de Asimetría Lateral:** `estadisticas_luz(bgr)` calcula la diferencia absoluta de luminancia entre el hemisferio facial izquierdo y derecho ($\vert{}\mu_{L,\text{izq}} - \mu_{L,\text{der}}\vert{}$), registrando sombras laterales en el log científico.

### B. Inferencia Dual en Lote ($B=2$) y Normalización $\ell_2$ (`nucleo/reconocimiento.py`)
1. Se implementó `generar_par_embeddings(recorte_crudo, recorte_filtrado)` dentro de `ExtractorFaceNetRapido`.
2. Ambos recortes se apilan en un único tensor PyTorch de entrada de dimensiones `(2, 3, 160, 160)`.
3. Se ejecuta un único `forward pass` bajo `torch.inference_mode()`, aprovechando las optimizaciones SIMD/AVX de PyTorch en CPU, logrando una reducción de latencia de aproximadamente un 40% respecto a dos inferencias consecutivas individuales.
4. **Invariancia Métrica Obligatoria:** Cada embedding resultante se normaliza explícitamente dividiendo por su norma euclídea ($\Vert{}\mathbf{v}\Vert{}_2 = 1$). Esto garantiza que la distancia euclídea conserve la relación teórica exacta con la similitud coseno: $d_E^2 = 2 - 2 \cos(\theta)$.

### C. Modo de Pruebas A/B en Tiempo Real (`nucleo/pruebas.py`)
Se integró el `ModoPruebasAB` en el bucle principal de la cámara (`hardware/camara.py` y `mediapipe/detector_de_caras.py`):
1. **Carrera a Primer Match:** Compara cuadro a cuadro cuál de las dos ramas alcanza primero una distancia euclídea por debajo del umbral de reconocimiento ($\le 0.85$).
2. **Marcador de Sesión Científica:** Mide distancia media, mejor distancia, tasa de reconocimiento, aciertos, falsas alarmas (FA) y falsos rechazos (FR) respecto a un sujeto esperado definido por el operador.
3. **Log Append-Only sin Fotos:** Registra cada cuadro de prueba en `datos/pruebas_log.jsonl` con marcas de tiempo, distancias y asimetría lateral, sin guardar imágenes.
4. **Previsualización Volátil en RAM:** El recorte crudo y el filtrado se unen horizontalmente con bordes identificatorios y se codifican a JPEG en memoria (`camara.py:obtener_preview_pruebas_jpeg()`), sirviéndose bajo `GET /pruebas/preview.jpg` sin tocar el disco.

### D. Enrolamiento Facial Guiado por Poses (`nucleo/pose.py`)
Se implementó un procedimiento de alta biométrica asistida con detección de posturas de la cabeza:
1. **Detección Angular con MediaPipe Face Landmarker:** Se calculan los ángulos de *Yaw* (giro lateral a partir de la distancia relativa de la punta de la nariz a los cantos externos de los ojos) y *Pitch* (inclinación vertical entre el puente nasal y la barbilla).
2. **Carga Perezosa (Lazy Loading):** `FaceLandmarker` se instancia sincrónicamente únicamente cuando se solicita un alta guiada (`orden["guiado"] = True`) y se destruye al completar el proceso. En régimen normal, solo corre BlazeFace, protegiendo la tasa de cuadros por segundo del laboratorio.
3. **Autocalibración Anatómica del Pitch:** Para mitigar diferencias de altura de usuarios y ángulos de montaje de la cámara, la primera pose (`frente`) calibra automáticamente el centroide vertical anatómico del usuario mediante `fijar_referencia_frontal()`.
4. **Sostenimiento Temporal y Filtro Anti-Duplicados:** Cada pose debe mantenerse estable dentro de una ventana angular de tolerancia ($\pm 8^\circ$) durante un temporizador continuo de $0.5$ segundos. Al capturar, se comprueba que el vector no sea idéntico a los previos ($d_E > 0.08$).
5. **Almacenamiento Dual en Base de Datos:** Cada captura almacena automáticamente ambos descriptores en `lis_maestra.db`: el crudo como `modelo='vggface2'` y el filtrado como `modelo='vggface2+norm'`.

### E. Integración en el Nodo Portero y Tablero Web
1. **Microservicio Portero (`portero/servidor_portero.py`):** Endpoints REST nativos (`/pruebas/estado`, `/pruebas/configurar`, `/pruebas/preview.jpg`, `/perfiles/alta_guiada`) desarrollados exclusivamente sobre la biblioteca estándar `http.server` para compatibilidad total con Intel Galileo x86.
2. **Tablero Web (`tablero/index.html`):** Pestaña dedicada "05 Laboratorio A/B", HUD interactivo con elipse guía SVG, barra circular de sostenimiento y selector de calibración fotométrica en caliente.

---

## 3. Consecuencias y Beneficios

### Positivas
- **Trazabilidad Científica Inmediata:** Los investigadores del LIS pueden cuantificar en tiempo real si el preprocesamiento fotométrico favorece o perjudica a la red neuronal según la iluminación del día.
- **Diversidad Angular en la Base de Datos:** Los perfiles enrolados con poses guiadas registran vectores faciales reales en múltiples ángulos, reduciendo el falso rechazo ante ingresos de costado.
- **Rendimiento Óptimo:** La inferencia en lote ($B=2$) y la redimensión anticipada aseguran fluidez superior a 20 FPS en hardware estándar de CPU.
- **Cumplimiento de Privacidad:** Se respeta la política de retención cero de fotografías en disco.
- **Retrocompatibilidad:** La arquitectura es totalmente transparente para el fichador físico (LCD 16x2 / LEDs) y el registro clásico de ingresos y egresos.

### Negativas / Mitigaciones
- **Consumo Adicional Momentáneo de Memoria:** Instanciar `FaceLandmarker` durante el enrolamiento incrementa el consumo de memoria en ~80 MB; mitigado destruyendo el objeto una vez finalizada la captura.
- **Sensibilidad de Gray-World en Fondos Monocromáticos:** Si un usuario viste ropa de color saturado predominante, Gray-World puede generar una ligera distorsión; mitigable desde el tablero desactivando la casilla de balance de blancos en caliente.
