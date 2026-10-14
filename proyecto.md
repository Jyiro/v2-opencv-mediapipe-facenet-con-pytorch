# Plataforma de herramientas internas del LIS — Contexto y guía para el agente

> Documento de contexto para guiar a un agente de desarrollo (IDE con IA).
> Léelo completo antes de tocar código. Si algo contradice lo que ves en el repo, **el repo manda**: avisa la diferencia y pregunta antes de asumir.

---

## 0. Cómo usar este documento (para el agente)

1. Lee las secciones 1 a 4 para entender el objetivo, las restricciones y el estado actual.
2. Trabaja **por fases** (sección 15). No saltees fases ni mezcles varias en un mismo cambio.
3. Cada fase tiene criterios de aceptación. Una fase no está terminada hasta que se cumplan.
4. Respeta las reglas de la sección 16.
5. Lo marcado como **VERIFICAR** es un dato sin confirmar. No lo des por cierto: pregúntale al humano o escribe código que lo compruebe.
6. Ante la duda sobre datos personales, modelo de reconocimiento o borrar funcionalidad existente: **detente y pregunta**.

---

## 1. Objetivo

Construir, sobre el hardware que ya tiene el Laboratorio de Investigación de Software (LIS, UTN FRC), un conjunto de **herramientas internas para el laboratorio** que reemplacen procesos manuales actuales:

| Hoy | Objetivo |
|---|---|
| Firmas y horas en papel para entrada y salida | Registro automático de entrada/salida (reconocimiento facial + gesto, con respaldo QR/PIN) y cálculo de horas |
| Grupo de WhatsApp para saber quién se llevó qué | Inventario con préstamo y devolución, tablero "qué hay y quién lo tiene" |
| Notas pegadas a los costados del pizarrón | Tablero de avisos con dueño y vencimiento |
| Nada | Estado del laboratorio (abierto, ruido, temperatura, luz) con sensores |

Es un proyecto **real, con usuarios reales** (los integrantes del lab). Eso lo diferencia de una demo: debe ser confiable, simple de usar y respetar la privacidad.

El proyecto también produce material de investigación en ingeniería de software (ver sección 12 y 17): diseño offline-first con servidor intermitente, calibración del reconocimiento con datos físicos de sensores, y pruebas de engaño (ataques de presentación).

---

## 2. Contexto y restricciones

- **Quién desarrolla:** un estudiante universitario del grupo, con avances semanales presentados al laboratorio. Cada semana debe terminar en algo demostrable.
- **Punto de partida:** ya existe un proyecto iniciado por un profesor del lab (repo `opencv-mediapipe-facenet-con-pytorch`, organización UTN-LIS). Los profesores validaron que se **expanda ese proyecto**. No se parte de cero ni se reescribe: se extiende.
- **El servidor no está encendido 24/7.** Una persona lo enciende y apaga. **Esta es la restricción que más condiciona la arquitectura** (sección 5).
- **Hardware viejo y limitado** (sección 4). No asumir potencia de cómputo en las placas ni en las tablets.
- **Datos personales biométricos** involucrados (sección 13). Requisito, no extra.
- Idioma de trabajo: **español** (documentación, mensajes, comentarios). Respetar el estilo de nombres del código existente (identificadores en español, p. ej. `detector_de_caras.py`, `consultar_embeddings`).

---

## 3. Estado actual del código (según un análisis previo del repo)

> Fuente: análisis hecho sobre el repo clonado. **VERIFICAR** contra el código real antes de apoyarse en cualquier detalle.

### Estructura

```
opencv-mediapipe-facenet-con-pytorch/
└── mediapipe/
    ├── detector_de_caras.py          # script único (~268 líneas)
    ├── blaze_face_short_range.tflite # detector de caras (MediaPipe)
    ├── face_landmarker.task          # landmarks (no se usa hoy)
    ├── caras/caras.csv               # base de embeddings + nombre
    ├── asistencia/asistencia.csv     # salida: asistencias
    └── manos/
        ├── gesture_recognizer.task   # gestos (MediaPipe)
        └── registro_de_ingresos.csv  # salida: ingresos y egresos
```

### Pipeline actual

1. Inicializa `FaceDetector` (BlazeFace) y `GestureRecognizer` de MediaPipe en modo `LIVE_STREAM` (callbacks asíncronos), y carga `MTCNN` + `InceptionResnetV1` (pesos `vggface2`) de `facenet_pytorch`.
2. Lee `caras.csv` y arma un diccionario `nombre → embedding` (512 dimensiones).
3. Captura con OpenCV, convierte BGR→RGB, y envía cada cuadro a `detect_async` y `recognize_async`.
4. Recorta cada cara a 160×160 y genera el embedding (MTCNN → InceptionResnetV1).
5. Compara por **distancia euclídea** contra todos los embeddings conocidos; si la mínima es **< 0.8**, devuelve el nombre.
6. **Gestos:** si la persona fue reconocida y la muñeca (landmark 0) está a la altura y cerca de la cara (`geometria_ok`): `Open_Palm` registra **ingreso**, `Thumb_Up` registra **egreso** (solo si ya había ingresado).
7. "Asistencia por permanencia": un contador de frames llega a 100 y registra `[fecha, usuario, proyecto, hora]` mientras se reconoce a alguien.
8. Alta de personas: doble clic sobre una cara abre un diálogo Tkinter para el nombre; genera el embedding, verifica que no exista (umbral 0.8) y lo agrega a `caras.csv`.
9. Al cerrar (ESC o excepción) escribe `asistencia.csv` y `registro_de_ingresos.csv`.

### Problemas conocidos (a tratar en las fases correspondientes)

| Problema | Fase |
|---|---|
| Rutas hardcodeadas de Windows (`C:\Users\...`) y `cv.CAP_DSHOW` (solo Windows) | 0 |
| `imshow`/`waitKey` dentro de `if resultados is not None`: sin detección la ventana no se refresca | 0 |
| Comentarios dicen "3 frames" pero el código usa 100; la etiqueta "FPS" muestra el contador de frames | 0 |
| `generar_embedding` captura excepciones y devuelve `None` sin avisar | 0 |
| MTCNN + ResNet en **cada cuadro y por cada cara**, solo CPU | 4 |
| Un solo embedding por persona; registrar el mismo nombre sobrescribe | 4 |
| Reingreso: `ingresos_y_egresos[usuario]` se sobrescribe y se pierde el registro previo | 1 |
| Umbral 0.8 fijo, sin calibración con datos propios | 4 |
| Doble clic registra a cualquiera; embeddings en CSV sin protección | 13 |
| "Asistencia por permanencia" registra solo por estar frente a la cámara (vigilancia pasiva) | 2 y 13 |
| `face_landmarker.task` presente pero sin uso | 12 |

---

## 4. Hardware disponible

| Ítem | Cantidad | Notas |
|---|---|---|
| **Grove Starter Kit Plus – Intel IoT Edition, Gen 1** | 4 o más (en cajas, con USB) | Ver contenido abajo |
| **Intel Galileo** | Las que vienen con los kits | **VERIFICAR** generación (Gen 1/2), imagen de Linux cargada, si trae Python/Node, y si mantiene la hora sin corriente (RTC con pila) |
| **Arduino UNO (clones)** | Una caja bastante llena (cajón 6) | Hay al menos dos variantes: una con ATmega328P en DIP y otra con ATmega328P SMD y probable CH340 como USB-serie (puede requerir driver). **VERIFICAR** con un "blink" cada placa |
| **Tablets Intel** | Algunas, muy viejas | **VERIFICAR** marca, modelo, sistema operativo, cámara y navegador. Son x86 |
| **Servidor** | 1, no 24/7 | **VERIFICAR** sistema operativo, RAM, CPU/GPU. Una persona lo enciende/apaga |
| Dron/LEGO | Posible, sin confirmar | **Fuera de alcance** por ahora |

### Contenido de cada kit Grove

Base Shield ×1, LCD RGB backlight ×1, relay inteligente ×1, buzzer ×1, sensor de sonido ×1, sensor táctil ×1, sensor de ángulo rotatorio ×1, sensor de temperatura ×1, LED ×1, botón ×1, LEDs DIP (azul, verde, rojo) ×1 de cada uno, mini servo ×1, 10 cables Grove, adaptador 9V a barrel jack, manual, caja plástica verde.

**El kit NO trae:** LED RGB, sensor de movimiento (PIR), distancia, acelerómetro, color ni NFC/RFID. Para indicar estado se usa el **color del fondo del LCD RGB** y los LEDs DIP.

### Limitaciones a respetar

- Las Galileo son de 32 bits, con muy poca RAM y CPU. **No** correr modelos de ML ahí ni dependencias pesadas. Preferir biblioteca estándar.
- Con un solo ejemplar de cada módulo por kit, **un nodo = un kit**. Con 4+ kits se arman 4+ nodos.
- Las tablets viejas pueden tener navegadores muy limitados. Interfaces en HTML/JS simple, sin frameworks pesados, y probar antes.

---

## 5. Arquitectura

### Principios

1. **Offline-first:** el sistema debe funcionar con el servidor apagado. El servidor suma historial, reportes y cómputo pesado, pero no es imprescindible para registrar.
2. **Event sourcing:** cada cosa que pasa es un **evento inmutable** con id único. El estado actual (quién está, quién tiene qué) se **deriva** de los eventos.
3. **Degradación progresiva:** reconocimiento facial → QR/PIN → caja fuerte en la Galileo. Cada nivel funciona sin el anterior.
4. **Adaptadores, no reescritura:** el pipeline de reconocimiento no conoce el hardware. Emite eventos a un "destino de eventos" y recibe retroalimentación por una interfaz.
5. **Privacidad por diseño:** solo embeddings (nunca fotos), consentimiento, la cámara no reconoce de forma pasiva (sección 13).

### Diagrama

```
 cámara (webcam USB o tablet por stream)
        │
        ▼
 PIPELINE PYTHON (MediaPipe + FaceNet)  ◄──serie──►  FICHADOR (UNO + Grove)
        │ eventos                                      LCD · buzzer · LEDs · touch
        ▼
 PORTERO (Galileo, siempre prendida)  ◄── caja fuerte (Galileo + Grove: perilla + botón)
   · registro JSONL append-only
   · API HTTP mínima
   · estado derivado
        │                         ▲
        │ pull al encenderse      │ datos de sensores
        ▼                         │
 SERVIDOR (intermitente)     TORRE DE ESTADO / ARMARIO (UNO + Grove)
   · base maestra
   · reportes (CSV/PDF)
   · backups, bot opcional
        │
        ▼
 TABLERO (tablet de pared, HTML simple) ← lee del Portero
```

### Componentes

| Componente | Dónde corre | Responsabilidad |
|---|---|---|
| **Pipeline** | Máquina con cámara y recursos (**VERIFICAR** cuál: servidor, PC o ambos) | Detección, embedding, comparación, gestos. Emite eventos |
| **Fichador** | Arduino UNO conectado por USB a la máquina del pipeline | Feedback físico (LCD, buzzer, LEDs). El touch/botón despierta la cámara |
| **Caja fuerte (respaldo)** | Galileo con Base Shield | PIN girando el sensor rotatorio + botón, LCD de feedback. Funciona con el servidor apagado |
| **Portero** | Galileo (siempre prendida) | Fuente de verdad en el borde: recibe eventos, los persiste, sirve estado |
| **Servidor** | Máquina del lab | Base maestra, reportes, backups. **Tira (pull)** de los eventos del Portero cuando se enciende |
| **Torre de estado** | UNO + temperatura, luz, sonido, rotary, LCD | Publica el estado del lab |
| **Armario** | UNO + servo, relay, touch | Traba y aviso del armario de equipos |
| **Tablero** | Tablet de pared | Muestra avisos, quién tiene qué, estado del lab, si el servidor está encendido |

Decisión clave: el **servidor tira los eventos del Portero** (pull). Así el Portero es pasivo, no necesita descubrir al servidor y la sincronización es simple.

---

## 6. Modelo de eventos

### Esquema

```json
{
  "id": "uuid-v4",
  "tipo": "entrada | salida | retiro | devolucion | aviso_creado | aviso_cerrado | estado_lab | ambiente | correccion",
  "persona_id": "string | null",
  "objeto_id": "string | null",
  "t": "2026-10-09T14:03:22Z",
  "reloj_confiable": true,
  "origen": "pipeline-facial | caja-fuerte | tablet-puerta | torre-estado | armario",
  "metodo": "cara+gesto | qr | pin | manual | sensor",
  "distancia": 0.52,
  "meta": { "lux": 310, "umbral": 0.8, "modelo": "vggface2" }
}
```

### Reglas

- **Append-only.** Nunca se edita ni se borra un evento. Las correcciones son un evento `correccion` que referencia el `id` original.
- **Idempotencia:** el `id` es único. Enviar el mismo evento dos veces no debe duplicarlo (el receptor deduplica por `id`).
- **Estado derivado:** "quién está" = último evento `entrada`/`salida` por persona. "Quién tiene qué" = último `retiro`/`devolucion` por objeto. Reingresos múltiples en el día son válidos y quedan todos.
- **Hora:** cada evento lleva `reloj_confiable`. Si la Galileo no mantiene la hora sin corriente, hasta sincronizar (NTP o desde el pipeline) marca `false` y se corrige al sincronizar. Un reporte de horas **no debe** confiar en eventos con reloj no confiable sin revisión.
- **Distancia en vez de "confianza":** guardar la distancia euclídea cruda y el umbral usado. No inventar porcentajes.

---

## 7. Identificación por capas

1. **Cara + gesto (nivel principal, requiere el pipeline activo):** reconocimiento + `Open_Palm` (entrada) / `Thumb_Up` (salida), como ya hace el código. Solo el **gesto explícito (o el touch del fichador)** registra. Estar frente a la cámara no registra nada.
2. **QR o PIN en tablet de puerta (respaldo):** cuando el pipeline no está disponible. El evento lleva `metodo: qr|pin`.
3. **Caja fuerte en la Galileo (respaldo de último nivel, sin servidor ni tablet):** PIN girando el sensor rotatorio (mapeado a dígitos 0–9), confirmando con el botón, feedback en LCD y buzzer.
4. **(Posterior) QR rotatorio:** la tablet de la puerta muestra un QR que cambia cada ~30 s (derivado con HMAC de una clave del lugar y el tiempo), que la persona escanea con su celular. Prueba presencia porque hay que ver la pantalla.

Cuando se use un nivel de respaldo, el evento queda marcado con su método para poder revisar después.

---

## 8. Fichador: protocolo serie PC ↔ Arduino

Línea de texto ASCII terminada en `\n`, **115200 baudios**. Campos separados por `|`. El LCD es de 16×2: truncar cada línea a 16 caracteres.

### PC → Arduino

| Comando | Efecto |
|---|---|
| `MSG\|línea1\|línea2` | Muestra texto en el LCD |
| `RGB\|r\|g\|b` | Color del fondo del LCD (0–255) |
| `BEEP\|ok` / `BEEP\|err` / `BEEP\|warn` | Patrones de buzzer distintos |
| `LED\|azul\|1` (también `verde`, `rojo`, con 0/1) | Enciende o apaga un LED DIP |
| `PING` | El Arduino responde `PONG` |

### Arduino → PC

| Mensaje | Significado |
|---|---|
| `TOUCH` | Se tocó el sensor táctil (despierta la cámara / confirma) |
| `BTN` | Se apretó el botón |
| `PONG` | Respuesta a `PING` |
| `SENS\|temp=..\|luz=..\|sonido=..` | Lecturas periódicas (opcional) |

### Notas de firmware

- El LCD RGB de Grove usa I2C; la biblioteca típica es `rgb_lcd` de Seeed (**VERIFICAR**).
- Definir y **documentar** en el README del firmware qué sensor va a qué puerto del Base Shield. No dejar la asignación solo en el código.
- Si el Arduino no responde a `PING`, el pipeline **debe seguir funcionando** sin feedback físico (degradar, no fallar).
- Placas con CH340: puede requerir driver según el sistema operativo.

---

## 9. API del Portero (Galileo)

Mínima, pensada para hardware limitado. Persistencia en un archivo **JSONL append-only** (una línea por evento), con `fsync` razonable. Sin base de datos pesada.

| Método y ruta | Función |
|---|---|
| `POST /events` | Recibe una lista de eventos. **Idempotente** (dedup por `id`). Responde cuáles aceptó |
| `GET /events?since=<cursor>` | Devuelve eventos posteriores al cursor (lo usa el servidor para sincronizar) |
| `GET /state` | Estado derivado: quién está, quién tiene qué, estado del lab |
| `GET /health` | Hora, versión, cantidad de eventos, si el reloj es confiable |

- **Autenticación:** token compartido por dispositivo en un encabezado. Solo red local. Documentar cómo rotar tokens.
- **Lenguaje:** depende de lo que tenga la imagen de la Galileo (**VERIFICAR**). Si hay Python, biblioteca estándar. Si no, C o lo disponible. **No** agregar dependencias que no compilen en 32 bits.
- **Sincronización con el servidor:** al encenderse, el servidor consulta `GET /events?since=<último cursor>`, los guarda en su base maestra y recuerda el cursor. Si el servidor se apaga a mitad, reintenta desde el último cursor confirmado (la idempotencia lo hace seguro).

---

## 10. Sensores adicionales

### Torre de estado (UNO + temperatura, luz, sonido, rotary, LCD)

- **Luz:** ¿luces encendidas?, señal indirecta de "lab abierto".
- **Sonido:** nivel de ruido (no grabar audio, solo nivel).
- **Temperatura:** ambiente.
- **Rotary:** el integrante elige un modo (`disponible` / `reunión` / `no molestar`).
- **LCD:** el **color del fondo** indica el estado (por ejemplo verde/rojo/azul).
- Publica eventos `estado_lab` y `ambiente` al Portero a intervalos razonables (no en cada lectura).

### Armario (UNO + servo, relay, touch)

- El **servo** traba una puerta liviana; se libera tras una identificación válida (evento `retiro`).
- **Seguridad física:** la cerradura nunca debe depender solo de la electrónica. Debe poder abrirse con llave si falla la corriente o el sistema.
- El **relay** puede encender un cartel o lámpara. **No** cablear tensión de red sin supervisión de alguien con experiencia; preferir bajo voltaje.

---

## 11. Calibración y evaluación del reconocimiento

Objetivo: reemplazar el umbral fijo 0.8 por uno **medido con datos propios**.

1. **Alta de personas:** 3 a 5 fotos por persona en condiciones distintas → varios embeddings (guardar todos o un promedio; evaluar ambos enfoques).
2. **Conjunto de evaluación:** intentos reales de reconocimiento con metadatos: persona real, distancia obtenida, **lux** (sensor de luz), hora, y si el resultado fue correcto.
3. **Métricas:** tasa de falsos aceptados (FAR) y falsos rechazados (FRR) en función del umbral; elegir el punto de operación según la prioridad (en un lab chico, evitar registrar a la persona equivocada).
4. **Análisis por condiciones:** cómo cambian los errores con la iluminación. Eso genera un dataset propio de condiciones físicas contra errores, útil para un informe.
5. **Rendimiento:** correr FaceNet **solo cuando una cara está estable** unos cuadros, no en cada frame. Medir fps antes y después.

Scripts de evaluación reproducibles en `evaluacion/`, con resultados guardados (sin fotos ni embeddings reales en el repo).

---

## 12. Experimentos de seguridad (anti-engaño)

Con autorización explícita de los profesores:

- **Ataque de presentación:** intentar entrar con una foto o un video en un celular. Medir tasa de éxito.
- **Contramedidas a evaluar:** verificación de gesto aleatorio con MediaPipe (por ejemplo parpadeo o girar la cabeza; `face_landmarker.task` ya está en el repo), confirmación física con el touch, rechazo por poca luz.
- **Ataques al sistema:** reproducir eventos, falsificar un token, duplicar un evento; documentar y mitigar.
- Documentar cada experimento con método, resultados y mitigaciones. Es material para un informe técnico.

---

## 13. Privacidad y consentimiento (requisitos obligatorios)

El embedding facial es un dato personal biométrico. Requisitos:

- Guardar **solo embeddings**, **nunca fotos** (ni temporales en disco).
- **Consentimiento informado** por persona, registrado antes del alta.
- **Alternativa sin cara:** quien no quiera ser reconocido usa QR/PIN/caja fuerte, sin penalización.
- **El alta de personas requiere autorización de un administrador.** El doble clic actual que registra a cualquiera debe restringirse.
- `caras.csv` (o su reemplazo) con permisos restringidos; evaluar cifrado en reposo.
- **Política de retención:** definir cuánto se conservan los eventos y los embeddings, y cómo se borran cuando alguien se va.
- Quién puede ver qué: registro de acceso a los datos.
- La cámara **no reconoce de forma pasiva**: solo trabaja tras un disparo explícito (touch/botón) o gesto.
- Red local únicamente. Sin envío de datos a servicios externos.
- Documentar todo en `docs/PRIVACIDAD.md` y acordarlo con los profesores **antes** de enrolar a personas reales.
- **Nunca** commitear embeddings, fotos ni datos reales. Usar datos sintéticos para pruebas.

---

## 14. Estructura de repositorio propuesta

Mantener el código existente funcionando y agregar alrededor. Trabajar en una **rama o fork**, con cambios chicos y revisables.

```
/
├── mediapipe/                 # código original (se refactoriza gradualmente)
├── nucleo/                    # lógica pura, sin hardware, testeable
│   ├── eventos.py             # esquema, validación, estado derivado
│   ├── destinos.py            # interfaz DestinoDeEventos (CSV, Portero, serie)
│   └── comparacion.py         # matching de embeddings, umbrales
├── hardware/
│   ├── fichador/              # firmware Arduino + README con mapa de puertos
│   ├── serie.py               # cliente del protocolo serie (con modo simulado)
│   └── torre_estado/ armario/ # firmware
├── portero/                   # servicio para la Galileo (stdlib)
├── servidor/                  # sincronización y reportes
├── tablero/                   # HTML/JS simple para tablets
├── evaluacion/                # scripts de calibración y métricas
├── docs/
│   ├── PRIVACIDAD.md
│   ├── PROTOCOLO_SERIE.md
│   ├── bitacora/AAAA-MM-DD.md # qué se hizo cada semana, con resultados
│   └── decisiones/            # registros breves de decisiones técnicas
├── tests/
├── config.example.toml        # sin secretos
└── CONTEXTO_PROYECTO.md       # este documento
```

---

## 15. Plan por fases

Cada fase termina con algo demostrable.

### Fase 0 — Línea base y portabilidad
**Tareas:** correr el repo en el entorno de desarrollo; rutas relativas con `pathlib` y configuración externa (`config.toml`); cámara multiplataforma (`VideoCapture(0)` o parámetro); arreglar el refresco de ventana sin detección; corregir etiquetas de FPS y comentarios; reemplazar el `except` silencioso por logging; README de ejecución; medir y anotar fps.
**Aceptación:** el script corre en Linux y Windows con la misma configuración; ningún error silencioso; hay una nota con los fps medidos.

### Fase 1 — Eventos y destinos
**Tareas:** módulo `nucleo/eventos.py` (esquema, validación, estado derivado); interfaz `DestinoDeEventos`; destino CSV **compatible con los formatos actuales**; reemplazar la lógica de ingreso/egreso por eventos append-only (arregla el reingreso); modo `--simulate` (cámara y serie falsas); tests con `pytest`.
**Aceptación:** los CSV actuales se siguen generando; reingresos múltiples quedan registrados; tests de eventos y estado derivado pasan sin hardware.

### Fase 2 — Fichador
**Tareas:** firmware del Arduino según la sección 8; `hardware/serie.py` con modo simulado; feedback (LCD con nombre y hora, buzzer, LEDs); el touch despierta la cámara; el pipeline degrada bien si el Arduino no está; desactivar el registro por permanencia (solo gesto/touch registra, configurable).
**Aceptación:** al hacer el gesto, el LCD muestra el nombre y suena el buzzer OK; con el Arduino desconectado, el pipeline sigue; sin gesto no se registra nada.

### Fase 3 — Portero y respaldo
**Tareas:** servicio del Portero con API de la sección 9 y JSONL; destino de eventos que envía al Portero (con reintento y cola local si falla); sincronización por pull desde el servidor; caja fuerte con perilla + botón + LCD; endpoint `/health` con estado del reloj.
**Aceptación:** con el servidor apagado se puede registrar entrada/salida por la caja fuerte; al encender el servidor, los eventos aparecen sin duplicarse aunque se corte la sincronización a mitad.

### Fase 4 — Calibración
**Tareas:** múltiples embeddings por persona; FaceNet solo con cara estable; scripts de evaluación (sección 11) con metadatos de luz; curvas FAR/FRR; elegir y documentar umbral.
**Aceptación:** hay un informe reproducible con métricas y un umbral justificado; mejora medida de rendimiento.

### Fase 5 — Estado del lab, tablero e inventario
**Tareas:** torre de estado; tablero HTML simple para tablets (probar en las tablets reales); inventario con etiquetas QR y eventos `retiro`/`devolucion`; vista "quién tiene qué".
**Aceptación:** el tablero muestra datos en vivo desde el Portero con el servidor apagado y es legible en la tablet.

### Fase 6 — Armario, avisos y extras
**Tareas:** armario con servo y apertura tras identificación (con apertura manual de emergencia); tablero de avisos con dueño y vencimiento; reporte de horas (CSV/PDF) en el servidor; (opcional) bot de Telegram para consultar "¿quién tiene X?".
**Aceptación:** una persona puede retirar un equipo, verlo en el tablero y devolverlo, con registro completo.

### Fase 7 — Seguridad y sistema distribuido (investigación)
**Tareas:** experimentos de la sección 12; réplica del registro entre dos Galileo; pruebas de falla (desenchufar nodos, particiones, servidor con datos viejos); medir consistencia y tiempos de sincronización; (opcional) QR rotatorio y gemelo 3D del laboratorio.
**Aceptación:** informe con método, resultados y mitigaciones.

---

## 16. Reglas para el agente

**Siempre**
- Cambios **pequeños e incrementales**; el proyecto debe poder ejecutarse después de cada uno.
- Respetar el código existente: no reescribir el script completo; extraer funciones de a poco.
- Configuración externa; sin rutas ni credenciales hardcodeadas; `pathlib` para rutas.
- Registrar errores con logging; **nunca** `except` que oculte fallos.
- Lógica pura separada del hardware, con tests (`pytest`). Todo lo que toque hardware debe tener un **modo simulado** para desarrollar sin placas.
- En el Portero (Galileo): solo biblioteca estándar y nada que requiera compilar dependencias pesadas.
- Commits chicos con mensajes claros en español. Actualizar `docs/bitacora/` al terminar cada tarea relevante, con lo hecho y los resultados medidos.
- Documentar los puertos de los sensores y los protocolos.

**Nunca**
- Commitear embeddings, fotos, `caras.csv` reales ni datos personales.
- Guardar fotos de personas (ni siquiera temporalmente).
- Registrar presencia por simple permanencia frente a la cámara.
- Agregar una dependencia pesada sin justificarla por escrito.
- Cambiar el modelo de reconocimiento (FaceNet/MediaPipe), borrar funcionalidad existente o enviar datos fuera de la red local **sin preguntar**.
- Presentar como confirmado algo marcado como VERIFICAR.

**Preguntar al humano antes de:** tocar datos personales, cambiar el modelo, modificar el formato de los CSV existentes, definir la política de retención, o cualquier cableado de tensión de red.

---

## 17. Riesgos, preguntas abiertas y cosas a verificar

### Verificar en el hardware (en el laboratorio)
- [ ] Generación de cada Galileo, si enciende, qué imagen de Linux trae, si tiene Python/Node, si mantiene la hora sin corriente.
- [ ] Si cada kit trae el Base Shield compatible con Galileo (y con UNO) y si el selector de voltaje (3,3/5 V) está bien ubicado.
- [ ] Un "blink" en una muestra de los UNO clones de cada variante; driver CH340 si hace falta.
- [ ] Tablets: marca, modelo, sistema operativo, cámara, navegador, si pueden leer/mostrar un stream o una página simple.
- [ ] Servidor: sistema operativo, RAM, CPU/GPU, y **dónde corre hoy el pipeline** (¿tiene cámara y monitor?).

### Preguntas para el equipo
- ¿Cuántas personas hay (o habrá) enroladas?
- ¿Qué registros deben seguir existiendo en papel (becas, convenios, etc.)?
- ¿Cuál es la política de consentimiento y de retención de datos?
- ¿Quién administra altas y bajas de personas?
- ¿Quién enciende y apaga el servidor, y es viable automatizarlo (Wake-on-LAN y apagado remoto)?

### Riesgos
| Riesgo | Mitigación |
|---|---|
| Hora incorrecta en la Galileo sin corriente | `reloj_confiable` + sincronización al arrancar + revisión de reportes |
| Tablets con navegador demasiado viejo | HTML/JS mínimo; fallback a la caja fuerte física |
| Rendimiento insuficiente de FaceNet en CPU | Correr solo con cara estable; evaluar GPU del servidor |
| Falsos aceptados | Calibración, confirmación física, gesto aleatorio |
| Rechazo de las personas por privacidad | Alternativa sin cara, consentimiento, cámara no pasiva |
| Pérdida de datos si falla una Galileo | Réplica del registro (Fase 7), backups en el servidor |
| Alcance excesivo | Trabajar por fases con entregables semanales |

---

## Apéndice — Primer prompt sugerido para el agente

> Lee `CONTEXTO_PROYECTO.md` completo. Trabajaremos solo la **Fase 0**. Antes de modificar nada: (1) revisa el código real de `mediapipe/detector_de_caras.py` y compáralo con la sección 3; lista las diferencias; (2) propón un plan de cambios mínimos para la Fase 0 y espera mi confirmación. No toques datos personales, no cambies modelos ni borres funcionalidad.