# Plan de Implementación: Bloques 1, 2, 3 y 4 (LIS - UTN FRC)
Integración de SQLite nativo, arquitectura de reconocimiento de alto rendimiento,
soporte para pruebas móviles/tablets (DroidCam Wi-Fi/IP), servicio Portero HTTP y Fichador serie.

## Reglas estrictas
1. Mantener todo el código, nombres de funciones, variables, logs y comentarios en **español**.
2. Trabajar de manera modular e incremental: paquete `nucleo/` para lógica pura, `hardware/` para periféricos, `portero/` para el servicio de borde y `servidor/` para sincronización.
3. **Privacidad obligatoria:** Nunca guardar fotos ni recortes de caras en disco (ni siquiera temporalmente) y nunca usar porcentajes de confianza inventados; mostrar siempre la distancia cruda junto al umbral.
4. **Restricción estricta en `portero/`:** Desarrollado exclusivamente con la biblioteca estándar de Python (`http.server`, `json`, `urllib`, `os`, `socket`, `pathlib`, `uuid`) para correr en hardware limitado (Intel Galileo de 32 bits).
5. Documentar la justificación técnica en `docs/decisiones/` y registrar la bitácora en `docs/bitacora/`.

---

## Bloque 1: Configuración, Portabilidad y Base de Datos SQLite (Fase 0 y Fase 1)
- Actualización de `config.toml` con soporte para cámara universal (`indice`, `url_stream`, `backend`, `espejo`, `mostrar_ventana_local`).
- Creación de `nucleo/base_datos.py` con `RepositorioLIS`, SQLite en modo WAL, 4 tablas relacionales (`personas`, `embeddings`, `eventos`, `auditoria_accesos`) y sincronización en espejo con CSVs.

---

## Bloque 2: Reconocimiento Optimizado, Comparación y Control de Asistencia (Fase 1 y Fase 4)
- Extractor FaceNet rápido en `nucleo/reconocimiento.py` sin la doble pasada redundante de MTCNN sobre BlazeFace.
- Matching vectorizado en `nucleo/comparacion.py` con agregación por distancia mínima o promedio normalizado.
- Caché de "Cara Estable" en `nucleo/estabilizador.py` para triplicar los FPS en CPU.
- Alta segura con PIN de administrador, consentimiento informado y ráfaga de 5 muestras.

---

## Bloque 3: Event Sourcing, Destinos y Protocolo Serie (Fase 1 y Fase 2)

### Paso 8: Motor de Eventos, Estado Derivado y Destinos (`nucleo/eventos.py` y `nucleo/destinos.py`)
1. **Esquema oficial de eventos (Sección 6):**
   - Inmutabilidad, UUID v4, campos `id`, `tipo`, `persona_id`, `objeto_id`, `t`, `reloj_confiable`, `origen`, `metodo`, `distancia`, `meta`.
2. **Función `derivar_estado(lista_eventos)`:**
   - Deriva personas `presentes` (manejando reingresos diarios múltiples), `prestamos` de equipos activos, `avisos` vigentes y estado de `ambiente`/`estado_lab`.
   - Soporte para eventos de `correccion` que anulan eventos previos.
3. **Adaptadores de Destino (`nucleo/destinos.py`):**
   - `DestinoSQLite`, `DestinoCSV`, `DestinoPorteroHTTP` (con cola local de reintentos `datos/cola_pendientes.jsonl` offline-first) y `DestinoCompuesto`.

### Paso 9: Protocolo Serie del Fichador y Simulador Físico Web (`hardware/serie.py` y `hardware/fichador/`)
1. **Protocolo serie a 115200 baudios:**
   - Delimitado por `|`, truncando textos a 16 caracteres por línea para display LCD 16x2.
   - Comandos: `MSG`, `RGB`, `BEEP`, `LED`, `PING` -> `PONG`, y eventos `TOUCH`, `BTN`, `SENS`.
2. **Modo Simulado Virtual:**
   - Emulación del estado del display LCD, color de fondo RGB y LEDs para pruebas sin Arduino físico.
3. **Firmware y mapa de pines:**
   - Sketch `hardware/fichador/fichador.ino`, `hardware/fichador/README.md` y `docs/PROTOCOLO_SERIE.md`.

---

## Bloque 4: Servicio del Portero, Streaming Web y Tablero para Celular/Tablet (Fases 3 y 5)

### Paso 10: Servicio del Portero HTTP (`portero/servidor_portero.py`)
1. **API REST en `0.0.0.0:8080` (Biblioteca Estándar):**
   - Persistencia en `datos/portero_eventos.jsonl` (append-only con `os.fsync`) y deduplicación por `id`.
   - Endpoints: `POST /events`, `GET /events?since=<cursor>`, `GET /state`, `GET /health`.
   - `GET /video_feed`: streaming MJPEG del último cuadro procesado con HUD.
   - `POST /camara/configurar`: cambio en caliente de fuente de cámara (Webcam, DroidCam o Simulada).
   - `POST /hardware/accion`: recepción de acciones táctiles y de sensores desde el navegador.

### Paso 11: Sincronización por Pull del Servidor (`servidor/sincronizador.py`)
1. Sincronización periódica por pull desde el Portero hacia SQLite y CSVs, persistiendo el cursor confirmado.
2. Botón interactivo para simular el servidor apagado/encendido y verificar la tolerancia a cortes.

### Paso 12: Interfaz Web Unificada para Tablet y Celular (`tablero/index.html`)
1. **Vista 1: Modo Tablet de Puerta (Fichador + Cámara en Vivo):**
   - Video feed en tiempo real con selector de DroidCam / Webcam.
   - Display LCD 16x2 virtual con color RGB interactivo y buzzer sonoro vía Web Audio API.
   - Botón grande `TOUCH` para despertar la cámara y simulación de gestos `Open_Palm` / `Thumb_Up`.
   - Fichado alternativo por PIN y QR de respaldo.
2. **Vista 2: Modo Tablero de Pared:**
   - Tarjetas de presentes con métricas y distancias crudas.
   - Inventario de equipos ("¿Quién tiene qué?").
   - Pizarrón de avisos con dueño y vencimiento.
   - Historial en vivo de eventos inmutables.
3. **Vista 3: Laboratorio de Pruebas:**
   - Caja fuerte de respaldo con perilla PIN interactiva.
   - Torre de estado con controles de lux, temperatura, ruido y modo de laboratorio.
   - Simulador de servidor encendido/apagado.

### Paso 13: Lanzador Todo-en-Uno y Pruebas Unitarias
1. `iniciar_demo.py`: Detecta la IP Wi-Fi local y coordina Portero, Sincronizador y Pipeline de video.
2. Suite de tests completa con `pytest` en `tests/test_nucleo.py` y `tests/test_eventos_y_portero.py` (14/14 tests aprobados).
