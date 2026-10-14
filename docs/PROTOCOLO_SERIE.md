# Protocolo de Comunicación Serie — PC ↔ Arduino Fichador

Documentación oficial del protocolo serie implementado para el sistema LIS (Sección 8 de `proyecto.md`).

---

## 1. Parámetros de Comunicación
- **Velocidad:** `115200` baudios.
- **Formato:** 8 bits de datos, sin paridad, 1 bit de parada (`8N1`).
- **Codificación:** ASCII / UTF-8 básico sin caracteres especiales.
- **Delimitadores:** Campos separados por pleca (`|`), final de trama con salto de línea (`\n`).

---

## 2. Mensajes PC → Arduino

| Comando | Formato | Descripción | Ejemplo |
|---|---|---|---|
| **Mensaje LCD** | `MSG\|linea1\|linea2` | Muestra hasta 16 caracteres por línea en el display LCD 16×2. Textos más largos se truncan automáticamente. | `MSG\|Juan Perez\|Ingreso OK` |
| **Color LCD** | `RGB\|r\|g\|b` | Modifica el color de fondo RGB del display (0 a 255 por canal). | `RGB\|0\|255\|0` (Verde) |
| **Buzzer** | `BEEP\|patron` | Dispara un patrón de sonido preprogramado: `ok` (doble tono corto), `err` (tono largo grave), `warn` (tono medio). | `BEEP\|ok` |
| **LED** | `LED\|color\|estado` | Enciende (`1`) o apaga (`0`) los LEDs DIP: `azul`, `verde` o `rojo`. | `LED\|verde\|1` |
| **Ping** | `PING` | Comprueba disponibilidad de la placa. | `PING` |

---

## 3. Mensajes Arduino → PC

| Mensaje | Significado | Acción en Pipeline |
|---|---|---|
| `TOUCH` | Se tocó el sensor táctil capacitivo (D2). | Despierta la cámara si estaba en reposo / confirma presencia. |
| `BTN` | Se presionó el botón físico (D3). | Acción complementaria o confirmación. |
| `PONG` | Respuesta al comando `PING`. | Confirma enlace serie activo y sincronizado. |
| `SENS\|temp=..\|luz=..\|sonido=..` | Lectura periódica de sensores de la Torre de Estado (opcional). | Alimenta los metadatos ambientales (`meta.lux`, etc.). |
