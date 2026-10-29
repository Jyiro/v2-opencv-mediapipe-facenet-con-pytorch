---
name: hardware-fichador-lis
description: >-
  Guía completa y runbook de hardware para el Fichador de Control de Acceso LIS (Arduino UNO + Grove Base Shield).
  Usar cuando se trabaje con conexión física, cableado, flasheo de firmware, configuración de Arduino IDE o depuración de comunicación serie PC-Arduino.
---

# Fichador LIS — Hardware, Firmware y Comunicación Serie

Guía de referencia rápida y runbook para el nodo físico del Fichador LIS (Arduino UNO + Grove Base Shield).

---

## 1. Inventario y Mapeo Físico de Pines (Circuito Simplificado)

Para evitar cablerío innecesario, el Fichador utiliza únicamente **3 cables Grove**:

### Placa Base
- **Arduino UNO** (Clon con chip CH340 de Winchiphead).
- **Grove Base Shield v1:** Selector de voltaje **obligatoriamente en 5V** (3.3V es solo para Intel Galileo).

### Módulos Grove
| Módulo Grove | Puerto en Base Shield | Tipo | Función |
| :--- | :--- | :--- | :--- |
| **Grove LCD RGB Backlight** | **I2C** (cualquiera) | I2C (SDA/SCL) | Display 16x2 y color completo de estado en fondo |
| **Grove Touch Sensor** | **D2** | Entrada Digital | **Exclusivo para DESPERTAR** del modo Stand-by (25s) |
| **Grove Buzzer** | **D4** | Salida Digital | Melodías amables 8-bit (`tone()`: Mario coin, Salida, Wake) |

> [!NOTE]
> **Componentes liberados para otros nodos:** El botón físico mecánico (D3) y los LEDs DIP (D5, D6, D7) fueron removidos del Fichador. El color de estado se representa íntegramente en la pantalla LCD RGB y el botón liberado se reserva para la Caja Fuerte (Fase 3) o Armario (Fase 6).

---

## 2. Entorno CachyOS / Arch Linux

1. **Detección del puerto:**
   - Se monta en `/dev/ttyUSB0` (módulo del kernel `ch341`).
2. **Permisos Serie:**
   - Grupo del sistema: `uucp`.
   - Asignación: `sudo usermod -aG uucp $USER`.
   - Si Arduino IDE fue abierto desde el entorno de escritorio antes de reiniciar sesión:
     `sudo chmod a+rw /dev/ttyUSB0`
3. **Exclusividad del puerto:**
   - El puerto solo puede ser abierto por un proceso a la vez. **Cerrar siempre el Serial Monitor de Arduino IDE** antes de lanzar scripts de Python (`./iniciar.sh`).
4. **Shell Fish:**
   - El script `activate` estándar falla en Fish shell con error sintáctico. Usar `source ~/.venv-ml/bin/activate.fish` o el wrapper ejecutable `./iniciar.sh`.

---

## 3. Flasheo con Arduino IDE

- **Sketch:** `hardware/fichador/fichador.ino`
- **Placa:** `Arduino Uno`
- **Puerto:** `/dev/ttyUSB0`
- **Biblioteca requerida:** `Grove - LCD RGB Backlight` (de Seeed Studio, instalable desde Library Manager).
- **Configuración de Serial Monitor:**
  - Velocidad: **`115200 baud`**
  - Fin de línea: **`Newline`** o **`Both NL & CR`**

---

## 4. Protocolo Serie y Máquina de Estados (115200 8N1)

- **Comandos PC → Arduino:**
  - `MSG|linea1|linea2` (trunca a 16 caracteres por línea).
  - `RGB|r|g|b` (0-255 por canal).
  - `BEEP|in` $\rightarrow$ Monedita de Mario Bros (Si5 988 Hz $\rightarrow$ Mi6 1319 Hz).
  - `BEEP|out` $\rightarrow$ Tono grave descendente (Sol4 392 Hz $\rightarrow$ Do4 262 Hz).
  - `BEEP|wake` $\rightarrow$ Chirp suave de bienvenida (Do5 523 Hz $\rightarrow$ Mi5 659 Hz).
  - `BEEP|err` $\rightarrow$ Basso grave sutil (220 Hz $\rightarrow$ 165 Hz).
  - `PING` $\rightarrow$ Responde `PONG`
- **Eventos Arduino → PC:**
  - `TOUCH` (contacto capacitivo en D2 $\rightarrow$ despierta el pipeline por 25 segundos).
- **Comportamiento DTR:** La apertura del puerto serie en Linux provoca el auto-reset del Arduino UNO. Python espera ~1.7s antes de enviar tramas para permitir el arranque del bootloader.
- **Roles separados:** El sensor Touch solo despierta el sistema; el fichado se realiza **únicamente** con gestos frente a la cámara (Palma abierta = Entrada, Pulgar arriba = Salida).
- **Respaldo:** En caso de no usar reconocimiento facial, se utiliza la Terminal PIN en la tablet de pared (el QR queda descartado).

---

## 5. Matriz de Diagnóstico Rápido

| Síntoma | Causa Raíz | Solución |
| :--- | :--- | :--- |
| Rombos `\ufffd\ufffd\ufffd` en Serial Monitor | Baud rate en 9600 en vez de 115200 | Cambiar selector del monitor a `115200 baud`. |
| Comandos `MSG`/`RGB` no hacen nada | Monitor configurado como *No line ending* | Cambiar selector a `Newline`. |
| LCD prende luz pero no muestra letras | Switch del Base Shield en 3.3V | Mover switch a **5V**. |
| `Permission denied` en `/dev/ttyUSB0` | Falta pertenencia a `uucp` o sesión vieja | Correr `sudo chmod a+rw /dev/ttyUSB0`. |
| `Device or resource busy` en Python | Serial Monitor de Arduino IDE abierto | Cerrar la pestaña de Serial Monitor en el IDE. |
