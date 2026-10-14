# Fichador LIS — Hardware y Firmware (Arduino UNO + Grove Base Shield)

Este módulo implementa el nodo físico de asistencia e interacción en la entrada del laboratorio (Fase 2 de `proyecto.md`).

---

## 1. Asignación de Puertos en Grove Base Shield v1

| Módulo Grove | Puerto en Base Shield | Tipo de Señal | Función |
|---|---|---|---|
| **Grove LCD RGB Backlight** | **I2C** | I2C (SDA/SCL) | Pantalla 16×2 y color de fondo RGB |
| **Grove Touch Sensor** | **D2** | Entrada Digital | Despierta la cámara / confirma presencia |
| **Grove Button** | **D3** | Entrada Digital | Botón pulsador auxiliar |
| **Grove Buzzer** | **D4** | Salida Digital | Retroalimentación sonora (`ok`, `err`, `warn`) |
| **Grove LED Azul (DIP)** | **D5** | Salida Digital | Indicador de espera / sistema activo |
| **Grove LED Verde (DIP)** | **D6** | Salida Digital | Identificación correcta / ingreso |
| **Grove LED Rojo (DIP)** | **D7** | Salida Digital | Error / acceso no reconocido |

> **Nota sobre el selector de voltaje:** Asegurarse de que el switch del Grove Base Shield esté posicionado en **5V** para Arduino UNO (o **3.3V** si se monta sobre Intel Galileo).

---

## 2. Bibliotecas Requeridas en Arduino IDE
- `Wire.h` (integrada en Arduino core).
- `rgb_lcd.h` (Biblioteca oficial **Grove - LCD RGB Backlight** de Seeed Studio).

---

## 3. Modo Simulado
Si la computadora no tiene el Arduino conectado o no se dispone del cable USB, el sistema activa automáticamente el **Modo Simulado**, renderizando el display LCD y los controles interactivos en la interfaz web (`http://IP:8080/tablero`).
