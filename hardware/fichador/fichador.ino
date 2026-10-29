/*
  Fichador LIS — Firmware para Arduino UNO y Base Shield Grove (v1)
  Laboratorio de Investigación de Software (UTN FRC)

  Protocolo Serie bidireccional a 115200 baudios.
  Campos separados por '|', terminado en '\n'.

  Conexiones simplificadas en Grove Base Shield:
  - I2C:     Grove LCD RGB Backlight (Texto y retroiluminación en color)
  - D2:      Grove Touch Sensor (Despertar del Standby / Llamar atención)
  - D4:      Grove Buzzer (Melodías amables de 8-bits: Mario Coin, Salida, Wake)
*/

#include <Wire.h>
#include "rgb_lcd.h"

rgb_lcd lcd;

const int PIN_TOUCH = 2;
const int PIN_BUZZER = 4;

int estado_touch_prev = LOW;
String buffer_serial = "";

// Emite una nota con micro-pulsos (~1.5% a 2% de duty cycle)
// para lograr un volumen suave de escritorio sin ser invasivo.
void tono_suave(int frec_hz, int duracion_ms) {
  long periodo_us = 1000000L / frec_hz;
  long pulso_alto_us = constrain(periodo_us / 55, 8L, 20L);
  long pulso_bajo_us = periodo_us - pulso_alto_us;
  long ciclos = ((long)duracion_ms * 1000L) / periodo_us;

  for (long i = 0; i < ciclos; i++) {
    digitalWrite(PIN_BUZZER, HIGH);
    delayMicroseconds(pulso_alto_us);
    digitalWrite(PIN_BUZZER, LOW);
    delayMicroseconds(pulso_bajo_us);
  }
}

void sonar_mario_in() {
  tono_suave(988, 70);   // Si 5
  delay(10);
  tono_suave(1319, 240); // Mi 6 (Moneda de Mario)
}

void sonar_checkout_out() {
  tono_suave(392, 90);   // Sol 4
  delay(10);
  tono_suave(262, 210);  // Do 4 (Descenso cálido)
}

void sonar_wake() {
  tono_suave(523, 30);   // Do 5 (Chirp suave)
  delay(8);
  tono_suave(659, 45);   // Mi 5
}

void sonar_err() {
  tono_suave(220, 100);  // La 3 (Basso / Fallo de caricatura)
  delay(15);
  tono_suave(165, 190);  // Mi 3
}

void sonar_warn() {
  tono_suave(440, 80);
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_TOUCH, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // Inicializar LCD RGB Grove (I2C)
  lcd.begin(16, 2);
  lcd.setRGB(10, 60, 150); // Azul visible (Modo Reposo)
  lcd.setCursor(0, 0);
  lcd.print("UTN FRC - LIS   ");
  lcd.setCursor(0, 1);
  lcd.print("TOQUE EL SENSOR ");

  // Saludo sonoro sutil de arranque
  sonar_wake();
}

void loop() {
  // 1. Lectura de comandos desde PC
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (buffer_serial.length() > 0) {
        procesar_comando(buffer_serial);
        buffer_serial = "";
      }
    } else {
      buffer_serial += c;
    }
  }

  // 2. Detección de pulsación en Sensor Táctil (D2)
  int touch_actual = digitalRead(PIN_TOUCH);
  if (touch_actual == HIGH && estado_touch_prev == LOW) {
    Serial.println("TOUCH");
    delay(60); // Antirrebote
  }
  estado_touch_prev = touch_actual;
}

void procesar_comando(String cmd) {
  cmd.trim();
  int primer_sep = cmd.indexOf('|');
  String accion = (primer_sep == -1) ? cmd : cmd.substring(0, primer_sep);
  accion.toUpperCase();

  if (accion == "PING") {
    Serial.println("PONG");
  }
  else if (accion == "MSG") {
    int segundo_sep = cmd.indexOf('|', primer_sep + 1);
    String l1 = "";
    String l2 = "";

    if (primer_sep != -1) {
      if (segundo_sep != -1) {
        l1 = cmd.substring(primer_sep + 1, segundo_sep);
        l2 = cmd.substring(segundo_sep + 1);
      } else {
        l1 = cmd.substring(primer_sep + 1);
      }
    }

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(l1.substring(0, 16));
    lcd.setCursor(0, 1);
    lcd.print(l2.substring(0, 16));
  }
  else if (accion == "RGB") {
    int sep2 = cmd.indexOf('|', primer_sep + 1);
    int sep3 = cmd.indexOf('|', sep2 + 1);
    if (primer_sep != -1 && sep2 != -1 && sep3 != -1) {
      int r = cmd.substring(primer_sep + 1, sep2).toInt();
      int g = cmd.substring(sep2 + 1, sep3).toInt();
      int b = cmd.substring(sep3 + 1).toInt();
      lcd.setRGB(constrain(r, 0, 255), constrain(g, 0, 255), constrain(b, 0, 255));
    }
  }
  else if (accion == "BEEP") {
    String tipo = cmd.substring(primer_sep + 1);
    tipo.toLowerCase();
    if (tipo == "in" || tipo == "ok") {
      sonar_mario_in();
    } else if (tipo == "out") {
      sonar_checkout_out();
    } else if (tipo == "wake") {
      sonar_wake();
    } else if (tipo == "err") {
      sonar_err();
    } else if (tipo == "warn") {
      sonar_warn();
    }
  }
}
