/*
  Fichador LIS — Firmware para Arduino UNO y Base Shield Grove (v1)
  Laboratorio de Investigación de Software (UTN FRC)

  Protocolo Serie bidireccional a 115200 baudios.
  Campos separados por '|', terminado en '\n'.

  Conexiones en Grove Base Shield:
  - I2C:     Grove LCD RGB Backlight (Seeed Studio rgb_lcd)
  - D2:      Grove Touch Sensor (Sensor táctil capacitivo)
  - D3:      Grove Button (Pulsador físico)
  - D4:      Grove Buzzer (Piezoeléctrico)
  - D5:      Grove LED Azul (DIP)
  - D6:      Grove LED Verde (DIP)
  - D7:      Grove LED Rojo (DIP)
*/

#include <Wire.h>
#include "rgb_lcd.h"

rgb_lcd lcd;

const int PIN_TOUCH = 2;
const int PIN_BOTON = 3;
const int PIN_BUZZER = 4;
const int PIN_LED_AZUL = 5;
const int PIN_LED_VERDE = 6;
const int PIN_LED_ROJO = 7;

int estado_touch_prev = LOW;
int estado_boton_prev = LOW;

String buffer_serial = "";

void setup() {
  Serial.begin(115200);

  pinMode(PIN_TOUCH, INPUT);
  pinMode(PIN_BOTON, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_AZUL, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);

  // Inicializar LCD RGB Grove (I2C)
  lcd.begin(16, 2);
  lcd.setRGB(0, 120, 255); // Azul LIS inicial
  lcd.setCursor(0, 0);
  lcd.print("LIS Control");
  lcd.setCursor(0, 1);
  lcd.print("Iniciando...");

  // Bip inicial
  digitalWrite(PIN_BUZZER, HIGH);
  delay(100);
  digitalWrite(PIN_BUZZER, LOW);
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
    delay(50); // Antirrebote
  }
  estado_touch_prev = touch_actual;

  // 3. Detección de Botón Físico (D3)
  int boton_actual = digitalRead(PIN_BOTON);
  if (boton_actual == HIGH && estado_boton_prev == LOW) {
    Serial.println("BTN");
    delay(50); // Antirrebote
  }
  estado_boton_prev = boton_actual;
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
    if (tipo == "ok") {
      digitalWrite(PIN_BUZZER, HIGH);
      delay(80);
      digitalWrite(PIN_BUZZER, LOW);
      delay(40);
      digitalWrite(PIN_BUZZER, HIGH);
      delay(120);
      digitalWrite(PIN_BUZZER, LOW);
    } else if (tipo == "err") {
      digitalWrite(PIN_BUZZER, HIGH);
      delay(400);
      digitalWrite(PIN_BUZZER, LOW);
    } else if (tipo == "warn") {
      digitalWrite(PIN_BUZZER, HIGH);
      delay(150);
      digitalWrite(PIN_BUZZER, LOW);
    }
  }
  else if (accion == "LED") {
    int sep2 = cmd.indexOf('|', primer_sep + 1);
    if (primer_sep != -1 && sep2 != -1) {
      String color = cmd.substring(primer_sep + 1, sep2);
      int val = cmd.substring(sep2 + 1).toInt();
      color.toLowerCase();
      int estado_pin = (val == 1) ? HIGH : LOW;

      if (color == "azul") digitalWrite(PIN_LED_AZUL, estado_pin);
      else if (color == "verde") digitalWrite(PIN_LED_VERDE, estado_pin);
      else if (color == "rojo") digitalWrite(PIN_LED_ROJO, estado_pin);
    }
  }
}
