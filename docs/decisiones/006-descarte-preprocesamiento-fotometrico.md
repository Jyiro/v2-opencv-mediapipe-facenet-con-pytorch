# Registro de Decisión Arquitectural (ADR) 006: Descarte del Preprocesamiento Fotométrico Permanente en Producción y Calibración Anti-FAR

- **Fecha:** 2026-10-25
- **Estado:** Aceptado
- **Contexto:** Plataforma de herramientas internas del LIS (UTN FRC) — Módulo de Reconocimiento Facial (`nucleo/iluminacion.py`, `nucleo/reconocimiento.py`, `nucleo/comparacion.py`, `nucleo/pose.py`)
- **Fases del Proyecto:** Fase 1, Fase 4 y Trabajo de Investigación / Beca GAALA

---

## 1. Contexto y Objetivo del Experimento

Con el objetivo de mejorar la invarianza del sistema de reconocimiento facial frente a condiciones adversas de iluminación en el acceso al laboratorio (sombras laterales, contraluz o baja luminancia), se diseñó e implementó un **Laboratorio Comparativo A/B en Tiempo Real**.

Se evaluaron dos ramas de inferencia en paralelo sobre recortes faciales de $160 \times 160$ píxeles ingresando al modelo `InceptionResnetV1` (preentrenado con `vggface2`):
1. **Rama A (Cruda / Baseline):** Recorte directo de cámara estandarizado únicamente a $[-1, 1]$.
2. **Rama B (Filtrada / `vggface2+norm`):** Pipeline secuencial compuesto por:
   * Balance de blancos *Gray-World* ($\mu_R = \mu_G = \mu_B$).
   * Ecualización adaptativa de histograma limitada por contraste (**CLAHE**, `clip=2.0`, grilla $8\times8$) sobre el canal $L$ del espacio CIE $L^*a^*b^*$.
   * Corrección **Gamma Adaptativa** dinámica apuntando a un brillo medio objetivo de $0.50$.

---

## 2. Resultados Empíricos Obtenidos

Durante las pruebas controladas en vivo ($N = 193$ inferencias apareadas bajo iluminación interior estándar con asimetría lateral leve de $0.15$), se registraron las siguientes métricas:

| Métrica Evaluada | Rama A: Crudo (`vggface2`) | Rama B: Filtrado (`vggface2+norm`) | Diferencia ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Distancia Euclídea Instantánea (Iluminación Normal)** | **0.420** | 0.480 | $+0.060$ *(Peor en Filtrado)* |
| **Mejor Distancia de la Sesión (Mínima)** | **0.357** | 0.365 | $+0.008$ *(Peor en Filtrado)* |
| **Distancia Media de la Sesión** | **0.532** | 0.534 | $+0.002$ *(Neutro / Leve degradación)* |
| **Aciertos Verdaderos / Falsos Rechazos** | 180 / 13 (93.3%) | 184 / 9 (95.3%) | $+2.0\%$ *(4 cuadros rescatados)* |
| **Carrera a Primer Match (Velocidad de Detección)** | 0 victorias | 0 victorias | 1 Empate |

---

## 3. Análisis Técnico de las Causas

El análisis visual y matemático de los tensores intermedios explica por qué la normalización fotométrica agresiva no mejora —e incluso degrada levemente— la calidad de los embeddings en condiciones operativas normales:

1. **Violación de la hipótesis *Gray-World* en recortes cerrados (ROI facial):**
   El algoritmo *Gray-World* asume que el promedio espacial de reflectancias de una escena completa es acromático (gris neutro). Sin embargo, en un recorte facial cerrado ($160 \times 160\text{ px}$), más del 75% de los píxeles corresponden a piel humana, la cual posee una dominante cálida natural ($R > G > B$). Al forzar el promedio de canales a la igualdad, el filtro introduce un **desplazamiento cromático artificial hacia el cian/verdoso**, alejando el tensor de entrada de la distribución estadística natural con la que fue entrenada la red `InceptionResnetV1`.
2. **Sobre-exaltación de ruido de alta frecuencia por CLAHE:**
   La aplicación de CLAHE sobre el canal de luminancia incrementa abruptamente el contraste local en micro-texturas (poros, vello facial y ruido propio del sensor CMOS). En redes convolucionales profundas, estas alteraciones no lineales de gradiente local modifican las activaciones de las primeras capas convolucionales, incrementando la distancia euclídea intra-clase (de $0.42$ a $0.48$) y reduciendo el margen de seguridad frente a falsos positivos (FAR).
3. **Relación Costo / Beneficio desfavorable:**
   Una mejora marginal del $2.0\%$ en casos de borde (ángulos de transición), a costa de empeorar sistemáticamente la distancia base en condiciones normales y requerir el mantenimiento de una doble base de vectores en SQLite (`vggface2` y `vggface2+norm`), no justifica su activación permanente.

---

## 4. Decisión Tomada

1. **Desactivar el preprocesamiento fotométrico en el flujo de fichado oficial:** Se establece `usar_en_reconocimiento = false` por defecto en `config.toml`, operando exclusivamente con el recorte crudo (`vggface2`) que demostró menor distancia euclídea y mayor fidelidad biométrica.
2. **Conservar el módulo `nucleo/iluminacion.py` y la pestaña `05 Laboratorio A/B` únicamente como instrumental de diagnóstico:** El banco de pruebas permanece disponible en la consola para experimentación docente y auditoría de cámaras, sin penalizar el consumo de CPU ni la latencia durante la operación normal del laboratorio.
3. **Mitigar errores de identificación por la vía geométrica y estadística:** En lugar de alterar los píxeles mediante filtros de imagen, la precisión del reconocimiento y la mitigación de Falsa Aceptación (FAR) se aseguran mediante:
   * **Calibración del Umbral de Decisión:** Ajuste de `umbral_reconocimiento = 0.64` (en lugar de `0.80`), equilibrando la dispersión intra-clase observada en cámara ($0.45 - 0.58$) y la separación inter-clase ($> 0.68$).
   * **Modo de Comparación Robusto (`top2`):** Sustitución del modo `"minimo"` por el promedio ponderado de las 2 muestras más afines (`top2`) con un 20% de peso al centroide global, cancelando la influencia de poses ruidosas aisladas.
   * **Margen de Separación (Ratio Test de Lowe):** Verificación de ambigüedad multiclase ($\Delta d_{\min} \ge 0.05$) antes de otorgar un reconocimiento, evitando cruces entre perfiles parecidos.
   * **Restricción Angular en Alta Guiada (`nucleo/pose.py`):** Límite máximo de guiñada (*yaw*) a $\pm 38^\circ$ (reducido de $48^\circ$) para garantizar amplia variabilidad geométrica sin perder la visibilidad binocular completa.
   * **Ajuste de Margen de Recorte Facial:** Reducción de `pad_crop_cara` al $2\%$ (reducido de $8\%$) para concentrar el tensor $160 \times 160$ en rasgos fisonómicos puros y descartar ruido de fondo y cabello.
