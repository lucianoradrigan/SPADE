# Patch 2: retiro del Módulo C, alcance final del módulo de vibración

Este documento modifica `docs/addendum_vibracion_v1.md`. Léase junto a ese archivo — aquí solo se
documentan los cambios surgidos de la investigación empírica hecha durante la implementación
(Macro-fase A.3).

## 1. Hallazgo que motiva este patch

Se probó la premisa central del Módulo C (predecir el residuo de vibración a partir de variables
eléctricas/mecánicas) con la métrica correcta — coherencia — sobre datos reales de rodamientos
sanos de Paderborn:

| Variable condicionante | Coherencia media (banda relevante) | Coherencia máxima | Conclusión |
|---|---|---|---|
| Torque | 0.022 | 0.201 | Sin señal |
| Velocidad (promedio 12 corridas) | 0.031 | 0.148 | Sin señal |
| Fuerza radial (promedio 12 corridas) | 0.044 | 0.227 | Sin señal |

No hay modulación dinámica rápida de la vibración explicable por ninguna de las variables
disponibles en `SCMLSystem`. Se probaron 3 técnicas de calibración/corrección distintas (ajuste de
magnitud PSD, FRF con fase, corrección ML condicionada) — las tres fallan por la misma razón de
fondo, no por defectos de implementación.

Adicionalmente, se probó una hipótesis de nivel (no dinámica): si el nivel (RMS de envolvente) de
vibración escala con el punto de operación entre las 4 condiciones de Paderborn.

| Condición | RMS envolvente | Observación |
|---|---|---|
| 900 rpm, 0.7 Nm, 1000 N | 0.342 | |
| 1500 rpm, 0.1 Nm, 1000 N | 0.415 | |
| 1500 rpm, 0.7 Nm, 400 N | 0.397 | |
| 1500 rpm, 0.7 Nm, 1000 N | 0.411 | |

r(velocidad) = 0.973 — pero con solo 2 niveles reales de velocidad (3 de las 4 condiciones
comparten 1500 rpm), esto es una comparación de 2 grupos, no una correlación validable. Se decidió
no implementar una ley de escalado de amplitud basada en esto: el tamaño de la muestra no sostiene
ajustar ninguna forma funcional, y el efecto (~20% de nivel entre 900 y 1500 rpm) es menor
comparado con el salto de energía en banda que produce la inyección de falla (que sí está
validado, ver A.4/A.5 del roadmap original).

## 2. Decisión

- Se retira el Módulo C (corrector residual data-driven) de la arquitectura. No se reemplaza por
  ninguna variante condicionada en torque, velocidad o carga — ninguna tiene sustento empírico en
  este dataset.
- El Módulo B se mantiene como único componente del módulo de vibración, con el alcance que
  realmente está validado: dar contenido espectral plausible (energía correcta en las bandas
  BPFO/BPFI/BSF/FTF cuando hay falla) para modelos que consumen features de ventana — no
  reproducir forma de onda real muestra a muestra.
- **Limitación documentada, no resuelta:** la amplitud del ruido de fondo del Módulo B es
  constante respecto al punto de operación (no escala con velocidad/carga). Es una simplificación
  conocida, no un descuido. Revisar solo si en el futuro se incorpora KAIST_speed (de los 6
  datasets ya usados por `paper_federative`) como fuente para probar esto con un rango de
  velocidad continuo — no forzar más conclusiones de Paderborn para esta pregunta puntual.

## 3. Cambios concretos sobre lo ya especificado

Respecto a `docs/addendum_vibracion_v1.md` (aplicados directamente en ese archivo):

- **Sección 1** (Principio de diseño): el ajuste de framework de Módulo C a Keras queda sin efecto
  — el módulo no se implementa.
- **Sección 2** (arquitectura, capa Vibration): la capa pasa a ser solo Módulo B.
  `VibrationSynthesizer` (Sección 4 del Addendum) se simplifica: ya no combina `vib_b + vib_c`,
  retorna directamente la salida de `module_b.step(...)`.
- **Sección 5** (mapeo módulo→origen): eliminada la fila `sim/vibration/residual_model.py`.
- **Sección 7** (árbol de repositorio): eliminado `residual_model.py` de `sim/vibration/`.
- **Sección 9** (riesgos): el riesgo "Vibración sintética no suficientemente fiel / Módulo C
  sobreajustado" se reformula — ya no aplica el sobreajuste de C (no existe), pero se mantiene como
  riesgo vigente: "Módulo B no reproduce nivel absoluto de vibración por punto de operación (solo
  contenido espectral relativo). Mitigación: los clasificadores deben entrenarse y evaluarse con
  foco en features de banda/espectrales, no en amplitud absoluta; validar esto explícitamente en
  Macro-fase C.1."

## 4. Criterio de aceptación actualizado para el módulo de vibración

Reemplaza cualquier criterio basado en RMSE de forma de onda o mejora de C sobre B (ya no aplica,
C no existe):

**Separabilidad en espacio de features:** para cada tipo de falla simulada, la separación (p. ej.
distancia entre distribuciones, o AUC de un clasificador simple de referencia) entre "normal" y
"con falla" usando features de ventana (energía de banda, envolvente RMS) sobre la vibración
sintética del Módulo B debe ser del mismo orden que la separación equivalente calculada sobre
datos reales de Paderborn para el mismo tipo de falla.

Esto ya tiene evidencia a favor desde A.4/A.5 (energía sube en la banda correcta ante falla
inyectada) — formalizado como métrica numérica en `experiments/validate_separability.py`.

**Resultado de la primera corrida** (AUC vía Mann-Whitney U, features = energía de banda sobre la
**envolvente** — Hilbert, no espectro crudo; ver nota metodológica más abajo):

| Tipo de falla | AUC real (Paderborn, KA\*/KI\*) | AUC sintético @ `severity=8.0` (default de las demos) |
|---|---|---|
| outer_race | 0.712 | 1.000 (saturado) |
| inner_race | 0.518 | 1.000 (saturado) |

Con `severity=8.0` (el valor usado en `generate_first_dataset.py` / `healthy_and_faulted_grid`) la
separabilidad sintética satura a 1.0 para ambos tipos — muy por encima de la real. Búsqueda en
grilla de la severidad que mejor iguala cada AUC real:

| Tipo de falla | Severidad calibrada | AUC sintético con esa severidad | AUC real |
|---|---|---|---|
| outer_race | 0.05 | 0.766 | 0.712 |
| inner_race | 0.02 | 0.512 | 0.518 |

**Con severidad calibrada por tipo de falla, el criterio SÍ se cumple** — pero no con una
severidad fija universal, y `severity=8.0` (usado en todas las demos hasta ahora) es ~150-1500x
más alto que el valor que realmente iguala la dificultad de detección real.

**Hallazgo adicional, no resuelto:** la severidad calibrada difiere por tipo de falla (0.05 vs.
0.02) porque en datos reales `inner_race` es notoriamente más difícil de detectar que
`outer_race` (modulación por zona de carga, camino de transmisión más largo — fenómeno conocido en
diagnóstico de rodamientos). El Módulo B no tiene esa asimetría incorporada: la única forma de
replicarla hoy es fijar `severity` distinto por tipo de falla al construir el escenario, no es
algo que el modelo capture automáticamente. Documentado como limitación conocida.

**Nota metodológica:** la primera versión de este script usaba energía de banda sobre el
**espectro crudo**, no la envolvente, y daba AUC real invertido (<0.5) para ambos tipos de falla
— no porque no hubiera separabilidad, sino porque un defecto de rodamiento típicamente modula la
amplitud de una resonancia de alta frecuencia en vez de aparecer como pico limpio en el espectro
crudo de baja frecuencia. El análisis de envolvente (demodulación Hilbert) es la técnica estándar
para esto — la misma razón por la que `paper_federative` predice envolvente RMS
(`docs/propuesta_consolidacion.pdf` Sec. 2.4), no forma de onda cruda.

## 5. Actualización de próximos pasos

Agregado a la lista de `INSTRUCTIONS.md` / Sección 10 del Addendum:

- Eliminar código y referencias a `residual_model.py` si ya se había empezado a implementar. —
  **Hecho**: archivo eliminado, sin referencias activas en `sim/vibration/__init__.py` ni en
  `datagen/runner.py`.
- Formalizar la métrica de separabilidad de features (Sección 4 de este patch) como parte del
  criterio de cierre de A.5, antes de dar la Macro-fase A por terminada. — Ver
  `experiments/validate_separability.py` y su resultado.

## Ver también

- `docs/patch3_mejora_modulo_B.md` — desacoplar ruido de fondo y excitación de falla en Módulo B
  (investigación de seguimiento: ¿el torque tiene sustento como excitador del Módulo B, aunque no
  como corrector residual? También negativo, con el mismo método de coherencia aplicado a las
  frecuencias de falla específicas).
- `docs/patch4_modulacion_zona_carga.md` — explica *por qué* `inner_race` calibra a una severidad
  distinta de `outer_race` (Sección 4 de este documento) en vez de solo compensarlo numéricamente.
