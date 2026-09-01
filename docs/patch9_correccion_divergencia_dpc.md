# Patch 9 — Corrección de la divergencia DPC en R bajo

**Contexto de origen:** revisión crítica del repositorio (fuera del roadmap de fases de
`INSTRUCTIONS.md`), tarea de prioridad Alta #1 de un documento externo de "tareas de corrección
priorizadas". Corrige un hallazgo que Patch 8 (`tests/test_dpc_robustness_grid.py`) había dejado
documentado como regresión aceptada, sin corregir.

## 1. Síntoma

`tests/test_dpc_robustness_grid.py::TestKnownDivergenceAtLowR` ya documentaba (Patch 8) que el
lazo cerrado DPC diverge (estado crece sin límite a NaN en ~2000 pasos) para resistencia de carga
R en el rango [1.0, 3.0]Ω, dentro del propio rango del slider "Load resistance" del dashboard
(1.0–20.0Ω). Reproducido de nuevo aquí, con instalación real (`uv venv` + `pytest`, 207/207 antes
de este patch).

## 2. Causa raíz (nueva, no documentada hasta ahora)

Dos hallazgos, el segundo más fuerte que el primero:

1. **R es constante en todo el dataset de entrenamiento.** `experiments/finetune_dpc_closed_loop.py`
   (línea ~46) lo dice explícitamente: *"R is fixed at the one value we have real data for"*.
   `Data4train.mat` y ambas rondas de fine-tuning (v2, v3) usan R = 8.0064Ω en absolutamente todas
   las filas, en las tres versiones del checkpoint. Aunque R es una entrada explícita de la red
   (`control/dpc/network.py`, posición 6 de 15), nunca tuvo variación de la que aprender un
   comportamiento correcto para otro R.

2. **La planta es inestable en lazo abierto para R bajo, independientemente del controlador.**
   Plegando la realimentación resistiva de carga (`i_load = vc/R`) en la matriz de estado
   (`Adf`/`Bdf`, identificadas en `DPC4PowerElectronics`, ver `control/dpc/loss.py`) y calculando
   su radio espectral en función de R (`sim/vsc_system.py::load_feedback_spectral_radius`, nueva en
   este patch):

   | R (Ω) | radio espectral | estado |
   |---|---|---|
   | 0.5 | 12.16 | inestable |
   | 1.0 | 5.63 | inestable |
   | 2.0 | 2.35 | inestable |
   | 3.0 | 1.25 | inestable |
   | **3.3707** (R\*, cruce exacto) | **1.00** | umbral |
   | 3.5 | 0.93 | estable |
   | 8.0064 (nominal) | 0.79 | estable |
   | 20.0 | 0.80 | estable |

   El umbral analítico R\* ≈ 3.3707Ω coincide con el límite empírico observado. Esto significa que
   el modelo de planta discreto en sí mismo (sin ningún controlador) es inestable por debajo de
   ~3.37Ω, con un polo real que crece de magnitud 1.25 (R=3.0Ω) a 12.16 (R=0.5Ω). Ninguna red
   entrenada de la forma en que v1/v2/v3 lo fueron (horizonte de 5 pasos, R fija durante el
   entrenamiento) puede razonablemente domar un polo de esa magnitud.

   Adicionalmente, el enfoque DAgger usado para v2/v3 (`collect_closed_loop_rows`) tiene un
   problema de arranque no resuelto para esta zona: recolectar datos de lazo cerrado en R∈[1,3]
   requeriría correr el controlador actual ahí, pero ahí mismo diverge a NaN de inmediato — no
   existe un conjunto de "estados visitados por la política actual" bien formado que recolectar
   con el método existente, sin rediseñarlo.

**Conclusión:** no es un hueco de cobertura de datos que un reentrenamiento incremental resuelva
razonablemente — es una propiedad estructural de la planta identificada. Reentrenar para intentar
estabilizar R∈[1,3]Ω se documenta como línea de investigación aparte (no perseguida en este patch),
no como corrección de corto plazo.

## 3. Corrección implementada

Mitigación (acotar y advertir, no ocultar), en línea con la Sección 1 de este documento:

- `sim/vsc_system.py`: nueva función `load_feedback_spectral_radius(r_ohm)` (el mecanismo analítico
  de la Sección 2.2) y nueva constante `MIN_STABLE_LOAD_RESISTANCE_OHM = 4.0` (~19% de margen sobre
  R\*).
- `viz/dashboard.py`: el slider "Load resistance R (Ω)" ahora tiene piso en
  `MIN_STABLE_LOAD_RESISTANCE_OHM` (antes 1.0Ω) en vez del mínimo físico arbitrario anterior. La
  zona inestable sigue siendo alcanzable vía el checkbox "Custom value" (para poder reproducir el
  hallazgo deliberadamente), pero ahora se muestra una advertencia visible explicando la causa
  analítica cuando el R efectivo cae por debajo del piso.
- `tests/test_dpc_robustness_grid.py`: nueva clase `TestOpenLoopStabilityThreshold` que verifica el
  mecanismo analítico en sí (no solo el síntoma) — falla si `MIN_STABLE_LOAD_RESISTANCE_OHM` deja
  de tener margen seguro sobre R\*, o si `Adf`/`Bdf` cambian de forma que el margen ya no se
  sostenga. `TestKnownDivergenceAtLowR` se mantiene (la planta genuinamente sigue divergiendo ahí —
  es física, no un bug a "arreglar"), pero su docstring ya no describe la divergencia como
  aceptada indefinidamente sin mitigación.

## 4. Qué NO se hizo (alcance explícitamente fuera de este patch)

- No se intentó un reentrenamiento (v4) para estabilizar R∈[1,3]Ω. Dada la magnitud del polo
  inestable (hasta 12.16 en R=0.5Ω) y el problema de arranque de recolección de datos descrito en
  la Sección 2, no hay razón para esperar que la misma técnica de fine-tuning tenga éxito; haría
  falta una arquitectura de control distinta (con ganancia dependiente de R) o determinar primero
  si R∈[1,3]Ω corresponde a un régimen de operación físicamente sensato para este convertidor
  (implica una corriente de carga muy alta a tensión nominal, cercana a cortocircuito) antes de
  invertir ese esfuerzo.
## 4bis. Adenda — Tarea 5 (tabla de RMSE por sub-rango de R), generada tras este patch

`experiments/measure_dpc_rmse_by_r_range.py` (nuevo) recorre los 3 checkpoints contra la misma
simulación de lazo cerrado, en la grilla completa del slider del dashboard:

| R (Ω) | v1 (solo Data4train.mat) | v2 (fine-tune lazo cerrado) | v3 (fine-tune extendido) |
|---|---|---|---|
| 1.0 | diverge (NaN) | diverge (NaN) | diverge (NaN) |
| 2.0 | diverge (NaN) | diverge (NaN) | diverge (NaN) |
| 3.0 | diverge (NaN) | diverge (NaN) | diverge (NaN) |
| 3.5 | 27.055 V | 19.527 V | 20.478 V |
| 4.0 | 24.468 V | 17.231 V | 17.959 V |
| 5.0 | 19.777 V | 12.616 V | 13.255 V |
| **8.0064 (nominal)** | **16.668 V** | **4.660 V** | **1.181 V** |
| 10.0 | 25.307 V | 8.747 V | 6.449 V |
| 15.0 | 139.467 V | 16.916 V | 11.960 V |
| 20.0 | 558.631 V | 20.429 V | 14.688 V |

Dos hallazgos adicionales, no anticipados en la Sección 2:

1. **Los tres checkpoints divergen exactamente en el mismo rango** (R∈[1.0,3.0]Ω) -- confirma de
   forma independiente (una corrida real por checkpoint, no solo el análisis de v3) que la
   inestabilidad es de la planta, no de un checkpoint específico.
2. **Cerca del umbral de estabilidad, el RMSE sigue siendo alto en los tres checkpoints, aunque ya
   no diverge.** R=4.0Ω (el piso elegido para `MIN_STABLE_LOAD_RESISTANCE_OHM`) da 17-24V de RMSE
   -- lejos de "buen seguimiento", solo de "no diverge a NaN". Esto es esperable: cerca de un polo
   que cruza el círculo unitario, la dinámica decae muy lentamente incluso del lado estable, así
   que un margen de estabilidad (radio espectral < 1) no implica un margen de *calidad* de
   seguimiento en una ventana de tiempo fija. **`MIN_STABLE_LOAD_RESISTANCE_OHM` sigue siendo
   correcto para lo que dice prevenir (divergencia a NaN) y ningún texto de este patch afirma
   que garantiza buen seguimiento** -- pero se deja consignado explícitamente aquí para que no se
   asuma lo segundo a partir de lo primero.
3. **v1 (sin fine-tuning de lazo cerrado) se degrada severamente a R alto** (558.6V en R=20Ω, vs.
   14.7V de v3) mientras que v2/v3 generalizan razonablemente ahí -- pese a que el fine-tuning de
   v2/v3 nunca varió R durante el entrenamiento (Sección 2). Esto sugiere que el fine-tuning de
   lazo cerrado mejoró la robustez general del controlador (no solo el RMSE en el punto nominal),
   probablemente porque la augmentación DAgger expone al modelo a una variedad más realista de
   trayectorias de estado que la muestra i.i.d. original de `Data4train.mat`, aunque R en sí nunca
   varió. No se investigó más a fondo (fuera de alcance de esta corrección).

## 5. Verificación

`pytest` completo tras el patch: 217/217 (207 previos + 10 nuevos en
`TestOpenLoopStabilityThreshold`/ajustes de `TestKnownDivergenceAtLowR`), instalación real
(`uv venv --python 3.11` + `uv pip install -e ".[dev,viz]"`), no solo mensaje de commit.
