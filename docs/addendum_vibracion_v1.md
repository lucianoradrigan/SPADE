# Addendum: incorporación del módulo de vibración (B+C) a v1 de driveflow

**Este documento modifica y extiende "Propuesta de Consolidación: Plataforma Única de Simulación, Control Predictivo Diferenciable y Diagnóstico ML para Accionamientos Eléctricos" (20-ago-2026).** Léase junto al documento original — aquí solo se detallan los cambios. Las secciones de DPC4PowerElectronics, GEM y paper_federative del documento original no cambian.

**Decisión que motiva este addendum:** el documento original resuelve la ausencia de vibración en GEM difiriéndola a una "Fase 8 (opcional)" y usando solo MCSA (firma en corriente) para v1. Se decidió que **la síntesis real de vibración (x, y, z) forma parte de v1**, no de una extensión futura, usando el módulo B (físico calibrado) + C (residual data-driven) ya especificado previamente para GEM. Este addendum lo adapta a la arquitectura de `driveflow` (que, a diferencia de GEM, no usa la capa `gymnasium.Env`).

---

## 1. Ajuste al Principio de diseño (Sección 4.1 del original)

El documento original fija: *"Un solo framework de ML: TensorFlow/Keras 3, no PyTorch"* (Principio #1). El diseño previo del Módulo C (corrector residual) asumía PyTorch — **se adapta a Keras/TensorFlow 3** para no romper este principio. El Módulo B no se ve afectado: es un modelo físico/numérico (NumPy/SciPy), sin dependencia de framework de ML, coherente con el resto de `sim/`.

> **[Patch 2, ver `docs/patch2_retiro_modulo_C.md`]** Este ajuste queda sin efecto: el Módulo C se retiró de la arquitectura (falta de sustento empírico — coherencia ≈0 entre vibración y cualquier variable eléctrica/mecánica disponible en `SCMLSystem`) y no se implementó. No aplica ningún framework de ML a este módulo.

## 2. Nueva capa en la arquitectura (extiende Sección 4.2)

Se agrega una capa entre **Sim** y **Datagen**:

| Capa | Responsabilidad | Origen |
|---|---|---|
| **1.5 Vibration** *(nueva)* | Sintetiza aceleración `(x, y, z)` a partir de la salida de Sim (`ω, i_d, i_q, T`). Módulo B (físico, calibrado contra datos reales) + Módulo C (residual, Keras, corrige lo que B no captura). | Nuevo, calibrado/entrenado con dataset externo KAt-DataCenter (Paderborn) |

Consume la salida de la capa **Sim** (no de **Control**) y su salida alimenta directamente a **Datagen**, que ahora exporta `acc_x/y/z` con valores reales en vez de `NaN`.

> **[Patch 2]** La capa pasa a ser **solo Módulo B** — no hay Módulo C. `VibrationSynthesizer` (Sección 4) se simplifica: ya no combina `vib_b + vib_c`, retorna directamente la salida de `module_b.step(...)`.

## 3. Reescritura de la Sección 4.3 (puente técnico)

El documento original titula esta sección *"Puente técnico simulación → diagnóstico: inyección de fallas vía firma MCSA"* y la resuelve solo con corriente. Se **combina** con síntesis de vibración — no son alternativas, son complementarias y comparten la misma base:

1. `sim/vibration/bearing_frequencies.py` calcula BPFO/BPFI/BSF/FTF a partir de `ω(t)` y la geometría del rodamiento — **este cálculo ya lo necesitaba `datagen/fault_injection.py` del documento original**; con el addendum se reutiliza como módulo compartido en vez de duplicarse.
2. Esas frecuencias alimentan **dos caminos en paralelo**:
   - **Camino eléctrico (MCSA, como en el original):** ripple de par → `BearingFaultLoad` → acoplamiento electromecánico → bandas laterales en la corriente simulada por GEM.
   - **Camino mecánico (nuevo, Módulo B+C):** las mismas frecuencias excitan un modelo modal masa-resorte-amortiguador → aceleración sintética `x_B, y_B, z_B` → corrección residual por Módulo C → `acc_x/y/z` final.
3. Ambos caminos quedan disponibles en el dataset exportado. Esto es una ventaja frente al documento original: los clasificadores de `paper_federative` (entrenados mayormente sobre vibración real — CWRU, MaFaulDa) pueden entrenarse sobre el canal para el que fueron diseñados, no solo sobre un proxy en corriente.

### Fórmulas (sin cambios respecto a la especificación previa)

```
BPFO = (n/2) · f_r · (1 − (d/D)·cos φ)
BPFI = (n/2) · f_r · (1 + (d/D)·cos φ)
BSF  = (D/2d) · f_r · (1 − (d/D)²·cos²φ)
FTF  = (f_r/2) · (1 − (d/D)·cos φ)
```

Modelo modal por modo `k`:

```
x_k''(t) + 2·ζ_k·ω_n,k·x_k'(t) + ω_n,k²·x_k(t) = F(t) / m_k
```

`F(t)` = tren de impulsos en BPFO/BPFI/BSF/FTF (modulados por `ω(t)` real, no constante) + rizado de par/armónicos electromagnéticos ya disponibles como `T(t)` desde `SCMLSystem`.

## 4. Integración con `SCMLSystem` (reemplaza el diseño anterior basado en `PhysicalSystemWrapper`)

**Importante:** la especificación de vibración original (entregada antes de conocer `driveflow`) proponía un `VibrationWrapper(PhysicalSystemWrapper)`, apoyado en el mecanismo de wrappers de Gym de GEM. **Ese diseño ya no aplica**: el Principio de diseño #2 de `driveflow` desacopla la simulación de `gymnasium.Env` y usa `SCMLSystem` directo. La integración correcta es una clase simple que se llama manualmente dentro de `datagen/runner.py`, no un wrapper de Gym:

```python
# src/driveflow/sim/vibration/vibration_synthesizer.py

class VibrationSynthesizer:
    """Se invoca manualmente tras cada SCMLSystem.simulate(), NO es un
    PhysicalSystemWrapper (driveflow no usa la capa gymnasium.Env de GEM)."""

    def __init__(self, module_b: "ModalVibrationModel", module_c: "ResidualCorrector", tau: float):
        self.module_b = module_b
        self.module_c = module_c
        self.tau = tau

    def step(self, omega: float, i_d: float, i_q: float, torque: float) -> tuple[float, float, float]:
        vib_b = self.module_b.step(omega, i_d, i_q, torque, dt=self.tau)
        vib_c = self.module_c.predict(omega, i_d, i_q, torque, vib_b)
        return tuple(vib_b + vib_c)  # (acc_x, acc_y, acc_z)
```

Uso en el loop de generación de datos:

```python
# src/driveflow/datagen/runner.py  (fragmento)
for k in range(n_steps):
    action = controller.control(state, reference)
    state = scml_system.simulate(action)                       # ya existe en driveflow
    omega, i_d, i_q, torque = extract_states(state, scml_system) # usa OMEGA_IDX/TORQUE_IDX/CURRENTS_IDX
    acc_x, acc_y, acc_z = vibration_synth.step(omega, i_d, i_q, torque)
    record = {**state_to_dict(state), "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z, ...}
```

`extract_states` puede apoyarse en los índices ya definidos por `SCMLSystem` (`OMEGA_IDX`, `TORQUE_IDX`, `CURRENTS_IDX`), evitando lógica ad-hoc por tipo de motor.

## 5. Actualización del mapeo módulo → origen (extiende Sección 4.4)

| Módulo propuesto | Origen | Acción | Detalle |
|---|---|---|---|
| `sim/vibration/bearing_frequencies.py` | Nuevo (compartido con `datagen/fault_injection.py`) | Nuevo | BPFO/BPFI/BSF/FTF — una sola implementación, usada por el camino eléctrico (MCSA) y el mecánico |
| `sim/vibration/modal_model.py` | Nuevo | Nuevo | Módulo B: sistema modal masa-resorte-amortiguador, NumPy/SciPy |
| `sim/vibration/force_synthesis.py` | Nuevo | Nuevo | Síntesis de `F(t)`: impulsos de falla + rizado de par |
| `sim/vibration/calibration.py` | Nuevo | Nuevo | Ajuste de `{ω_n,k, ζ_k, ganancias}` contra PSD real del dataset KAt-DataCenter (Paderborn) |
| `sim/vibration/vibration_synthesizer.py` | Nuevo | Nuevo | Orquesta B+C; se llama en `datagen/runner.py` (ver Sección 4) |

> **[Patch 2]** Eliminada la fila `sim/vibration/residual_model.py` — el Módulo C se retiró, ver `docs/patch2_retiro_modulo_C.md`.

## 6. Actualización del esquema de datos (extiende Sección 4.5)

| Columna | Cambio respecto al original |
|---|---|
| `acc_x`, `acc_y`, `acc_z` | Antes: *"Reservadas para v2; vacías/NaN en v1"*. **Ahora: pobladas en v1** por `VibrationSynthesizer`. |
| `vibration_source` *(nueva)* | `"synthetic_b"` (solo Módulo B) o `"synthetic_b_plus_c"` (B+C) — deja explícito que el canal es sintético, no medido, para no confundir aguas abajo con datos reales de validación. |
| `audio` | Sin cambios: sigue fuera de alcance, `NaN` en v1. |

## 7. Actualización del árbol de repositorio (extiende Sección 4.6)

```
src/driveflow/sim/
├── motors/
├── converters.py
├── loads.py
├── supplies.py
├── solvers.py
├── scml_system.py
├── plant_configs/
│   └── vsc_lc.py
└── vibration/                     # NUEVO
    ├── bearing_frequencies.py
    ├── modal_model.py             # Módulo B
    ├── force_synthesis.py
    ├── calibration.py
    └── vibration_synthesizer.py   # orquesta Módulo B, llamado desde datagen/runner.py
```

> **[Patch 2]** Eliminado `residual_model.py` del árbol — el Módulo C se retiró, ver `docs/patch2_retiro_modulo_C.md`. (`force_synthesis.py` también fue reemplazado más adelante por `background_noise.py` + `fault_impulses.py`, ver `docs/patch3_mejora_modulo_B.md`.)

## 8. Actualización del roadmap (reemplaza Sección 5)

Se inserta una fase nueva (Fase 4) para el módulo de vibración, y se renumeran las fases siguientes. La Fase 4 puede correr **en paralelo a las Fases 1-3**, porque calibrar el Módulo B y entrenar el Módulo C se hace directamente contra tramos reales del dataset de Paderborn (que trae `ω, i_d/i_q, T` reales junto con la vibración medida) — no depende de que GEM/DPC ya estén integrados.

| Fase | Objetivo | Depende de | Notas |
|---|---|---|---|
| 0 | Setup y decisiones | — | Agregar: confirmar Keras también para Módulo C |
| 1 | Extraer motor de simulación (GEM) | Fase 0 | Sin cambios |
| 2 | Planta VSC+LC y controladores | Fase 1 | Sin cambios |
| 3 | Portar DPC y validar | Fase 2 | Sin cambios |
| **4** *(nueva)* | **Módulo de vibración (B+C)**: implementar `bearing_frequencies.py`, `modal_model.py`, `force_synthesis.py`; calibrar Módulo B contra PSD real de KAt-DataCenter; entrenar Módulo C (Keras) sobre el residual | Fase 0 (puede correr en paralelo a Fases 1-3) | Descargar dataset Paderborn (~20.8 GB) antes de empezar |
| 5 *(antes Fase 4)* | Generación de datos + inyección de fallas | Fases 1-4 | Ahora exporta `acc_x/y/z` reales vía `VibrationSynthesizer`, no `NaN` |
| 6 *(antes Fase 5)* | Portar modelos ML y entrenar | Fase 5 (dataset); puede arrancar en paralelo con datos reales mientras Fase 5 madura | Los clasificadores ahora pueden entrenar sobre corriente **y** vibración, más fiel al dominio original de `paper_federative` |
| 7 *(antes Fase 6)* | Pipelines end-to-end + reportes | Fases 5-6 | Sin cambios de fondo |
| 8 *(antes Fase 7)* | Empaquetado, docs, CI, publicación | Fase 7 | Agregar nota de licencia (Sección 9) |
| 9 *(antes Fase 8, opcional)* | Extensiones futuras | Fase 8 | **Ya no incluye vibración** (movida a v1); mantiene: capa Gym/RL opcional, exploración de agregación federada como módulo aparte, posible upgrade de Módulo C a GRU/TCN si el MLP/Keras simple no alcanza |

## 9. Actualización de riesgos (extiende Sección 6)

| Riesgo | Impacto | Mitigación |
|---|---|---|
| **[Patch 2, reformulado]** Módulo B no reproduce nivel absoluto de vibración por punto de operación (solo contenido espectral relativo) | Los clasificadores podrían aprender a depender de la amplitud absoluta en vez de la forma espectral, que no generaliza entre puntos de operación reales | Los clasificadores deben entrenarse y evaluarse con foco en features de banda/espectrales, no en amplitud absoluta; validar esto explícitamente en Macro-fase C.1 |
| Dataset KAt-DataCenter (Paderborn) es CC BY-NC | Mismo riesgo de licencia ya identificado para `paper_federative`, ahora también aplica al módulo de vibración | No se sube el dataset ni pesos que puedan memorizar señales identificables (Principio de diseño #5, ya presente en el original); documentar en el README que la calibración requiere descarga separada bajo licencia propia |
| Desalineación de framework (Módulo C diseñado originalmente en PyTorch) | Rompe el Principio de diseño #1 (framework único) si se implementa tal cual la especificación previa | Portar a Keras/TensorFlow 3 antes de implementar (ver Sección 1) |

## 10. Actualización de "próximos pasos inmediatos" (extiende Sección 7)

Se agregan dos puntos a la lista original del documento:

5. Confirmar que el Módulo C (residual) se implementa en Keras/TensorFlow, no en PyTorch como en el borrador original de la especificación de vibración — coherencia con el Principio de diseño #1.
6. Descargar el dataset KAt-DataCenter de Paderborn (`https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download`, ~20.8 GB) para poder arrancar la Fase 4 (calibración de Módulo B) en paralelo a las Fases 1-3, en vez de esperar a que el resto de `driveflow` esté listo.
