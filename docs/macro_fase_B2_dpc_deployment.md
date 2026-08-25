# Macro-fase B.2: despliegue en lazo cerrado del DPC entrenado

**Estado:** planta VSC + controlador + generador de referencia construidos; hallazgo inicial (el
holdout estático de B.1 no predecía bien el desempeño en lazo cerrado real) **corregido** vía
fine-tuning con datos de lazo cerrado real (augmentación estilo DAgger), en dos rondas. Checkpoint
final: `configs/dpc_trained_v3_closed_loop.weights.h5`. `controller_type="DPC"` en
`datagen/runner.py` todavía no integrado -- eso queda para el resto de B.2/B.3.

## Qué se construyó

- `sim/vsc_system.py` -- `VscSystem`: la planta discreta (`Adf`/`Bdf`, las mismas matrices de
  `control/dpc/loss.py`) simulada directamente, no una nueva ODE re-derivada. No se construyó
  sobre `SCMLSystem` (no hay motor ni carga mecánica acá -- ver docstring del archivo).
- `control/dpc/reference.py` -- `RotatingReference`: vector de referencia rotando a 50Hz,
  magnitud 50.0V (verificado contra `Data4train.mat`, no asumido de un borrador viejo). El período
  de muestreo (`TAU=1e-4s`) tampoco estaba documentado en ningún lado del toolbox original --se
  infirió empíricamente del propio avance de fase entre las columnas de horizonte de
  `Data4train.mat` (~0.032 rad/paso, consistente en las 10000 filas, std~0), que coincide con
  `2π·50·1e-4` dentro de ~2%.
- `control/dpc/controller.py` -- `DpcController`: despliegue en horizonte deslizante estándar
  (predice los 5 pasos, aplica solo el primero, vuelve a predecir en el siguiente paso real desde
  el estado recién medido).

## Validación: por qué el holdout estático no alcanza

Las filas de `Data4train.mat` son pares (estado, referencia) muestreados de forma i.i.d. -- la red
nunca vio, durante el entrenamiento, que sus propias acciones determinan el siguiente estado que
va a ver. Un holdout con R²≈1.000 dice "para un snapshot dado, la red predice un buen `v_o`"; no
dice nada sobre si aplicar solo el primer paso, repetido cientos de veces reales en lazo cerrado,
es **estable** -- errores podrían acumularse, o el régimen de despliegue (horizonte deslizante)
podría comportarse distinto a la distribución de entrenamiento.

**Chequeo de que el wrapper está bien cableado** (no es un bug de integración): se tomó una fila
real del holdout, se llamó al modelo directamente (como en `evaluate_dpc.py`) y se llamó a
`DpcController.control()` con el mismo estado/referencia reconstruidos -- los `v_o` coinciden
(ej. `(39.18, -62.53)` directo vs. `(39.34, -62.29)` vía el controlador; la pequeña diferencia es
solo por reconstruir la fase inicial vía `atan2` en vez de leer la columna exacta). El wrapper
reproduce fielmente lo que la red predice.

## Resultado del lazo cerrado real (`experiments/evaluate_dpc_closed_loop.py`, 2000 pasos = 200ms)

```
Transitorio (primeros 50 pasos):
  error máximo:            39.05 V

Asentado (pasos 50:2000):
  RMSE:                     16.67 V   (vs. referencia de 50.0 V -- ~33% de error relativo)
  MAE:                      14.55 V
  máximo:                   29.00 V
  no diverge -- el error queda acotado en todo el tramo
```

Diagnóstico adicional (magnitud y fase de `v_c` vs. `v_ref` en el tramo asentado):

```
|v_c| promedio:            46.35 V  (vs. 50.0 V de referencia, ~7% bajo, con oscilación: std=5.1V)
desfasaje v_c respecto a v_ref:  ~17.3°  (con dispersión, std~8.6°)
```

**Interpretación:** el lazo no diverge, pero tampoco converge al seguimiento casi perfecto que
sugería el holdout -- hay un desfasaje de fase persistente (~17°) y una magnitud algo por debajo
de la referencia, con ondulación. Dos causas plausibles, no excluyentes entre sí:

1. **Corrimiento de distribución entrenamiento -> despliegue**: la red se entrenó sobre estados
   muestreados uniformemente al azar, no sobre la distribución de estados que un lazo cerrado real
   efectivamente visita. Es un fenómeno conocido en control predictivo diferenciable entrenado
   offline (no es específico de este port).
2. **Arquitectura sin memoria/integración**: la MLP es puramente feedforward -- no tiene estado
   interno ni término integral. Contra una referencia que rota permanentemente (no un escalón fijo
   que se pueda alcanzar y sostener), un controlador sin acción integral tiene error de régimen
   permanente estructural (el mismo motivo por el que un controlador solo-P, sin I, no anula el
   error de estado estacionario) -- un desfasaje de fase persistente es consistente con eso, no
   necesariamente una falla del port.

Ninguna hipótesis se investigó más a fondo todavía (backlog).

## Corrección: fine-tuning con datos de lazo cerrado (`experiments/finetune_dpc_closed_loop.py`)

Se aplicó la opción elegida (augmentación estilo DAgger, adaptada a la loss de DPC -- no hacen
falta etiquetas de "acción óptima" porque la loss ya es auto-supervisada/basada en modelo, a
diferencia de imitation learning clásico):

1. Se corrió el controlador **actual** (v1, el de B.1) en 8 rollouts de lazo cerrado real (1500
   pasos cada uno, arrancando desde 8 fases iniciales distintas de la referencia -- la única
   fuente de diversidad disponible ya que `R` solo se conoce en un valor real), registrando cada
   estado `(i_f, v_c)` efectivamente visitado junto con su referencia (conocida analíticamente, no
   hace falta medirla). 12000 filas nuevas.
2. Se combinaron con las 6000 filas originales de `Data4train.mat` (18000 filas totales).
3. Se continuó el entrenamiento (fine-tuning desde los pesos v1, no desde cero, `lr=0.0005`) 300
   épocas sobre el set combinado con la misma `dpc_loss` de siempre.

**Resultado -- lazo cerrado (2000 pasos, tramo asentado):**

| | v1 (solo Data4train.mat) | v2 (fine-tuned, +datos de lazo cerrado) |
|---|---|---|
| RMSE tensión | 16.67 V | **4.66 V** |
| MAE tensión | 14.55 V | **3.51 V** |
| desfasaje de fase | ~17.3° (std 8.6°) | **~-3.9° (std 3.9°)** |
| magnitud de v_c | 46.35 V (7% bajo, std 5.1) | **51.38 V (2.8% sobre, std 1.76)** |

~3.6x menos RMSE, ~4.5x menos desfasaje y su dispersión. Confirma que el diagnóstico (corrimiento
de distribución entrenamiento->despliegue) era la causa correcta, no una casualidad.

**Costo -- holdout estático de `Data4train.mat` (se degrada un poco, esperado):**

| paso horizonte | RMSE tensión v1 (V) | RMSE tensión v2 (V) | R² v1 | R² v2 |
|---|---|---|---|---|
| 1 | 0.42 | 1.68 | 1.000 | 0.998 |
| 3 | 0.20 | 0.82 | 1.000 | 0.999 |
| 5 | 0.18 | 0.90 | 1.000 | 0.999 |

Tasa de éxito (error < 2.5V) baja de 100% a 90.6%. Es el trade-off esperado: parte de la
capacidad de la red ahora se reparte hacia los estados de lazo cerrado reales, a costa de un poco
de precisión en la distribución i.i.d. original -- neto positivo, porque lo que hasta ahora se
sabía que importaba (el despliegue real) mejoró mucho más de lo que se perdió en una métrica que
ya sabíamos que no predecía bien el desempeño real.

**Nota:** la loss seguía bajando lentamente al terminar las 300 épocas (30.6 -> 30.3 -> 30.0, sin
aplanarse del todo) -- se continuó entrenando (ver siguiente sección).

## Segunda ronda: 1500 épocas más (`experiments/continue_finetune_dpc.py`)

Se continuó el entrenamiento de v2 sobre el **mismo** set aumentado de 18000 filas (reconstruido
de forma determinística desde los rollouts de v1, no un dataset nuevo), 1500 épocas más,
`lr=0.0005`. La loss esta vez sí se aplanó de verdad (27.30 -> 27.13 -> 27.10 -> 27.02 en las
últimas ~500 épocas, variación mínima) -- checkpoint guardado como
`dpc_trained_v3_closed_loop.weights.h5`.

**Resultado -- las tres versiones, lazo cerrado (tramo asentado) y holdout estático:**

| | v1 (solo datos originales) | v2 (+300 épocas fine-tune) | v3 (+1500 épocas más) |
|---|---|---|---|
| RMSE lazo cerrado | 16.67 V | 4.66 V | **1.18 V** |
| Desfasaje de fase | ~17.3° | ~-3.9° | **~-0.37°** |
| Magnitud de v_c | 46.35 V (7% bajo) | 51.38 V (2.8% sobre) | **50.11 V (0.2% sobre)** |
| RMSE holdout (paso 1) | 0.42 V | 1.68 V | **0.82 V** |
| Tasa de éxito holdout | 100% | 90.6% | **100%** |

v3 es mejor que v2 **en ambas métricas a la vez** -- no es un trade-off más marcado, es que 300
épocas no habían sido suficientes para que la red terminara de ajustar bien tanto la distribución
original como la de lazo cerrado; con más entrenamiento, ajusta las dos. El desfasaje de fase
(~17° -> ~0.4°) queda esencialmente resuelto.

## Integración en `datagen/runner.py`

`controller_type="DPC"` ya está integrado, contra `plant_config_id="vsc_dpc_v1"` -- pero **como un
segundo camino separado**, no como una rama más dentro del loop del motor DC. Motivo: un VSC no
tiene motor, parte mecánica, ni nada para que el Módulo B de vibración enganche -- forzar los dos
sistemas físicos a compartir un mismo cuerpo de loop hubiera sido una abstracción mala. En cambio:

- `Scenario` ahora valida el par `(controller_type, plant_config_id)` contra un set explícito de
  combinaciones válidas: `("PI","dc_perm_ex_v1")` y `("DPC","vsc_dpc_v1")` -- cualquier otra
  combinación (incluido DPC contra el motor DC) levanta `NotImplementedError`, no silenciosamente
  ignora el mismatch.
- `runner.py` despacha por `plant_config_id` a `_run_dc_motor_scenario` (el loop que ya existía) o
  `_run_vsc_scenario` (nuevo). El nuevo usa `dpc_trained_v3_closed_loop.weights.h5` fijo (no es un
  parámetro de `Scenario` todavía -- no hay otro checkpoint razonable para elegir) y `R=8.0064Ω`
  (el único valor con datos reales detrás).
- `seed` en un escenario DPC no siembra ninguna aleatoriedad real (la planta y el controlador son
  determinísticos) -- se reutiliza para elegir la fase inicial de la referencia rotante, así
  seeds distintas igual dan corridas distinguibles y reproducibles.
- El esquema de `export_parquet.py` se extendió con 6 columnas nuevas (`v_ref_real`, `v_ref_imag`,
  `vc_real`, `vc_imag`, `i_f_real`, `i_f_imag`), NaN en registros del motor DC -- mismo patrón que
  `current_s`/`current_t` ya siendo NaN para el motor DC (un solo canal de corriente). Los
  registros DPC, a su vez, dejan en NaN todas las columnas específicas de motor/vibración
  (`rpm`, `torque_nm`, `acc_x/y/z`, `bpfo_hz`...) y `label="normal"` (no existe un modelo de falla
  para este sistema todavía).
- 5 tests nuevos en `tests/test_datagen.py::TestDpcVscScenario` (validación del par, esquema,
  reproducibilidad, fase por seed, y un test de regresión de calidad de seguimiento -- RMSE
  asentado < 5V, con margen generoso sobre el ~1.18V medido para v3, para tolerar ruido entre
  corridas sin que dejen de detectar una regresión real) + 1 test de export mixto DC+VSC. 110/110
  tests totales en verde.

**Limitación honesta sobre el alcance de B (INSTRUCTIONS.md):** la sección B de
`INSTRUCTIONS.md` habla de "comparar DPC vs. PI vs. MPC bajo las mismas condiciones" -- en la
práctica, tal como está el toolbox `DPC4PowerElectronics` original, DPC controla un VSC (filtro
LCL + carga resistiva) y PI/MPC (Macro-fase A/B futura) controlan un motor DC -- son plantas
físicamente distintas, no la misma planta con controladores intercambiables. No se fuerza una
comparación "bajo las mismas condiciones" que no existe en los datos/modelo original; el dataset
ahora sí puede tener corridas DC (PI) y VSC (DPC) exportadas juntas en el mismo esquema, pero
comparar sus métricas de desempeño directamente no sería una comparación válida (miden cosas
distintas: velocidad/torque de un motor vs. tensión/corriente de un convertidor).

## Backlog

- Posible ronda adicional de augmentación (recolectar lazo cerrado con el controlador v3, no v1)
  si en algún momento se quiere exprimir más -- rendimientos marginales ya parecen chicos dado que
  la loss se aplanó.
- Exponer el checkpoint DPC y `R` como parámetros de `Scenario` si en algún momento hay más de un
  checkpoint/condición de carga validados (hoy solo hay uno de cada).
- ~~Verificar la hipótesis del período de muestreo~~ -- **descartada** (ver historial): se probó
  con `Ts=1.0179e-4s` empírico en vez de `1e-4s`, cambio marginal (16.51V -> 16.05V), confirma que
  el corrimiento de distribución era la causa real, no un error de escala temporal.
