# INSTRUCTIONS.md (v2 — consolidado)

**Nota de versión.** Este documento reemplaza la versión original de `INSTRUCTIONS.md` (conservada
como `INSTRUCTIONS_v1_original.md` por trazabilidad) y consolida el contenido decisorio de los
Patches 1 a 7. Los patches y el documento `macro_fase_B2_dpc_deployment.md` se conservan en
`driveflow/docs/` como registro histórico de la investigación que fundamenta cada decisión, pero
no es necesario leerlos para conocer el estado vigente del proyecto: este documento es
autocontenido. La Sección 9 indica dónde consultar el detalle histórico si se requiere
justificación extendida de alguna decisión.

## 1. Principios de diseño vigentes

- **Un solo framework de ML: Keras/TensorFlow 3.** No se emplea PyTorch ni sklearn/XGBoost como
  modelo final.
- **Simulación desacoplada de `gymnasium.Env`** — se emplea `SCMLSystem` directamente para el
  dominio motor.
- **DPC no es un controlador intercambiable con PI/MPC sobre la misma planta.** Opera sobre un
  dominio físico distinto (convertidor VSC), sin variables de comparación comunes con el dominio
  motor. Principio corregido respecto a la formulación original (Patch 5).
- **Esquema de datos común en Parquet**, extendido con columnas de vibración sintética
  (`acc_x/y/z`, `vibration_source`) y con columnas del dominio VSC (`v_ref_real/imag`,
  `vc_real/imag`, `i_f_real/imag`), en NaN cruzado entre dominios.
- **No se suben datasets pesados ni datos protegidos al repositorio** (el dataset KAt-DataCenter de
  Paderborn es CC BY-NC — se descarga aparte, nunca se redistribuye).

## 2. Estado y nomenclatura de fases

| Fase | Contenido | Depende de | Estado |
|---|---|---|---|
| **A** | Generador de datos: motor DC + PI/MPC + Módulo B (vibración) + inyección de fallas | — | Completada |
| **B** | Control DPC portado y validado en lazo cerrado sobre VSC | Independiente de A | Completada — no se persigue el backlog de refinamiento adicional (ver Sección 4) |
| **C** | Clasificadores/regresor de `paper_federative` sobre el dataset de la Fase A | A | Siguiente fase a ejecutar |
| **D.1** | Regresor de degradación de seguimiento (dominio DPC) | B (datos ya generados, sin nueva simulación) | Pendiente, costo bajo |
| **D.2** | Clasificador de fallas de convertidor (degradación de capacitor) | B + veredicto del Paso 0 de separabilidad | Pendiente, condicionada |
| **E** | Dashboard web, integra A, C y D | A, C, D | Pendiente, al final |

La numeración cambió durante la investigación previa a este documento; esta tabla es la referencia
definitiva. No usar la nomenclatura de patches anteriores (macro-fases A/B/C/D de la v1 original)
en comunicación futura.

## 3. Fase A — estado implementado (resumen ejecutivo)

- Motor de simulación de GEM extraído como `SCMLSystem`, sin dependencia de `gymnasium.Env`.
  Controlador PI nativo (no un port de GEM, ver `control/classical/pi_controller.py`). El
  controlador MPC que este párrafo daba por existente ("PI/MPC ya existentes de GEM") en realidad
  nunca se implementó hasta Patch 10 (`control/mpc/` era solo un `.gitkeep`) -- corregido; ver
  `driveflow/docs/patch10_implementacion_mpc.md` para el detalle y por qué no afecta a Patch 5.
- Módulo de vibración (`sim/vibration/`): `background_noise.py` (ruido estocástico calibrado por
  PSD/RMS de rodamientos sanos de Paderborn, sin dependencia de variables eléctricas/mecánicas) +
  `fault_impulses.py` (tren de impulsos en BPFO/BPFI/BSF/FTF, cinemático — depende de velocidad,
  no de torque).
- No existe corrector residual (Módulo C). Se retiró tras confirmar, por coherencia, que torque,
  velocidad y carga son estadísticamente independientes de la vibración real en rodamientos sanos
  (Patch 2). El torque tampoco se usa como excitador en el camino mecánico del Módulo B: el
  control cruzado de frecuencias (BSF más alto que BPFO en un defecto de pista externa) descartó
  un modulador específico por tipo de falla (Patch 3, incluye la verificación de banda de
  armónicos de giro en sanos como su Paso 3b).
- El camino eléctrico (MCSA, `BearingFaultLoad`) y el camino mecánico (Módulo B) usan parámetros
  de amplitud desacoplados, en unidades físicas distintas (hallazgo de Patch 2, Sección 4;
  modulación de amplitud por zona de carga añadida en Patch 4):
  - `torque_ripple_amplitude_nm = 8.0` (eléctrico, validado en corriente).
  - `vibration_fault_amplitude` por tipo de falla: `outer_race = 0.05`, `inner_race = 0.02`. Tipos
    no calibrados deben lanzar excepción explícita, no usar un valor por defecto.
- El filtro modal se calibra exclusivamente contra segmentos sanos (Patch 3, Paso 2).
- Criterio de aceptación del módulo de vibración: separabilidad de features (AUC) entre normal y
  con falla, comparada contra la separabilidad real de Paderborn — no fidelidad de forma de onda
  ni RMSE.

## 4. Fase B — estado implementado (resumen ejecutivo)

- `sim/vsc_system.py` (`VscSystem`, planta discreta Adf/Bdf), `control/dpc/reference.py`
  (`RotatingReference`, 50Hz/50V), `control/dpc/controller.py` (`DpcController`, horizonte
  deslizante).
- Progresión de checkpoints: v1 (solo `Data4train.mat`, RMSE lazo cerrado 16.67V) → v2
  (fine-tuning con augmentación de lazo cerrado real, estilo DAgger, RMSE 4.66V) → v3 (1500 épocas
  adicionales, RMSE 1.18V, desfasaje -0.37°). Checkpoint vigente:
  `dpc_trained_v3_closed_loop.weights.h5`.
- Integrado en `datagen/runner.py` como camino separado (`_run_vsc_scenario`), con validación
  explícita de pares `(controller_type, plant_config_id)` válidos.
- Backlog no perseguido en este ciclo de trabajo (queda documentado, no se ejecuta): ronda
  adicional de augmentación con v3 en vez de v1; exponer checkpoint y R como parámetros de
  `Scenario`. No bloquean ninguna fase posterior.
- **Limitación de alcance (Patch 5):** DPC y PI/MPC no son comparables bajo las mismas
  condiciones — dominios físicos distintos, sin variables comunes. La Fase B no aporta datos a la
  Fase C.

## 5. Fase C — instrucciones de ejecución (fase activa)

- Portar `sensor_dscnn.py`, `gateway_resnet_se.py`, `envelope_forecaster.py` y `windowing.py`
  desde `paper_federative`.
- Entrenar sobre el dataset de la Fase A.
- **Salvaguarda obligatoria, condición de entrada a esta fase (Patch 5, Sección 4):** verificar
  que el pipeline de construcción del conjunto de entrenamiento filtra explícitamente por
  `plant_config_id` (o campo equivalente), excluyendo la totalidad de los registros del dominio
  VSC. Debe existir un test de regresión que verifique esta exclusión **antes de iniciar
  cualquier entrenamiento**, no como verificación posterior.
- Comparar métricas (F1, MAE vs. naive) contra lo ya reportado en `paper_federative` sobre datos
  reales, como prueba de sanidad de que la síntesis de vibración no degrada excesivamente el
  desempeño esperado.
- Alcance C.1: PC, sin restricciones de tamaño de modelo. C.2 (embebido — TFLite/ESP32, benchmark
  Raspberry Pi): diferido, no iniciar sin decisión explícita posterior a la validación de C.1.

## 6. Fase D — instrucciones de ejecución

### D.1 — Regresor de degradación de seguimiento (ejecutar primero: costo bajo, datos ya disponibles)

- **Tarea:** predecir el error de seguimiento futuro (`|v_c − v_ref|`, horizonte corto) a partir
  de una ventana de contexto de `(v_ref, v_c, i_f)`.
- No requiere nueva simulación: se deriva de las trayectorias ya generadas en la Fase B
  (evaluaciones de v1/v2/v3 y los 8 rollouts de augmentación).
- **Arquitectura:** adaptar `build_envelope_forecaster()` de `paper_federative` (contexto pasado →
  magnitud futura), sustituyendo la envolvente de vibración por el error de seguimiento DPC.
- **Entregable:** `models/regressors/dpc_tracking_forecaster.py`, dataset derivado sin nueva
  simulación, evaluación con holdout por rollout completo (no por muestra individual).

### D.2 — Clasificador de fallas de convertidor (ejecutar después de D.1; condicionada a Paso 0)

- Alcance acotado a degradación de capacitor del filtro LC (ESR y capacitancia respecto al valor
  nominal `C = 15µF`). Fallas de conmutación quedan fuera de alcance — `VscSystem` es un modelo
  promediado discreto, sin representación de fallas a nivel de dispositivo de conmutación.
- **Paso 0 (obligatorio, no omitir):** parametrizar `VscSystem` con `C_efectivo` y `ESR_efectivo`
  variables; generar un conjunto reducido de corridas bajo condición nominal y 2-3 niveles de
  degradación; calcular separabilidad (AUC) sobre features de ventana de `(v_c, v_ref, i_f)`. Si
  la separabilidad resulta baja (AUC cercano a 0.5), detener la Fase D.2 en este punto y
  documentar como limitación — no proceder a construir un clasificador sobre una señal no
  separable.
- **Advertencia epistemológica de consignación obligatoria:** no existe dataset real de
  referencia para degradación de capacitores (a diferencia de Paderborn para rodamientos). La
  validación de cualquier clasificador de esta sub-fase es exclusivamente contra la propia
  definición sintética de falla, no contra evidencia empírica independiente. Debe documentarse
  con el mismo nivel de explicitud que las limitaciones ya registradas para el Módulo B.
- **Entregable (condicionado al resultado del Paso 0):** `sim/vsc_faults.py`, dataset etiquetado
  sano/degradado, clasificador (arquitectura adaptada de `sensor_dscnn.py` o
  `gateway_resnet_se.py`), documentación de la limitación epistemológica.

## 7. Fase E — Dashboard web (al final)

Sin cambios respecto a la formulación original: interfaz que integra generación de datos (Fase A),
resultados de diagnóstico (Fase C) y resultados de control/diagnóstico DPC (Fase D). Stack
sugerido: Streamlit. No iniciar hasta que A, C y D estén validadas.

## 8. Regla de precedencia

No se avanza a una fase subsiguiente hasta validar la anterior dentro de la misma línea de
dependencia. La independencia entre B y {A, C} implica que B ya pudo ejecutarse sin esperar a C;
la independencia entre C y D implica que ambas pueden ejecutarse en cualquier orden o en paralelo.
E depende de la validación conjunta de A, C y D.

## 9. Referencia a documentación histórica

| Documento | Contenido decisorio |
|---|---|
| `driveflow/docs/addendum_vibracion_v1.md` | Diseño original del módulo de vibración (Módulo B+C), previo a su revisión |
| `driveflow/docs/patch2_retiro_modulo_C.md` | Retiro del corrector residual por falta de sustento de coherencia; incluye el desacople de severidad eléctrica/mecánica (Sección 4) |
| `driveflow/docs/patch3_mejora_modulo_B.md` | Desacople de ruido de fondo y excitación de falla; retiro del torque como excitador; incluye la verificación de banda de armónicos de giro en sanos (Paso 3b) |
| `driveflow/docs/patch4_modulacion_zona_carga.md` | Modulación de amplitud por zona de carga en `fault_impulses.py` |
| `driveflow/docs/resumen_macro_fase_A.md` | Resumen narrativo completo de las decisiones y experimentos (fallidos y exitosos) de la Fase A |
| `driveflow/docs/macro_fase_B1_dpc.md` | Port a Keras de la red DPC y su loss basada en modelo; entrenamiento completo y métricas de evaluación |
| `driveflow/docs/macro_fase_B2_dpc_deployment.md` | Despliegue y validación en lazo cerrado del controlador DPC (v1→v2→v3) e integración en `runner.py` |
| `driveflow/docs/patch5_alcance_macrofase_B.md` | Corrección de alcance: DPC y PI/MPC no son comparables bajo las mismas condiciones |
| `driveflow/docs/patch7_fase_D_dpc.md` | Definición de la Fase D (D.1, D.2) |
| `driveflow/docs/patch9_correccion_divergencia_dpc.md` | Corrección (no ampliación de fase) de la divergencia DPC en R∈[1.0,3.0]Ω documentada por Patch 8: causa raíz analítica (inestabilidad de planta en lazo abierto, no hueco de datos) y mitigación en dashboard/tests |
| `driveflow/docs/patch10_implementacion_mpc.md` | Implementación de MPC lineal nativo para el motor DC (`control/mpc/`), resolviendo la discrepancia entre "Fase A completada" e INSTRUCTIONS.md mencionando PI/MPC sin que MPC existiera; confirma que Patch 5 no se ve afectado |
| `driveflow/docs/patch11_archivado_modulo_c.md` | Archivado de `experiments/train_module_c*.py` (código histórico NOT RUNNABLE de Módulo C, ver Patch 2) a `experiments/_archive/`, para reducir la confusión "Módulo C" vs. "Fase C" |
| `driveflow/docs/patch12_auditoria_estadistica_modulo_c.md` | Auditoría estadística del retiro de Módulo C (Patch 2): primera ejecución real de las reconstrucciones de coherencia de Patch 8 contra el dataset Paderborn, umbral de significancia formal (no existía antes), reconfirma Patch 2/3 |

**Nota sobre esta tabla respecto al borrador original de este documento:** no existe un
`patch3b_verificacion_sanos.md` ni un `patch4_desacople_severidad.md` como archivos separados —
ese contenido vive dentro de `patch3_mejora_modulo_B.md` (Paso 3b) y `patch2_retiro_modulo_C.md`
(Sección 4) respectivamente. Tampoco se creó `patch6_renumeracion_fases.md`: no hay un documento
de "primera renumeración" real que preceda a este; la Sección 2 de este documento es la única
numeración vigente, no hace falta un patch histórico para algo que nunca existió con otra forma.

## 10. Primer mensaje para Claude Code

Lee este documento (`INSTRUCTIONS.md`) completo antes de escribir código. No es necesario leer los
documentos de la Sección 9 salvo que necesites la justificación extendida de alguna decisión ya
consolidada aquí.

Trabaja únicamente en la Fase C (Sección 5). No toques la Fase D ni la Fase E todavía.

Antes de iniciar cualquier entrenamiento, implementa y verifica la salvaguarda de filtrado por
dominio (punto 3 de la Sección 5) como test de regresión. No avances al portado de arquitecturas
de `paper_federative` hasta que ese test exista y pase.
