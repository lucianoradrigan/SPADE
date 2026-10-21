SPADE: instrucciones para Claude Code, ronda del 2026-10-21

Este documento es autocontenido: no necesitas otros documentos de planificación para ejecutarlo. Las decisiones que contiene las tomé yo, Luciano Radrigan, el 2026-10-21. No me las vuelvas a preguntar. Todo lo que no esté decidido aquí sigue requiriendo mi confirmación (§8).

0. Primer paso obligatorio

Trabaja en el worktree ~/post_doc/plataform_proposed/SPADE-dev (rama dev). Usa su entorno, .venv, con Python 3.12 y la instalación editable de esa carpeta. No uses los entornos de la carpeta SPADE/, que está en main. data/ es un enlace a ../SPADE/data.

```bash
cd ~/post_doc/plataform_proposed/SPADE-dev
git fetch origin && git status -sb      # se espera: dev == origin/dev en acebd43
.venv/bin/python -m pytest -q           # se esperan 1002 aprobadas (≈11 min)
```

Guarda la salida en docs/audit/expansion/round_2026-10-21_baseline.txt. Si no coincide, detente y repórtame la diferencia antes de seguir.

Lee también CLAUDE.md (reglas del repo). Lee docs/design/SPADE_GUI_Complete_Specification.md (spec de la GUI v3.0) solo en las secciones que toque cada tarea.

1. Contexto mínimo

SPADE (paquete driveflow) es una plataforma de simulación, control (PI, MPC, DPC) y diagnóstico con IA para accionamientos eléctricos y electrónica de potencia.

Todo es simulado. No hay datos reales de la máquina ni hardware de borde. Cada resultado se reporta "sobre ventanas simuladas" o "en la grilla de condiciones".

Ramas: main está en a2f2cd3 y no se toca. dev está en acebd43.

Ya hecho en la expansión (en dev; no lo reimplementes):

| Commit | Tarea | Qué dejó |
|---|---|---|
| 07986df | E0 | Runs y jobs durables. runs/schemas.py: PlantProfile, EnvironmentCapability, ExperimentSpec, Run, ApprovalRecord. runs/store.py: SQLite en workspace/, configurable con $SPADE_WORKSPACE; los tests usan un workspace aislado. Worker: python -m driveflow.jobs.worker serve, con un lock por motor. Un tipo de job nuevo se agrega en jobs/handlers.py HANDLERS (fn, validate, title, engine). Página Experiment Runs. Componentes base. |
| 4820e88 | E1 | Entornos SPADE/DcMotor-v0 y SPADE/Vsc-v0: pasan check_env y step() reproduce run_scenario bit a bit. Recompensa en envs/reward.py (términos tracking, overcurrent, action_rate; evaluate() devuelve el total y el desglose). Registro en envs/registry.py. Página Environments. |
| c619a0c | E6 | sim/buck_system.py: Buck promediado síncrono, ZOH exacto, τ = 20 µs. control/classical/buck_pi.py. Scenario(plant_config_id="buck_v1", controller_type="PI"). Entorno SPADE/Buck-v0. Página DC-DC Converters. |
| 0342573 | E10a | ai/condition.py: estados Known / Uncertain / Unknown / Input invalid en AI Diagnostics. |
| 48d6617 | E10b | ai/pinn/identification.py: identificación de L, R_f y C del VSC (PINN en Keras y mínimos cuadrados ZOH). Pestaña Parameter Identification. Reporte en docs/audit/ai/pinn_identification.md. |
| d1c96f6 | E12 | Página Help / Algorithm Guide (configs/algorithm_guide.yaml). |
| acebd43 | E2 | integrations/plecs/ (rpc_client.py, manifest.py, batch.py), perfiles plecs_rpc_batch, barridos como jobs, Scenario(plant_config_id="plecs:<id>", controller_type="circuit"), página PLECS Plant Studio (ruta Dataset). Servidor PLECS falso en tests/integrations/plecs/fake_plecs.py. Nunca se corrió con PLECS real. |

Perfiles nativos: native:dc_perm_ex_v1, native:vsc_dpc_v1, native:pmsm_v1 y native:buck_v1 (en runs/profiles.py).

Modelos en el registro (configs/registry.yaml):

- dc_motor: solo clasificadores. Los de PC y RPi5 son de 5 clases con ventana de 2048; el de ESP32 es de 3 clases. El clasificador de PC promovido es pc_server_5class_w2048/2026-10-20_run01, que no tiene bloque ood.
- vsc_dpc: regresores (pronosticadores) pc_full, rpi5_tcn y esp32_tiny.

Infraestructura para modelos que ya existe:

- arquitecturas configurables en models/regressors/builder.py + YAML (recurrent_type: lstm, gru o none = TCN; input_window; horizon);
- experiments/train_model.py: build_direct_forecast_windows, split por condición (test_groups) y persistence_test_rmse;
- destilación con --distill [--teacher-run];
- experiments/export_tflite.py --run-dir;
- experiments/report_tflite_parity.py --candidate;
- experiments/report_forecaster_condition_split.py.

Datos del motor DC: experiments/generate_diagnosis_grid_dataset.py (≈8 min) genera una grilla de 7 velocidades × 5 severidades, con ruido y deriva de planta, a 10 kHz. Canales: current_r, rpm, torque_nm, voltage_v, acc_x, acc_y, acc_z.

Repos externos que son fuente de algunas tareas:

- Fund. = XinzeLee/Fundamentals_of_AI_for_PE. El código es Apache 2.0; el contenido que no es código (texto, modelos .plecs, datos) es CC BY-NC 4.0. Todo su ML está en PyTorch.
- Sel. = XinzeLee/AI_for_PE_Algorithm_Selector. No tiene licencia: no se copia nada.

Trucos ya conocidos:

- Capturas: uv run --with playwright python <script>. Carga primero la raíz de la app y espera unos 8 s antes de navegar; usa un viewport de unos 3000 px de alto.
- Trabajo en paralelo: git worktree add ../SPADE-eN -b feat/… dev y ejecuta con PYTHONPATH=src ../SPADE-dev/.venv/bin/python.
- Con pkill -f/pgrep -f, usa el truco de corchetes ("[f]ake_serve") en un comando aparte; si no, el patrón mata o espera al propio shell.

2. Reglas de esta ronda

Ramas y commits

- Una tarea por rama, con nombre feat/<ID>-<nombre> y creada desde dev.
- pytest -m "not slow" antes de cada commit. Commits pequeños, en inglés, terminados con: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
- Al cerrar cada tarea:
  - suite completa en verde, sin xfails nuevos salvo los ligados a un hallazgo;
  - evidencia en docs/audit/... (.md + .json), con el comando que produjo cada número;
  - log de la suite en docs/audit/expansion/e<N>_branch_pytest.txt;
  - merge fast-forward a dev y push de dev. Nunca push a main.
- Al cerrar cada tarea, escríbeme un resumen de 3 a 6 líneas: qué cambió, la evidencia (rutas), qué quedó pendiente y qué decisión necesito tomar.

Evidencia

- No declares PASS, "verificado" ni "validado" sin haberlo ejecutado. Si algo no se pudo ejecutar, escribe "no ejecutado: <motivo>".
- Los resultados negativos son entregables. Documéntalos como hallazgos; no fuerces la implementación.
- No ajustes umbrales, hiperparámetros ni decisiones de diseño mirando el test o datos OOD. Si ocurrió, decláralo.
- Hallazgos: antes de crear uno, revisa la numeración en docs/AI_TECHNICAL_AUDIT.md (AI-N…) y docs/SIMULATION_STABILITY_REPORT.md (S-…) y usa el siguiente ID libre. Cada hallazgo lleva severidad, evidencia, estado (abierto, corregido, mitigado o documentado) y sus pruebas.

Código y arquitectura

- Capas: view → adapter → dominio. Nada de física ni de entrenamiento dentro de la UI.
- El backend decide las capacidades: ningún botón se habilita por apariencia y ninguna gráfica se muestra sin un Run real.
- Layout fijo: runs/, jobs/, integrations/plecs/, envs/, optim/, control/rl/, ai/pinn/, ai/prognostics/. No crees src/driveflow/experiments/ (se confunde con la carpeta de scripts experiments/).
- No copies código de PyTorch. Reimplementa en Keras/TensorFlow citando el notebook de origen en el docstring.
- Las pruebas byte a byte de datagen y test_scml_system_regression.py no se tocan.
- Modelos:
  - se definen por YAML y se entrenan con train_model.py;
  - los splits se hacen por adquisición o condición, nunca por ventana;
  - nunca se mezclan dominios;
  - los pesos *.h5 se agregan con git add -f;
  - los datasets no se commitean;
  - cada modelo nuevo lleva su audit_findings.json.

Interfaz

- Textos en inglés, temas claro y oscuro.
- Etiquetas de procedencia válidas: SIMULATED, SIMULATED · PLECS, SIMULATED · SURROGATE, EXTERNAL FILE y MEASURED · EXTERNAL DATASET.
- Nunca LIVE, production ready ni hardware verified.
- Toda predicción se muestra como AI SUGGESTION — REQUIRES VERIFICATION.

Archivos externos: van a data/upstream/fund/, no se commitean y su licencia se documenta en DATA.md.

Si una tarea resulta inviable o mucho más grande de lo esperado: detente, explica el motivo y propone cómo dividirla. Sigue con la siguiente tarea solo si no depende de la que falló.

3. Plan de ejecución (en este orden)

| # | Tarea | Rama | Depende de |
|---|---|---|---|
| 1 | T1: registro de hallazgos | feat/E10-findings | — |
| 2 | T2: E14, regresor dc_motor por residuo (prioridad) | feat/E14-residual-forecaster | T1 |
| 3 | T3: E7, Tuning Studio | feat/E7-tuning-studio | — (puede ir en un worktree paralelo a T2) |
| 4 | T4: E5, sustituto y Compare with SPADE | feat/E5-surrogate | — |
| 5 | T5: E8, RL Lab | feat/E8-rl-lab | ApprovalGate (E0 o T3) |
| 6 | T6: SPADE/Pmsm-v0 | feat/E1-pmsm-env | — |
| 7 | T7: E11, RUL de IGBT | feat/E11-igbt-rul | — |
| 8 | T8: E9, DAB (solo la parte sin PLECS) | feat/E9-dab-prep | — |
| 9 | T9: cierre parcial | feat/E13-partial-closure | T1–T8 |

No ejecutar en esta ronda:

- E3 y E4 (FMU de PLECS paso a paso): falta una máquina con PLECS 5.0 y licencia;
- la evidencia real de E2 (barrido de 3×3 con PLECS);
- integrar un detector OOD al clasificador promovido (decisión D6, §7).

4. T1. Registro de hallazgos de E10a y E10b

Regístralos en docs/AI_TECHNICAL_AUDIT.md y en docs/SIMULATION_STABILITY_REPORT.md. Toma las cifras de los reportes de E10a y E10b y cita el comando de origen.

| Hallazgo | Tipo | Severidad | Estado |
|---|---|---|---|
| La PINN estima C sistemáticamente bajo: −9 % de error medio con 5 % de ruido | AI | Media | Abierto |
| El intervalo bootstrap de la PINN no está calibrado: cobertura del 52 % al 78 % con ruido, frente al 90 % nominal | AI | Media | Abierto. Verifica que la GUI diga "coverage not validated"; si no lo dice, corrígela |
| El lazo DPC diverge con C×0,4 | Simulación | Media (relevante para la tarea A4 del TODO principal) | Documentado. Mide en qué factor de C, entre ×0,7 y ×0,4, empieza la divergencia, sobre la grilla de cargas |
| El clasificador promovido no tiene detector OOD, y el detector de A1 detecta solo el 13 % de las ventanas de falla de bola | AI | Alta (extiende AI-N12) | Abierto. Se vincula con T2 |

Agrega una prueba por hallazgo cuando sea posible: un strict xfail con su ID para los defectos abiertos, o una prueba que fije el comportamiento documentado.

5. T2. E14: regresor dc_motor como detector de anomalías por residuo

Caso de uso (aprobado): pronosticar current_r y rpm con un modelo entrenado solo con ventanas sanas. Cuando aparece el rizado de par de una falla, el error de pronóstico crece; ese residuo es el puntaje de anomalía.

Por qué este caso (no lo reabras):

- es un detector que no depende de las clases del clasificador, y el OOD actual detecta solo el 13 % de las ventanas de bola;
- corre en el borde sin necesitar el simulador, a diferencia del agente de simulación;
- la protección por sobrecorriente aporta poco porque el PI ya limita la corriente (S-07);
- un sensor virtual sería circular: el par es ψ·i y la vibración es sintética.

Diseño:

- Objetivos: current_r y rpm, con pronóstico directo multihorizonte (build_direct_forecast_windows). La vibración no es objetivo, porque es aleatoria por semilla.
- Entradas: el pasado de current_r, rpm y, según la variante, voltage_v.
- Entrena dos variantes, con y sin voltage_v, y reporta ambas.
  - El motivo: en lazo cerrado, el PI compensa el rizado moviendo voltage_v, y esa entrada podría ocultar la falla.
  - La variante para la GUI se elige solo con los criterios 1 y 3 (datos sanos). Si eliges mirando el AUROC, decláralo.
- Ventana y horizonte: parte de configs/regressors/pc_full.yaml.
  - El horizonte debe ser menor que medio período de la frecuencia de falla más alta de la grilla. Calcúlala desde sim/vibration y datagen/fault_injection.py para las 7 velocidades y anótala en el reporte.
  - Si decimas la señal, usa un filtro antialias y verifica que las frecuencias de falla queden bajo Nyquist.
  - Nada de esto se elige mirando ventanas con falla.
- Puntaje: RMSE del residuo por ventana, normalizado por canal con estadísticas de validación sana.
  - Umbral = cuantil 99 del puntaje en validación sana.
  - fit_threshold() debe fallar si recibe filas con falla.

Datos y split:

- Entrenamiento y validación: solo adquisiciones normal del dataset de grilla.
- test_groups por condición: velocidades no vistas (115, 190 y 250 rad/s), severidad ×0,6, ruido del 100 % y deriva de planta.
- Las ventanas con falla se usan solo en la evaluación.

Metas fijas (sobre ventanas simuladas, en condiciones excluidas del entrenamiento):

| # | Criterio | Meta | Si no se cumple |
|---|---|---|---|
| 1 | RMSE frente a la referencia de persistencia, por canal | ≤ 0,8 × persistence_test_rmse | Detente. Documenta el hallazgo |
| 2 | AUROC en outer_race e inner_race, con severidad ≥ ×1 | ≥ 0,9 | Si es < 0,7, el resultado es negativo: se documenta como hallazgo y no se integra a la GUI |
| 3 | Falsas alarmas con ruido del 100 %, deriva y velocidades no vistas | ≤ 1 % de las ventanas sanas | Repórtalo por condición; nunca subas el umbral |
| 4 | Bola, jaula y la falla de orden 1,3 | Sin meta: reporta el AUROC y la tasa de detección con el umbral fijo | — |
| 5 | Arranque y escalones de referencia | Reporta las falsas alarmas por separado | Si son altas, propónme filtrar por régimen como una decisión aparte |
| 6 | Borde (RPi5 y ESP32) | Paridad TFLite frente a Keras, costo en host y 0 operaciones Flex | No se exporta a borde |

Implementación:

- Modelos: YAML configs/regressors/dc_motor_residual_{pc,rpi5,esp32}.yaml, dominio dc_motor. RPi5 y ESP32 se entrenan con --distill --teacher-run <pc>. Para RPi5 usa una TCN (recurrent_type: none), que evita las operaciones Flex.
- Puntaje: ai/residual.py, con fit_threshold(val_healthy) y score(window). Los parámetros se guardan en metrics.json["residual_detector"].
- Evaluación: experiments/report_residual_anomaly.py, que escribe docs/audit/ai/dc_motor_residual_anomaly.{md,json}. Debe incluir:
  - los 6 criterios;
  - las 2 variantes;
  - la frecuencia de falla y el horizonte elegido;
  - la comparación con el agente de simulación en los mismos escenarios.
- Interfaz, solo si se cumplen los criterios 1 y 2:
  - Monitoring: cuarto detector del consenso ("Forecast residual"), con la etiqueta SIMULATED y sin disparar alertas por sí solo.
  - AI Diagnostics: panel "Forecast residual", con pronóstico y señal real superpuestos, el puntaje frente al umbral, la etiqueta AI SUGGESTION — REQUIRES VERIFICATION y el texto "behaviour differs from learned healthy dynamics — not a fault class".
  - Model Registry: el bloque regressor de dc_motor.
  - Sin modelo: mostrar Residual detector not available.
- Pruebas:
  - el umbral se calcula solo con datos sanos;
  - el split no comparte grupos entre entrenamiento y prueba;
  - una ventana sana queda bajo el umbral y una con rizado sintético grande queda sobre él;
  - AppTest de los tres estados de la interfaz (sin modelo, sano y anómalo).
- Cierre:
  - audit_findings.json de cada run, con sus límites: solo simulado, transitorios y efecto de voltage_v;
  - actualizar docs/VALIDATION_SUMMARY.md y docs/AI_TECHNICAL_AUDIT.md;
  - correr el smoke test (python scripts/smoke_test.py --launch);
  - no promover los modelos.

6. Tareas T3 a T8

T3. E7: Tuning Studio

Origen: Fund. 1_MHA/ (PSO y multiobjetivo) y 2_Classic_ML/gaussian_process_bayesian_optimization.ipynb.

- Optimizadores: PSO y NSGA-II propios en numpy (optim/pso.py, optim/multiobjective.py) y optimización bayesiana con sklearn.gaussian_process.GaussianProcessRegressor (optim/bayes.py).
  - Pruébalos sobre Sphere, Rastrigin y ZDT1, con semilla.
  - No agregues pyswarms ni pymoo. Si la implementación propia es inviable, detente y repórtamelo.
- Función objetivo (optim/objectives.py): corre run_scenario o un entorno SPADE/… sobre una grilla de sintonía y devuelve métricas: sobreimpulso, RMSE, tiempo de establecimiento y CPU.
  - La grilla de evaluación es distinta y se fija antes de empezar.
  - Si se reusa la de sintonía, la GUI muestra The evaluation reused tuning conditions; results are not an independent generalization estimate.
- Las corridas son jobs de E0.
- GUI (spec §4), página Design & Control / Tuning Studio:
  - pasos: Select system → Search space → Objectives & method → Execute → Review & approve;
  - curva del mejor valor contra número de evaluaciones;
  - frente de Pareto con selección por clic;
  - comparación de la línea base con el candidato en la grilla de evaluación.
- Save candidate no cambia nada. Use as default pasa por ApprovalGate y no está autorizado en esta ronda.
- Primer caso: el escalón pequeño de S-04 (≈15 % de sobreimpulso del PI del motor DC). Si la tarea C2 del TODO principal ya agregó setpoint_weight, inclúyelo en el espacio de búsqueda.
- Reporte: docs/audit/expansion/e7_tuning_s04.{md,json}, también si el candidato no mejora la sintonía actual.

T4. E5: sustituto y "Compare with SPADE"

- Fuente: el Buck nativo (native:buck_v1), etiquetado SIMULATED y nunca SIMULATED · PLECS. Esto valida la mecánica del sustituto, no su fidelidad a PLECS; dilo así en el reporte.
- Acción del Gym sustituto: el ciclo de trabajo (duty). Para los circuitos PLECS, documenta en el manifiesto que el duty se espera expuesto como ModelVar D o como entrada del subsistema.
- Familias (envs/surrogate_env.py): primero identificación lineal ZOH (reutiliza E10b), luego una red Keras pequeña. Reporta ambas.
- Evaluación:
  - split por condición;
  - error por canal y por condición;
  - dominio de validez;
  - advertencia fuera de dominio (Surrogate outside validated operating range…).
- Publicación: SPADE/BuckSurrogate-v0, con la etiqueta SIMULATED · SURROGATE y el error de ajuste visible en Environments.
- Vista "Compare with SPADE" (spec §3.14.4):
  - se habilita para nativo frente a sustituto, con ejes alineados, error por condición y casos fallidos visibles;
  - no afirmes equivalencia si las condiciones no son comparables;
  - la comparación frente a PLECS sigue en estado vacío.
- Reporte: docs/audit/plecs/e5_surrogate_and_comparison.{md,json}.

T5. E8: RL Lab

Origen: Fund. 7_Reinforcement_Learning/RL_buck_control.ipynb (DQN) y DDPG_buck_control.ipynb (DDPG). Están en PyTorch: reimplementa en Keras.

- Verifica si existe ApprovalGate en viz/components/. Si no existe, impleméntala en esta rama como componente compartido, con sus 5 estados.
- Agentes: control/rl/dqn.py y control/rl/ddpg.py, con semilla, checkpoints y como jobs de E0.
  - DQN solo con acciones discretas; si discretizas una acción, guarda la transformación.
- control/rl/policy_controller.py envuelve una política con la interfaz de controlador de PI, MPC y DPC.
- GUI (spec §5): entorno → recompensa (componente RewardEditor; la vista previa solo después de correr un episodio real) → entrenamiento (recompensa por episodio, pérdidas, violaciones y reproducción de episodios ejecutados) → benchmark en la misma grilla.
- Casos:
  - SPADE/Buck-v0 frente al PI del Buck;
  - VSC frente a DPC v3 y v4, en la grilla de experiments/measure_dpc_rmse_by_r_range.py.
- Reproducibilidad: misma semilla, misma curva, dentro de una tolerancia que declares.
- El benchmark se publica aunque RL pierda. Ninguna política se usa como default.
- Reporte: docs/audit/expansion/e8_rl_benchmark.{md,json}.

T6. SPADE/Pmsm-v0

- Entorno Gymnasium para el PMSM, en envs/native.py, con el patrón de E1.
- check_env debe pasar. Con las acciones del FOC/MTPA de control/classical/pmsm_foc.py, step() debe reproducir bit a bit la simulación que ya existe.
- No cambies la física ni el solver del PMSM: Euler con 3 subpasos, que es la corrección de S-01.
- Regístralo en envs/registry.py y en la página Environments.

T7. E11: RUL de IGBT con datos de la NASA

Origen: Fund. 9_Case_Studies_PE/IGBT_Maintenance/rul_prediction.ipynb (BiLSTM probabilístico, en PyTorch).

- Primero la licencia y la procedencia del dataset de envejecimiento acelerado de IGBT de la NASA (https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging) y del .mat incluido en Fund.
  - Documéntalas en DATA.md y descarga a data/upstream/fund/igbt/, sin commitear.
  - Si la licencia no permite el uso o no está clara, detente y repórtamelo.
- Dominio nuevo igbt_aging: BiLSTM con salida gaussiana (media y log-varianza, pérdida NLL) en Keras, definido como YAML en configs/regressors/.
- Split por dispositivo o serie, sin ventanas de una misma unidad en entrenamiento y prueba.
- Reporta la cobertura empírica del intervalo del 90 %. Si no se puede evaluar, la GUI dice Coverage not validated.
- GUI (spec §8.4), página Intelligence / Prognostics:
  - etiqueta MEASURED · EXTERNAL DATASET, con la fuente y la licencia;
  - texto This dataset is separate from SPADE's simulated plant diagnostics.
- Entregables: docs/audit/ai/igbt_rul.{md,json} y audit_findings.json. No promuevas el modelo.

T8. E9: DAB, solo la parte que no necesita PLECS

Origen: Fund. 9_Case_Studies_PE/DAB_Design/ y 8_PE_Simulation_Automation/PlecsAutomation/DAB_sample1.plecs.

- Primero la licencia de DAB_sample1.plecs (probablemente CC BY-NC 4.0, porque no es código). Documéntala en DATA.md.
  - El archivo no se commitea: va a data/upstream/fund/.
  - Si la licencia impide el uso, detente y repórtamelo.
- Lo que se hace:
  - perfil y manifiesto del DAB para la ruta de dataset de E2 (ModelVars del notebook: Vin, Vref, P, D1, D2, Ro, declarados a mano);
  - pestaña DAB en DC-DC Converters, con estado vacío hasta tener datos;
  - modelo de modulación TPS en Keras, dominio dab sin mezclar con otros, por la vía de YAML → train_model.py. Solo el código y las pruebas con datos sintéticos de prueba, sin entrenamiento declarado.
- Bloqueado: generar el dataset y entrenar con datos reales del circuito. Escribe "no ejecutado: sin PLECS".
- No copies el notebook de adquisición de PLECS de Fund. Tiene dos defectos:
  - writerow llama get_a_param_set(prelim) sin índice, así que todas las filas llevan los parámetros de la combinación 0;
  - la URL 'http://localhost:1080/RPC2"' tiene una comilla de más.

T9. Cierre parcial

- docs/VALIDATION_SUMMARY.md:
  - sección "Expansion (E0–E14)", con la tabla de pruebas por tarea;
  - qué se corrió con PLECS (nada, salvo que alguien lo haya hecho);
  - los hallazgos de T1 y los nuevos.
- Matriz de la spec §14: cada fila marcada como aprobada, fallida o no ejecutada, con el motivo.
- Capturas de cada pantalla nueva en sus 5 estados (vacío, cargando, error, éxito y resultado negativo), en tema claro y oscuro, commiteadas en docs/screenshots/expansion/.
- Pendientes anotados: medir el contraste de los tokens de color y el camino del heartbeat del worker en Windows.
- No abras el PR de dev a main. Escríbeme el resumen de la ronda y espera.

7. Decisiones tomadas (resumen)

| ID | Decisión |
|---|---|
| D1 | E7 con implementación propia, sin dependencias nuevas; "Use as default" no autorizado |
| D2 | E14 aprobado como residuo de pronóstico, con las metas de §5 |
| D3 | E5 se construye con el Buck nativo (SIMULATED); la acción es el duty |
| D4 | E8 no depende de E7; ninguna política como default |
| D5 | Los hallazgos de E10a y E10b se registran ahora (T1) |
| D6 | No se integra el detector OOD actual al clasificador promovido; se reevalúa con E14 |
| D7 | fmpy autorizada como extra opcional [plecs], solo en la rama de E3; E3 y E4 no empiezan hasta que yo confirme que hay una máquina con PLECS 5.0 y licencia |
| D8 | E9 autorizada solo en la parte que no necesita PLECS, con la licencia como condición previa |
| D9 | E11 autorizada, con la licencia como condición previa; sin promoción |
| D10 | SPADE/Pmsm-v0 autorizado |
| D11 | Cierre parcial sin PR a main |

8. Lo que sigue requiriendo mi confirmación

Si llegas a uno de estos puntos, prepara la evidencia, explica las opciones con sus costos y pregúntame:

- promover cualquier modelo (experiments/promote_run.py), incluidos los de E14, E9 y E11, o hacer rollback del registro;
- cambiar un default: controladores, checkpoint DPC, solver, umbrales de alerta o de abstención, parámetros de planta;
- que el residuo de E14 dispare alertas por sí solo;
- usar como default un candidato de E7 o una política de E8;
- agregar cualquier dependencia (salvo fmpy, y solo en E3);
- modificar o re-medir umbrales de pruebas existentes;
- borrar runs, checkpoints o documentos de auditoría;
- empezar E3 o E4;
- usar un archivo externo cuya licencia no esté clara;
- abrir el PR a main, hacer cualquier push a main o reescribir historia.
