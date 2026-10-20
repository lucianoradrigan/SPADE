# SPADE: lista de tareas para Claude Code

*Generada el 2026-10-20 · Base: commit `5427819` (main = dev) · Autor: Luciano Radrigan*

## Cómo usar este archivo

1. Este archivo vive en la raíz del repo como `TODO_CLAUDE_CODE.md`.
2. Pídele a Claude Code: *"Lee TODO_CLAUDE_CODE.md y ejecuta la tarea V0"*.
3. Haz una tarea por sesión y por rama. Al cerrar una tarea, marca su casilla en este archivo.

> **Restricción de alcance: no hay datos reales.** Ninguna tarea de esta lista necesita mediciones de una máquina física. Las comparaciones contra gym-electric-motor y DPC4PowerElectronics se hacen contra implementaciones de referencia, no contra datos medidos. Validan que el port es fiel al original, no que el modelo represente una máquina real, y así hay que documentarlas.

## Reglas para Claude Code (aplican a todas las tareas)

- **Ramas:** crear una rama desde `dev`, con la forma `feat/<ID>-<nombre-corto>` (por ejemplo `feat/A1-ood-detector`). No hacer push a `main`. El push se hace por SSH (`git@github.com:lucianoradrigan/SPADE.git`).
- **Entorno:** `uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev,viz]"`.
- **Pruebas:**
  - correr `pytest -m "not slow"` antes de cada commit;
  - correr `pytest` completo (≈10 min) antes de dar una tarea por cerrada;
  - cero fallos y ningún xfail nuevo sin un hallazgo asociado.
- **Pruebas existentes:** las de regresión byte a byte no se tocan. Si una cambia, detenerse y explicar por qué.
- **Pesos y datos:**
  - `*.h5` está en `.gitignore`. Un run nuevo se agrega con `git add -f`; si no, el CI falla en la prueba del registro.
  - Los datasets van en `data/` y nunca se commitean.
  - Los archivos descargados de repos externos van en `data/upstream/` y tampoco se commitean.
- **Promoción de modelos:** no promover ningún modelo (`promote_run.py`) ni cambiar un default (checkpoint DPC, solver, umbral) sin confirmación explícita del usuario. Dejar el run listo y la evidencia escrita, y preguntar.
- **Honestidad de la evidencia:**
  - No escribir PASS, "verificado" ni "validado" sin haberlo ejecutado.
  - Todo número va acompañado del comando que lo produjo y de un archivo de evidencia en `docs/audit/...`.
  - La interfaz nunca dice "production ready", "live" ni "hardware verified".
  - Cada clasificador nuevo que se promueva lleva su `audit_findings.json`.
- **Documentación al cerrar cada tarea:**
  - actualizar `docs/VALIDATION_SUMMARY.md`;
  - actualizar el reporte que corresponda (`AI_TECHNICAL_AUDIT.md` o `SIMULATION_STABILITY_REPORT.md`) con un hallazgo nuevo o con el estado del existente;
  - si se agregó un script, documentarlo en su docstring con el comando de uso.
- **Bloqueos:** si una tarea resulta inviable, o el resultado niega la hipótesis (por ejemplo, AUC ≈ 0,5), detenerse, documentar el resultado negativo como hallazgo y no forzar la implementación.

## Resumen y orden sugerido

| Orden | ID | Tarea | Prioridad | Esfuerzo | Depende de |
|---|---|---|---|---|---|
| 1 | P0 | Preparación: entorno, línea base y CI de `5427819` | Alta | S | — |
| 2 | V0 | Fijar con una prueba que ADF/BDF = ZOH exacto de (L, R_f, C, Ts) | Alta | S | P0 |
| 3 | V1 | Extraer parámetros de los `.slx` de DPC4PowerElectronics | Alta | M | V0 |
| 4 | V2 | Port del CCS-MPC original como línea base y comparación DPC vs MPC | Alta | M | V1 |
| 5 | C1 | Precisión del DPC en el punto nominal | Alta | L | V2 (recomendado) |
| 6 | A1 | Detector OOD para fallas no vistas (bola y jaula) | Alta | L | P0 |
| 7 | V4 | Ampliar la equivalencia física contra gym-electric-motor | Media | M | P0 |
| 8 | V5 | Controladores de referencia de GEM frente a PI, FOC y MTPA de SPADE | Media | M | V4 |
| 9 | A3 | Pronosticador RPi5 sin operaciones Flex | Media | M | P0 |
| 10 | A4 | Paso 0 de separabilidad: degradación de capacitor en el VSC | Media | M | V0 |
| 11 | P1 | Conectar el PMSM al pipeline de datos | Media | M | V4 (recomendado) |
| 12 | C2 | S-04: ponderación de setpoint opcional en el PI | Baja | S | P0 |
| 13 | C3 | Reducir el costo de CPU del MPC | Baja | M | P0 |
| 14 | P2 | Página de Transfer Learning bajo Streamlit 1.60 (Mac Intel) | Baja | S | P0 |
| 15 | V3 | Fidelidad del entrenamiento DPC frente a `Main.txt` | Baja | S | V1 |
| 16 | V6 | Deriva de GEM upstream desde la versión 3.0.3 | Baja | S | V4 |
| 17 | A2 | (Opcional) Clases ball y cage en el clasificador | Opcional | M | A1 + decisión |
| 18 | P3 | (Opcional) Kit de benchmark de borde, solo preparación | Opcional | M | A3 |
| 19 | P4 | Cierre: resumen consolidado y CI en verde | Alta | S | todas |

Esfuerzo: S ≈ menos de media jornada · M ≈ 1 jornada · L ≈ 2 o más jornadas.

---

## P0. Preparación

- [x] **P0. Entorno, línea base y CI** — cerrada 2026-10-20: 655 aprobadas (no slow), smoke 6/6, CI de `5427819` en verde ([`p0_baseline.txt`](docs/audit/validation/p0_baseline.txt)).
  - Crear el entorno con Python 3.12 y correr `pytest -m "not slow"` y `python scripts/smoke_test.py --launch`. Guardar el resultado en `docs/audit/validation/p0_baseline.txt`.
  - Revisar el estado del CI del commit `5427819`. Ojo: `gh run list --commit` necesita el SHA **completo**; con el corto devuelve una lista vacía. Usar `gh run list --commit $(git rev-parse 5427819)`. Si está en rojo, la tarea pasa a ser arreglarlo.
  - **Criterio de cierre:** línea base registrada (conteo de pruebas aprobadas y estado del CI).

---

## Bloque V. Validación cruzada contra los repos originales

**Por qué:** SPADE porta la física del motor DC y del PMSM desde gym-electric-motor (versión 3.0.3), y la red DPC desde DPC4PowerElectronics, que es MATLAB/Simulink. Hoy la equivalencia está probada en dos puntos:

- `tests/test_scml_system_regression.py`: un solo caso, motor DC de excitación permanente con Euler, 200 pasos;
- `tests/test_dpc_loss.py`: valida la traducción de la pérdida, pero no contra una red entrenada en MATLAB.

Este bloque amplía esa cobertura sin datos reales.

**Patrón obligatorio para GEM:** igual que en `tests/data/gen_ref_data.py`. Las trayectorias de referencia se generan una vez con GEM instalado en un entorno aparte (`uv run --with ...`), se guardan en `tests/data/*.npz`, y la prueba no importa `gym_electric_motor`. GEM no pasa a ser dependencia de `driveflow`.

### V0. Fijar el modelo identificado del VSC

- [x] **V0. ADF/BDF son el ZOH exacto de un filtro L-C con parámetros físicos** — cerrada 2026-10-20: `sim/vsc_params.py`, `tests/simulation/test_vsc_params.py`; V1 confirmó que el hardware original es LC (`Inverter_LC_filter1`).

  **Hecho ya verificado al preparar esta lista (2026-10-20, con scipy):**
  - `logm(ADF)/Ts` con Ts = 1e-4 s, y el despeje ZOH de BDF, dan exactamente L = 5,02 mH, R_f = 2,0 Ω y C = 15,0 µF, con los términos cruzados en ±1 (KCL y KVL);
  - el modelo continuo es `di_f/dt = (v_o − v_c − R_f·i_f)/L` y `dv_c/dt = (i_f − i_load)/C`;
  - como C escala con Ts, que salga exactamente 15 µF (el `CAPACITANCE_F` de `loss.py`) confirma Ts = 100 µs.

  **Hacer:**
  - Agregar `driveflow/sim/vsc_params.py` (o constantes en `vsc_system.py`) con L, R_f y C, y una función `discretize_zoh(L, R_f, C, Ts)` que devuelva `(ADF, BDF)`.
  - Agregar una prueba que verifique que `discretize_zoh(5.02e-3, 2.0, 15e-6, 1e-4)` reproduce ADF/BDF de `control/dpc/loss.py` con tolerancia de 1e-6 relativo.
  - Corregir el docstring de `sim/vsc_system.py`: Ts ya no es "inferido con ~2 %", sino confirmado por identificación.
  - Revisar las 7 menciones de "LCL" en el código y la documentación (README, `loss.py`, `vsc_system.py`, `macro_fase_B2_dpc_deployment.md`). El modelo identificado tiene 2 estados (filtro L-C con carga resistiva). Confirmar con V1 si el hardware original era LCL; si no hay evidencia, cambiar a "LC filter" o aclarar "modelo identificado de 2 estados".

  **Criterio de cierre:** prueba nueva en verde y documentación corregida. Esta tarea habilita A4, que necesita variar C y re-discretizar.

### V1. Parámetros desde los `.slx` originales

- [ ] **V1. Extraer parámetros de `CCS_MPC.slx` y `DPCvalidation.slx` sin MATLAB**
  - Descargar a `data/upstream/dpc4pe/` (sin commitear) `CCS_MPC.slx`, `DPCvalidation.slx`, `Main.txt`, `Custom_Loss.txt`, `Model_gradients.txt` y `Data4train.mat`, todos con licencia MIT.
  - Un `.slx` es un zip (OPC). Descomprimirlo y parsear `simulink/blockdiagram.xml`, o `simulink/systems/*.xml` según la versión, para encontrar:
    - período de muestreo;
    - L, C, R_f, cargas R;
    - V_dc y límites de voltaje;
    - frecuencia y magnitudes de referencia;
    - pesos de costo del MPC;
    - estructura del MPC (horizonte, restricciones, saturación).
  - Si algún parámetro vive en un script de workspace que no está en el repo, registrarlo como **no encontrado**, sin inventarlo.

  **Entregables:**
  - script `experiments/extract_dpc4pe_slx_params.py`;
  - reporte `docs/audit/upstream/dpc4pe_parameters.md`, con una tabla parámetro → valor en el `.slx` → valor en SPADE → ¿coincide?

  **Criterio de cierre:** tabla completa con evidencia (ruta del XML y bloque). Toda discrepancia debe quedar como hallazgo nuevo (S-09…).

### V2. El MPC original como línea base

- [ ] **V2. Portar el CCS-MPC y compararlo con DPC v3 y v4**

  Según su README, DPC4PowerElectronics aprende de un MPC explícito de un paso. Ese MPC es la referencia natural de lo que el DPC debería lograr.

  **Hacer:**
  - Implementar `control/mpc/vsc_ccs_mpc.py` con la estructura extraída en V1. Si V1 no logra extraerla completa, implementar el CCS-MPC estándar sobre el mismo ADF/BDF y la misma función de costo de `Custom_Loss.txt`, con horizonte 1 y 5, y rotularlo **"reconstruido, no extraído"** en el código y en el reporte.
  - Correrlo sobre la misma grilla que `tests/test_dpc_robustness_grid.py` y `experiments/measure_dpc_rmse_by_r_range.py`: R entre 3,5 Ω y 5 kΩ, y |v_ref| = 10, 35, 50, 120 y 150 V.
  - Métricas: RMSE estabilizado de v_c, THD de v_c, esfuerzo de control y tiempo de CPU por paso, para MPC, DPC v3 y DPC v4.

  **Entregables:**
  - `experiments/compare_dpc_vs_mpc.py`;
  - `docs/audit/stability/dpc_vs_mpc.md` con su JSON y una figura;
  - una prueba (`slow`) que fije que el MPC sigue la referencia en el punto nominal.

  **Criterio de cierre:** tabla comparativa completa. Esta tabla define la meta realista para C1.

### V3. Fidelidad del entrenamiento DPC

- [ ] **V3. Comparar `train.py` contra `Main.txt` y `Model_gradients.txt`**
  - Tabla de fidelidad: optimizador, learning rate, épocas, batch, `num = 6000`, normalización, inicialización, arquitectura y semilla. Marcar cada diferencia como intencional (con su justificación) o como hallazgo.
  - Intento acotado a 1 hora: buscar `trainedNet1nDPCv4_1.mat` en la historia git del repo original y comprobar si es MAT v7.3 (HDF5), legible con h5py.
    - Si los pesos son recuperables, comparar sus salidas con las de la red DPC v1 de SPADE sobre las filas de holdout de `Data4train.mat` (6000–9999).
    - Si no, documentar por qué (objeto `dlnetwork` MCOS) y cerrar.

  **Entregable:** sección en `docs/audit/upstream/dpc4pe_parameters.md`.

### V4. Equivalencia física contra gym-electric-motor

- [ ] **V4. Ampliar las trayectorias de referencia de GEM**
  - Confirmar qué versión de GEM se portó (3.0.3, según los docstrings de `sim/`) e instalar esa misma en un entorno aparte.
  - Casos nuevos, cada uno con su `.npz` en `tests/data/` y su script generador junto a `gen_ref_data.py`:
    1. **Motor DC** con otro solver de GEM (`ScipySolveIvpSolver` o RK, el que exista en 3.0.3), 2000 pasos, carga distinta y escalones de voltaje.
    2. **PMSM** con los mismos parámetros que usa `control/classical/pmsm_foc.py`, una secuencia de voltajes dq o abc fija, y velocidad alta (≥ 400 rad/s). Comparar tres casos: Euler de 1 subpaso en GEM, Euler de 3 subpasos en SPADE (fix de S-01) y `solve_ivp` en GEM como verdad. Esto valida S-01 de forma independiente de SPADE.
    3. **Motor de inducción** (`sim/motors/squirrel_cage_induction_motor.py`), solo si se usa en alguna parte. Si no se usa, documentarlo y omitirlo.
  - Tolerancias:
    - mismo solver y mismo paso: igualdad exacta o `rtol = 1e-12`;
    - solvers distintos: error relativo reportado y acotado (por ejemplo, < 0,5 % en i_s y ω), con el valor medido en el docstring.

  **Criterio de cierre:** pruebas nuevas con marcador `simulation` en verde, sin importar GEM.

### V5. Controladores de referencia de GEM

- [ ] **V5. Comparar los controladores clásicos de GEM con los de SPADE**
  - Usar el framework de control clásico de GEM (paquete `gem-controllers` o el equivalente vigente; verificar el nombre) en un entorno aparte. Registrar las respuestas en `.npz`.
  - Comparaciones:
    1. **Lazo de velocidad del motor DC:** controlador de GEM frente al PI en cascada de SPADE, sobre la misma planta. Métricas: tiempo de subida, sobreimpulso, tiempo de establecimiento y error estacionario. Tiene que incluir el escalón pequeño de S-04 (≈15 % de sobreimpulso en SPADE).
    2. **Torque del PMSM con MTPA:** la curva MTPA de GEM (tabla o analítica) frente a la MTPA cerrada de `pmsm_foc.py` en todo el rango de torque. Hoy solo se compara un punto (`test_matches_gems_own_torque_limit_at_nominal_current`). Reportar el error máximo en i_d e i_q.
  - No se espera igualdad: son sintonías distintas. El objetivo es tener una línea base independiente, así que se documentan las diferencias.

  **Entregable:** `docs/audit/upstream/gem_controller_comparison.md` y una prueba de la curva MTPA completa.

### V6. Deriva de GEM upstream

- [ ] **V6. Revisar qué cambió en GEM después de la 3.0.3**
  - Hacer un diff de los archivos portados (motores, convertidores, cargas, solvers y sistema físico) entre la 3.0.3 y la versión actual de GEM. Listar las correcciones de física o de bugs posteriores al port.
  - Para cada una: ¿afecta a SPADE? ¿conviene incorporarla? No incorporar nada sin confirmación: solo reportar.

  **Entregable:** `docs/audit/upstream/gem_upstream_drift.md`.

---

## Bloque A. Capa de IA

### A1. Detector fuera de distribución (prioridad alta)

- [ ] **A1. Que una falla sin clase deje de clasificarse con confianza alta**

  **Problema** (AI-N12, ver `docs/AI_TECHNICAL_AUDIT.md` §7.2):
  - las fallas de bola salen como outer_race con confianza 0,993 y la abstención retiene el 0 %;
  - las fallas de jaula se retienen en un 54 %, y el resto sale como normal, lo que equivale a una falla no detectada.

  **Regla anti-fuga:** el detector y su umbral se ajustan solo con datos en distribución (healthy, outer_race e inner_race; train y val). Las ventanas de bola y jaula se usan solo para evaluar. Ninguna decisión de diseño se toma mirando los resultados de bola y jaula; si se itera, declararlo en el reporte.

  **Métodos a comparar** (implementar al menos los dos primeros):
  1. **Mahalanobis sobre el embedding del clasificador:** capa `gap` o `dense1` de `models/classifiers/builder.py`, con medias por clase y covarianza compartida, ajustadas en train.
  2. **Energy score** sobre los logits (antes del softmax).
  3. **Chequeo cruzado físico:** el agente basado en simulación (`agents/dc_motor_agent.py`) ya tiene hipótesis de bola y jaula. Marcar "unknown" cuando la mejor hipótesis del agente no coincide con la clase del clasificador.

  **Umbral:** el que retiene el 95 % de las ventanas en distribución de validación, igual que la abstención actual.

  **Integración:**
  - guardar los parámetros en `metrics.json["ood"]` del run;
  - extender `ai/calibration.decide()` con la etiqueta `unknown`;
  - mostrar en la GUI "Unknown condition — REQUIRES VERIFICATION";
  - la prueba de AI-N12 pasa a cubrir bola y jaula.

  **Alcance por nivel:** el PC es obligatorio. Para RPi5 y ESP32, documentar si es viable (requiere exportar el embedding como segunda salida); implementarlo solo si es barato.

  **Metas** (sobre ventanas simuladas, con la misma grilla de condiciones):
  - ≥ 90 % de las ventanas de bola marcadas como unknown;
  - ≥ 90 % de las de jaula marcadas como unknown o como falla (no como normal);
  - en distribución, la exactitud sobre lo respondido baja como máximo 1 punto;
  - reportar AUROC y la tasa de falsos unknown en corrimientos benignos: velocidades fuera de la grilla (50 y 300 rad/s) y 150 % de ruido.

  **Entregables:**
  - `ai/ood.py`;
  - `experiments/report_ood_detector.py`;
  - `docs/audit/ai/ood_detector.{md,json}`;
  - pruebas con marcador `ai`.

  Si no se alcanzan las metas, documentarlo como resultado y pasar la decisión a A2.

### A2. Clases ball y cage (opcional, requiere decisión)

- [ ] **A2. Agregar las clases solo si el usuario lo aprueba después de A1**

  **Advertencia:** la amplitud de las fallas de bola y jaula en `datagen/scenario.py` es un placeholder. KAt no tiene un código limpio de defecto único para esas fallas, así que no puede calibrarse sin datos reales.

  Si se hace, el clasificador queda rotulado como **"ball/cage: synthetic-only, uncalibrated amplitude"** en `audit_findings.json` y en la GUI.

  **Pasos:** regenerar la grilla con 5 clases, reentrenar el PC, destilar a RPi5 y ESP32, recalibrar, volver a correr A1 (las clases dejan de ser OOD, así que hay que buscar otro conjunto OOD, por ejemplo una falla personalizada desde `presets.py`) y no promover sin confirmación.

### A3. Pronosticador RPi5 sin operaciones Flex

- [ ] **A3. Exportar el pronosticador RPi5 solo con operaciones nativas de TFLite**

  **Problema:** `configs/regressors/rpi5_edge.yaml` usa un GRU, y la exportación cae en `SELECT_TF_OPS` (`ai/tflite_export.py`). Por eso no corre en tflite-runtime ni en LiteRT puro.

  **Opciones a probar, en orden:**
  1. **TCN** (convoluciones causales dilatadas) en `models/regressors/builder.py`, configurable por YAML. Ventana de 256 y horizonte de 64, igual que `pc_full`; es lo que exige la destilación.
  2. **GRU con `unroll=True`** (ojo: 256 pasos desenrollados inflan el grafo; medir el tamaño).

  Entrenar con `--distill` desde `vsc_dpc/pc/regressor` y exportar a float16.

  **Criterios:**
  - `ai/deployment_status.py` reporta cero operaciones Flex;
  - el RMSE sobre el split por condición (`experiments/report_forecaster_condition_split.py`) es ≤ 1,1 veces el del `rpi5_edge` actual y mejor que la persistencia;
  - la paridad TFLite frente a Keras queda en `report_tflite_parity.py`.

  Dejar el run listo y no promover sin confirmación.

### A4. Paso 0: degradación de capacitor en el VSC

- [ ] **A4. Medir la separabilidad antes de construir cualquier clasificador**

  Especificación en `docs/INSTRUCTIONS.md` §6, D.2, Paso 0 (obligatorio).

  Ahora es viable gracias a V0: se parametriza C_efectivo (por ejemplo 100 %, 90 %, 80 % y 70 % del nominal) y ESR, y se re-discretiza con `discretize_zoh`. Antes no se podía porque ADF/BDF eran constantes fijas.

  **Ojo:**
  - el DPC fue entrenado con C nominal y su pérdida usa C nominal, así que el lazo cerrado puede enmascarar la degradación;
  - incluir en las features el esfuerzo de control v_o además de (v_c, v_ref, i_f);
  - para ESR hay que extender el modelo continuo (resistencia en serie con C) antes de discretizar.

  Calcular el AUC por nivel de degradación, sobre varias cargas R y varias magnitudes de referencia.

  **Regla de corte** (fijarla antes de mirar resultados y escribirla en el reporte): si el AUC es < 0,7 en el nivel del 80 %, detener D.2 y documentar la limitación.

  **Entregables:**
  - `sim/vsc_faults.py`;
  - `experiments/verify_vsc_capacitor_separability.py`;
  - `docs/audit/ai/vsc_capacitor_separability.{md,json}`;
  - la advertencia epistemológica (no existe ningún dataset real de referencia para esta falla).

  El clasificador en sí no es parte de esta tarea. Si el AUC pasa el corte, proponerlo como tarea nueva y preguntar.

---

## Bloque C. Control

### C1. Precisión del DPC en el punto nominal

- [ ] **C1. Recuperar la precisión de v3 en el punto nominal sin perder el rango de v4**

  **Estado actual:**

  | Condición | v3 | v4 |
  |---|---|---|
  | 8 Ω | 1,18 V | 2,52 V |
  | \|v_ref\| = 10 V | 1,75 V | 4,2 V |

  v5 divergió (`docs/audit/stability/dpc_r_range_finetune_v5.json`).

  **Requisito:** `Data4train.mat` en `data/upstream/dpc4pe/` (lo descarga V1).

  **Opciones, en orden de costo:**
  1. **Selección por punto de operación, sin entrenar:** usar v3 cerca de 8 Ω y con referencias pequeñas, y v4 en el resto, con histéresis para no conmutar en cada paso. Medir el transitorio de conmutación.
  2. **Pérdida ponderada hacia referencias pequeñas.** Probable causa raíz: el error cuadrático absoluto hace despreciable el error cuando |v_ref| es pequeño. Probar el error normalizado por |v_ref| o un peso extra en R ≈ 8 Ω.
  3. **Red más grande** (más unidades o capas en `control/dpc/network.py`).

  **Lecciones de v5:**
  - guardar el mejor checkpoint según una evaluación en lazo cerrado sobre R_EVAL de held-out, no el último;
  - early stopping;
  - registrar cada ronda en `docs/audit/stability/dpc_*_v6.json`.

  **Metas:**
  - 8 Ω ≤ 1,3 V;
  - |v_ref| = 10 V ≤ 2,0 V;
  - 4, 20 y 1000 Ω siguen ≤ 3,5 V;
  - sin divergencia en ninguna R > R\* (3,3707 Ω).

  Usar como referencia lo que logra el MPC en V2.

  **Criterio de cierre:** la tabla v3, v4 y candidato en `SIMULATION_STABILITY_REPORT.md`. Los umbrales de `test_dpc_robustness_grid.py` se re-miden solo si se aprueba cambiar el default. No cambiar el checkpoint por defecto sin confirmación.

### C2. S-04: sobreimpulso del PI

- [ ] **C2. Agregar ponderación de setpoint opcional**
  - En `control/classical/pi_controller.py`, agregar el parámetro `setpoint_weight` b ∈ [0, 1], o en su defecto un prefiltro de referencia.
  - El default conserva el comportamiento actual (b = 1), para que las pruebas byte a byte y los datasets no cambien.
  - Medir el sobreimpulso en pequeña señal con b = 1, 0,7 y 0,5 frente al modelo lineal, comparando con el lazo de GEM (V5) si ya existe.
  - Exponer el parámetro en la página DC Motor como opción avanzada.

### C3. Costo de CPU del MPC

- [ ] **C3. Reducir las ≈65–75 veces que el MPC cuesta más que el PI**
  - Perfilar `control/mpc/controller.py` (`_solve` y `cost_and_grad`) y aplicar, en orden:
    1. precalcular las matrices condensadas;
    2. warm start con la solución desplazada;
    3. QP cerrado o MPC explícito si las restricciones lo permiten.
  - **Restricción:** las trayectorias deben coincidir con las actuales dentro de una tolerancia declarada (reportar el error máximo en ω e i).
  - **Meta:** al menos 3 veces más rápido. Medir con `experiments/run_stability_battery.py` o con un benchmark dedicado.

---

## Bloque P. Plataforma

### P1. PMSM en el pipeline de datos

- [ ] **P1. Hacer del PMSM un sistema con datasets, como los otros dos**
  - En `datagen/scenario.py`:
    - agregar `plant_config_id = "pmsm_v1"`;
    - agregar `controller_type` con los valores `"FOC_MTPA"` y `"FOC_naive"`;
    - registrar los pares en `_VALID_PAIRS`;
    - validar los campos igual que en S-03.
  - En `datagen/runner.py`, emitir filas con un esquema documentado: i_d, i_q, ω, torque, referencias y voltajes dq.
  - Crear el generador `experiments/generate_pmsm_dataset.py`.
  - Agregar las pruebas de datagen correspondientes. Las existentes no deben cambiar.

  **Fuera de alcance:** el diagnóstico de fallas del PMSM. Requiere definir primero un modelo de falla (por ejemplo, rizado de torque por rodamiento reutilizando `datagen/fault_injection.py`, o desmagnetización). Si se quiere, proponerlo como tarea nueva.

### P2. Página de Transfer Learning con Streamlit 1.60

- [ ] **P2. Corregir el fallo de AppTest al saltar al paso 4**
  - Reproducirlo en Linux con `constraints/macos-intel.txt`. Ya se emuló así antes: ver `docs/audit/install/macos_intel_emulated_on_linux.txt`, que reporta 3 fallos.
  - Corregir la página (`viz/pages/transfer_learning.py` y `viz/transfer_learning_dashboard.py`) para que el paso 4 sin pasos previos muestre un estado vacío claro en vez de fallar.
  - Revisar también el fallo de la heurística de tamaño TFLite de esa misma emulación.

  **Criterio de cierre:** 0 fallos con el conjunto de restricciones de Mac Intel, emulado en Linux.

### P3. Kit de benchmark de borde (opcional, solo preparación)

- [ ] **P3. Dejar listo lo que se correrá cuando exista hardware**
  - Crear `scripts/edge_benchmark/`:
    - para RPi5, un script autónomo que mida latencia (p50 y p99), memoria residente y concordancia contra salidas de referencia del PC guardadas en `.npz`, usando solo tflite-runtime o LiteRT;
    - para ESP32, un esqueleto de TFLite Micro con el modelo como arreglo C y los vectores de referencia.
  - No declarar nada medido en hardware. El README del kit debe decir explícitamente **"not executed on hardware"**.

### P4. Cierre

- [ ] **P4. Resumen consolidado**
  - Actualizar `docs/VALIDATION_SUMMARY.md`: tabla de pruebas, estado por módulo, hallazgos nuevos (S-09…, AI-N13…) y una sección nueva de validación cruzada contra implementaciones de referencia.
  - Confirmar el CI en verde en `dev` y abrir un PR de `dev` a `main` con el resumen.

---

## Descartadas: requieren datos reales

Quedan fuera hasta que existan datos medidos:

- Conjunto de prueba real para el clasificador (otra máquina u otro rodamiento, al menos dos velocidades, fallas de pista interna y externa).
- Calibración de vibración por rodamiento y multieje: hoy es gruesa y de un solo eje.
- Calibración de la amplitud de las fallas de bola y jaula contra datos medidos (ver A2).
- Umbrales de monitoreo validados fuera del simulador.

## Fuera de alcance por equipo físico (no por datos)

- Ejecutar los `.tflite` en una Raspberry Pi 5 o una ESP32 (P3 solo deja el kit listo).
- Instalación manual en Windows 11, Mac Apple Silicon, Mac Intel y Ubuntu 22.04 o 24.04 (el CI ya cubre la resolución y el smoke test).

## Requieren una decisión tuya antes de empezar

- **Regresor para `dc_motor`:** no hay un caso de uso definido. Una opción es pronosticar ω e i para detectar anomalías por residuo. Sin caso de uso, no se hace.
- **A2 (clases ball y cage):** se decide después de ver el resultado de A1.
- **Cambiar defaults o promover modelos** que salgan de C1, A1 o A3.
