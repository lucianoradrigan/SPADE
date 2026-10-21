# SPADE: pendientes de la expansión E0–E14 y contexto para Claude Code

Generado el 2026-10-21 al cerrar la sesión que hizo E0, E1, E6, E10a, E10b, E12 y E2.

Documentos compañeros:
- `TODO_EXPANSION_CLAUDE_CODE.md`: la lista original. Manda sobre este archivo en todo lo que no esté actualizado aquí.
- `docs/design/SPADE_GUI_Complete_Specification.md`: la spec de la GUI, v3.0.

Cómo usarlo: pídele a Claude Code "Lee PENDIENTES_EXPANSION_CLAUDE_CODE.md y TODO_EXPANSION_CLAUDE_CODE.md y continúa con la tarea X". Antes de empezar una tarea marcada como bloqueada, confirma la decisión que la desbloquea (§4).

---

## 1. Estado del repositorio

| Elemento | Valor |
|---|---|
| Rama de trabajo | `dev`, en `acebd43`, subida a `origin/dev` |
| `main` | `a2f2cd3`. No se ha tocado. El PR de `dev` a `main` se hace solo con la confirmación de Luciano. |
| Worktrees | `SPADE/` está en `main`. `SPADE-dev/` está en `dev`, con su propio `.venv`. Las tareas nuevas se hacen en `SPADE-dev`. |
| Suite completa | **1002 aprobadas, 0 fallidas** (≈11 min, `.venv/bin/python -m pytest -q`) |
| Línea base antes de la expansión | 760 aprobadas |

Commits de la expansión en `dev`, cada uno con una rama `feat/E<N>-…` subida a `origin`:

| Commit | Tarea | Suite al cerrar |
|---|---|---|
| `07986df` | E0: runs y jobs durables, worker, componentes, Experiment Runs | 823 |
| `4820e88` | E1: Gymnasium `SPADE/DcMotor-v0` y `SPADE/Vsc-v0`, recompensa, registro, página Environments | 845 |
| `c619a0c` | E6: Buck promediado, controlador PI de voltaje, `buck_v1`, `SPADE/Buck-v0`, página DC-DC Converters | 871 |
| `0342573` | E10a: estados Known / Uncertain / Unknown / Input invalid en AI Diagnostics | 890 |
| `48d6617` | E10b: identificación de L, R_f y C del VSC (PINN Keras y mínimos cuadrados ZOH) y pestaña Parameter Identification | 905 |
| `d1c96f6` | E12: Help / Algorithm Guide | 949 |
| `acebd43` | E2: cliente RPC de PLECS, perfiles `plecs_rpc_batch`, barridos como jobs, página PLECS Plant Studio (ruta Dataset) | 1002 |

La evidencia de cada tarea está en `docs/audit/expansion/e<N>_*.md`, con su `e<N>_branch_pytest.txt`.

---

## 2. Reglas que siguen vigentes

Vienen de `TODO_EXPANSION_CLAUDE_CODE.md` y de las instrucciones de Luciano.

**Ramas y commits**
- Una tarea por rama: `feat/E<N>-<nombre>`, creada desde `dev`. Al cerrarla: suite completa en verde, merge fast-forward a `dev` y push de `dev`.
- **Nunca hacer push a `main`.** El PR de `dev` a `main` solo con la confirmación de Luciano.
- Los commits terminan con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Dependencias y modelos**
- **No promover modelos ni cambiar valores por defecto** sin la confirmación explícita de Luciano.
- **Dependencias nuevas** (fmpy, pyswarms, pymoo, PyLTSpice, redis/rq u otras): confirmación antes de tocar `pyproject.toml`. Primero se intenta con la librería estándar, numpy, scipy o scikit-learn.
- **Sin copiar código de PyTorch.** Todo se reimplementa en Keras/TensorFlow, citando el notebook de origen en el docstring.

**Etiquetas de procedencia**
- Nunca se usa `LIVE`.
- Las etiquetas válidas son SIMULATED, SIMULATED · PLECS, SIMULATED · SURROGATE, EXTERNAL FILE y MEASURED · EXTERNAL DATASET.

**Evidencia y datos externos**
- Ningún PASS se declara sin haber corrido la prueba.
- Nada medido en PLECS se declara sin el log de la máquina donde corrió. Si no hay PLECS, se escribe "no ejecutado: sin PLECS".
- Archivos externos:
  - van a `data/upstream/fund/` y no se commitean;
  - el contenido de Fund. que no es código es CC BY-NC 4.0;
  - Sel. no tiene licencia, así que no se copia nada de él.
- Las pruebas de datagen byte a byte no se tocan.

**Arquitectura**
- Capas: view → adapter → dominio.
- El backend decide las capacidades. Ningún botón se habilita por apariencia.
- Layout de paquetes fijo: `runs/`, `jobs/`, `integrations/plecs/`, `envs/`, `optim/`, `control/rl/`, `sim/buck_system.py`, `ai/pinn/`, `ai/prognostics/`. **No se crea `src/driveflow/experiments/`**.

---

## 3. Qué existe y dónde (para no reinventarlo)

### Runs y jobs (E0)
- **Contratos** (`runs/schemas.py`): `PlantProfile`, `EnvironmentCapability`, `ExperimentSpec`, `Run`, `ApprovalRecord`.
- **Store** (`runs/store.py`): SQLite en `workspace/`. `$SPADE_WORKSPACE` cambia la ruta, y los tests usan un workspace aislado (`tests/conftest.py`). Tiene `list_profiles()` desde E2.
- **Jobs:**
  - `jobs/` con un worker como proceso aparte: `python -m driveflow.jobs.worker serve`.
  - Hay un lock por motor, así que un job `plecs_sweep` corre de a uno.
  - Para agregar un tipo de job: escribir una entrada en `jobs/handlers.py` `HANDLERS`, con `fn`, `validate`, `title` y `engine`.
- **Perfiles nativos** (`runs/profiles.py`): `native:dc_perm_ex_v1`, `native:vsc_dpc_v1`, `native:pmsm_v1` y `native:buck_v1`.

### Gymnasium (E1, E6)
- **Entornos** (`envs/native.py` y `envs/registry.py`):
  - `SPADE/DcMotor-v0`, `SPADE/Vsc-v0` y `SPADE/Buck-v0`;
  - los tres pasan `check_env`;
  - `step()` reproduce `run_scenario` bit a bit.
- **Recompensa** (`envs/reward.py`): términos `tracking`, `overcurrent` y `action_rate`. `evaluate()` devuelve el total y el desglose.
- **Pendientes:** `SPADE/Pmsm-v0` no tiene tarea asignada, y los entornos de plantas PLECS llegan con E3/E4.

### Buck (E6)
- `sim/buck_system.py`:
  - modelo promediado síncrono con ZOH exacto, τ = 20 µs;
  - error frente a `solve_ivp`: 3e-10 V.
- `control/classical/buck_pi.py`:
  - cascada de corriente promedio;
  - resultados: se asienta en 1,32 ms; ante un escalón de carga de 6 a 3 Ω cae a 22,0 V y se recupera en 1,7 ms.
- `Scenario(plant_config_id="buck_v1", controller_type="PI", …)`.

### IA
- **E10a** (`ai/condition.py`):
  - El clasificador promovido de `dc_motor` (`pc_server_5class_w2048/2026-10-20_run01`) **no tiene bloque `ood`**. Por eso la página muestra "OOD detection unavailable" y el estado Unknown no puede aparecer.
  - Ajustarle un detector es una decisión aparte. El detector de A1 detecta solo el 13 % de las ventanas de falla de bola.
- **E10b** (`ai/pinn/identification.py`; reporte en `docs/audit/ai/pinn_identification.md`):
  - La PINN es robusta al ruido, pero **estima C sistemáticamente bajo** (−9 % de error medio con 5 % de ruido).
  - Los mínimos cuadrados son exactos sin ruido y colapsan con él.
  - **El intervalo bootstrap no está calibrado**: cobertura del 52 % al 78 % con ruido.
  - **El lazo DPC diverge con C×0,4.**

### PLECS (E2)
- **Integración** (`integrations/plecs/`):
  - `rpc_client.py`: xmlrpc, solo `localhost`, errores tipados, `SolverOpts.Timeout`.
  - `manifest.py`: ModelVars y salidas declaradas a mano.
  - `batch.py`: perfiles con SHA-256 (quedan stale si cambia el archivo), diseño de grilla o LHS, `run_condition`.
- **Scenario:** `plant_config_id="plecs:<profile_id>"` con `controller_type="circuit"`.
- **Datasets:** se escriben en `data/plecs/<profile>/<run_id>/`. `$SPADE_PLECS_DATA` cambia la ruta.
- **Pruebas:**
  - servidor PLECS falso en `tests/integrations/plecs/fake_plecs.py`, multihilo, con timeout, no convergencia, caída de conexión y busy;
  - **no hubo PLECS real** (`docs/audit/plecs/e2_rpc_sweep.md` tiene los pasos para producir la evidencia).
- **Datos del RPC de PLECS** (documentación oficial, leída el 2026-10-21):
  - URL `http://localhost:1080`;
  - RPC deshabilitado por defecto y solo para clientes locales;
  - `StopTime` está obsoleto: se usa `TimeSpan`;
  - no hay un comando RPC que devuelva la versión.

### Páginas nuevas (`viz/app.py` `PAGES`)
- **Workspace:** Experiment Runs, Environments.
- **Simulation:** DC-DC Converters, PLECS Plant Studio.
- **Help:** Algorithm Guide (tabla en `configs/algorithm_guide.yaml`).
- **Pestañas nuevas en páginas existentes:** "Parameter Identification" en VSC / DPC, y los estados de condición en AI Diagnostics.

### Trucos útiles
- **Capturas** con `uv run --with playwright python <script>`:
  - cargar primero la raíz de la app y esperar unos 8 s antes de navegar a una página, porque entrar directo falla en frío;
  - usar un viewport alto (unos 3000 px), porque Streamlit hace scroll dentro de su propio contenedor.
- **Segundo worktree para trabajar en paralelo:** `git worktree add ../SPADE-eN -b feat/… dev`, y ejecutar con `PYTHONPATH=src ../SPADE-dev/.venv/bin/python`.
- **Cuidado con `pkill -f`/`pgrep -f`:** un patrón que aparece en la misma línea de comando mata o espera al propio shell. Hay que usar el truco de corchetes (`"[f]ake_serve"`) y ejecutarlo en un comando aparte.

---

## 4. Pendientes

### 4.1 Bloqueadas por una decisión de Luciano

| Tarea | Qué falta decidir | Notas |
|---|---|---|
| **E5**: surrogate y "Compare with SPADE" | (a) ¿se construye ya, validado con datos del Buck nativo declarados como sustituto, o se espera a tener datos reales de PLECS? (b) ¿qué señal del circuito es la **acción** del Gym surrogate? Propuesta: el duty, expuesto como ModelVar o como entrada. | E2 deja datasets en `data/plecs/…`. La pestaña "Compare with SPADE" ya existe en Plant Studio como estado vacío. El split es por condición y la etiqueta es `SIMULATED · SURROGATE`. |
| **E3**: spike FMU go/no-go | Agregar **fmpy** como extra opcional `[plecs]`, y conseguir **una máquina con PLECS 5.0, licencia y, si existe, PLECS Coder**. | Sin eso queda bloqueada y documentada así. Un NO-GO también es un resultado válido. |
| **E4**: `PlecsGymEnv` y el asistente de 6 etapas | Solo si E3 = GO. | La tarjeta Gym de Plant Studio ya muestra "Interactive stepping is not available with the detected backend." |
| **E7**: Tuning Studio (PSO, BO, multiobjetivo) | ¿pyswarms/pymoo o implementación propia? | **Recomendación** (la que ya dice el TODO): PSO y NSGA-II propios en numpy, y BO con `GaussianProcessRegressor` de scikit-learn, que ya es dependencia. Así no hace falta ninguna dependencia nueva. Primer caso: S-04 (escalón pequeño en el PI del motor DC), coordinado con C2. **"Use as default" pasa por ApprovalGate más la confirmación de Luciano.** |
| **E8**: RL Lab (DQN y DDPG en Keras) | Depende de E7, porque usa su ApprovalGate. | Primer caso: el Buck (`SPADE/Buck-v0`). Segundo: el VSC frente a DPC v3 y v4. El benchmark se publica aunque RL pierda. |
| **E9**: DAB | Si se quiere un tercer convertidor, y la licencia de `DAB_sample1.plecs` (posible CC BY-NC). | Entraría por E2 (dataset) y, si E3 = GO, por E4. |
| **E11**: RUL de IGBT con datos de la NASA | Si se quiere abrir el dominio `igbt_aging`, y la licencia del dataset. | Serían los primeros datos medidos, pero de otro componente. |
| **E14.0**: regresor `dc_motor` por residuo | El caso de uso (el recomendado es el residuo de pronóstico) y las metas: cuantil del umbral, AUROC ≥ 0,9, falsas alarmas ≤ 1 %, RMSE ≤ 0,8 × persistencia. | E14.1–E14.6 solo después de E14.0. |
| **Defaults y promociones** | Cualquier candidato de E7/E8, el detector OOD del clasificador promovido, y los regresores de `dc_motor`. | Nunca sin confirmación. |

### 4.2 Cierre (E13), cuando terminen las tareas que se hagan

- `docs/VALIDATION_SUMMARY.md`:
  - sección "Expansion (E0–E12)";
  - tabla de pruebas;
  - qué se corrió con PLECS: **nada hasta hoy**;
  - hallazgos nuevos (los de E10b: sesgo de C en la PINN, intervalo sin calibrar, divergencia de la DPC con C×0,4).
- Recorrer la matriz de la spec §14 y los criterios de §20, marcando cada fila como aprobada, fallida o no ejecutada, con el motivo.
- Capturas de cada pantalla nueva en sus 5 estados, en tema claro y oscuro, en `docs/screenshots/expansion/`. **Hasta ahora las capturas se revisaron, pero no se commitearon.**
- CI en verde en `dev`. **El PR de `dev` a `main` solo con la confirmación de Luciano.**

### 4.3 Diferidos que no son tareas E

- Medición de contraste de los tokens y el camino del heartbeat del worker en Windows (anotados para E13).
- Evidencia real de E2: un barrido de 3×3 con PLECS, siguiendo `docs/audit/plecs/e2_rpc_sweep.md`.
- `SPADE/Pmsm-v0`: no está asignado a ninguna tarea.

---

## 5. Cómo verificar el estado antes de seguir

```bash
cd ~/post_doc/plataform_proposed/SPADE-dev
git fetch origin && git status -sb          # dev == origin/dev en acebd43
.venv/bin/python -m pytest -q               # se esperan 1002 aprobadas
.venv/bin/streamlit run src/driveflow/viz/app.py
.venv/bin/python -m driveflow.jobs.worker serve   # en otra terminal, para los jobs
```
