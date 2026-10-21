SPADE: lista de tareas de expansión para Claude Code

Generada el 2026-10-20 · Base: rama dev vigente · Autor: Luciano Radrigan Documento compañero: docs/design/SPADE_GUI_Complete_Specification.md (especificación de GUI v3.0, en adelante la spec).

Cómo usar este archivo
Copia este archivo a la raíz del repo como TODO_EXPANSION_CLAUDE_CODE.md.
Copia la spec a docs/design/SPADE_GUI_Complete_Specification.md. Se mantiene en español, como documento de planificación; los textos de la interfaz que contiene ya están en inglés.
Pídele a Claude Code: "Lee TODO_EXPANSION_CLAUDE_CODE.md y ejecuta la tarea E0".
Haz una tarea por sesión y por rama. Al cerrar una tarea, marca su casilla.

Relación con TODO_CLAUDE_CODE.md:

Esta lista no lo reemplaza; se ejecuta después de su tarea P0.
Algunas tareas comparten terreno con ese archivo (ver la columna "Cruce").
Si otro agente está trabajando en la tarea cruzada, coordina antes de tocar los mismos archivos.

Qué agrega esta lista. Diez funcionalidades tomadas de dos repos externos:

Fund. = XinzeLee/Fundamentals_of_AI_for_PE. El código está bajo Apache 2.0; el contenido que no es código (texto, markdown y, por defecto, modelos y datos) está bajo CC BY-NC 4.0, según su NOTICE.
Sel. = XinzeLee/AI_for_PE_Algorithm_Selector. Es HTML/JS estático y el zip no trae LICENSE, así que no se copia código.

Prioridad del investigador: que un circuito diseñado en PLECS sea una planta del entorno tipo Gym de SPADE, sobre la que corran los algoritmos de Python (PI, MPC, DPC, RL, generación de datasets, diagnóstico). Las tareas E2 a E5 cubren esa prioridad.

Reglas (además de las de TODO_CLAUDE_CODE.md y CLAUDE.md)
Ramas: feat/E<N>-<nombre-corto>, creadas desde dev. Nunca push a main.
Separación por capas (spec §0 y §12): view → adapter → dominio. Nada de física, RL ni optimización dentro de un componente de UI.
El backend manda. Ninguna capacidad se habilita en la GUI por apariencia; los botones se habilitan según los reportes de capacidad del backend (spec §3.15). Ninguna gráfica sin un Run real detrás (spec "Principios no negociables").
No se copia código de PyTorch. Todo el ML de Fund. está en PyTorch y SPADE es solo TensorFlow/Keras. Se reimplementa en Keras citando el notebook de origen en el docstring.
Dependencias nuevas: fmpy, pyswarms, pymoo, PyLTSpice, redis/rq o cualquier otra requieren confirmación de Luciano antes de tocar pyproject.toml. Primero se intenta sin la dependencia: librería estándar, numpy, scikit-learn.
PLECS no corre en el CI ni en el contenedor. Cada prueba que necesite PLECS usa:
un servidor XML-RPC falso, para la lógica del cliente;
archivos .npz o .csv de referencia generados una vez en una máquina con PLECS, guardados en tests/data/plecs/ junto con el script que los generó, igual que el patrón de GEM.
Nada medido en PLECS se declara sin el log de la máquina donde corrió. Si no hubo PLECS disponible, se escribe "no ejecutado: sin PLECS/licencia".
Procedencia (spec §0, principio 2): SIMULATED, SIMULATED · PLECS, SIMULATED · SURROGATE, EXTERNAL FILE, MEASURED · EXTERNAL DATASET. Nunca LIVE. "PLECS available" significa que la herramienta responde; no es telemetría.
Archivos externos: el .plecs del DAB, el .mat de IGBT de la NASA y los CSV de Fund. se descargan a data/upstream/fund/ y no se commitean. Antes de commitear cualquier archivo que no sea código de Fund., verifica su licencia.
Hallazgos nuevos: siguen la numeración de CLAUDE.md (S-09…, AI-N13…). Si el otro agente ya usó un número, toma el siguiente libre.
Decisiones de diseño que esta lista fija sobre la spec

La spec es la referencia de UX. Estos puntos la precisan o la corrigen; si hay contradicción, gana esta lista.

Layout de paquetes único. Las secciones §3.7 y §12 de la spec proponen rutas distintas. Se usa este:
   src/driveflow/runs/            schemas.py, store.py, profiles.py   (PlantProfile, ExperimentSpec, Run, ApprovalRecord)
   src/driveflow/jobs/            schemas.py, service.py, worker.py, persistence.py
   src/driveflow/integrations/plecs/  rpc_client.py, fmu_adapter.py, diagnostics.py, manifest.py
   src/driveflow/envs/            native.py (DC, VSC, Buck), plecs_gym_env.py, surrogate_env.py, registry.py
   src/driveflow/optim/           pso.py, bayes.py, multiobjective.py, objectives.py
   src/driveflow/control/rl/      dqn.py, ddpg.py, policy_controller.py
   src/driveflow/sim/buck_system.py
   src/driveflow/ai/ood.py        (lo implementa la tarea A1 de TODO_CLAUDE_CODE.md)
   src/driveflow/ai/pinn/         identification.py
   src/driveflow/ai/prognostics/  rul.py
No se crea src/driveflow/experiments/ (propuesto por la spec §12), porque se confunde con la carpeta de scripts experiments/ que ya existe en la raíz.
Los datos persistentes de runs y jobs van en workspace/, que se agrega a .gitignore.
Página PLECS: el archivo es viz/pages/plecs_plant_studio.py, no circuits_plecs.py. Su título es PLECS Plant Studio. La spec §3.1 tiene una errata ("renombrar a sí mismo").
Acoplamiento paso a paso. Verificado contra la documentación oficial de PLECS el 2026-10-20 (https://docs.plexim.com/plecs/latest/functional-mockup-interface/ y …/scripting/):
El RPC de PLECS Standalone (XML-RPC o JSON-RPC) es local, viene deshabilitado por defecto y es bloqueante: atiende un solo comando bloqueante a la vez. No admite pausar ni reanudar. Un step() hecho con RPC no es un Gym válido (spec §3.7).
PLECS 5.0 exporta FMU de co-simulación (FMI 2.0/3.0) de tipo tool-coupling. Para usarla:
PLECS tiene que estar instalado y con licencia en la máquina donde corre la FMU;
hay que activar "Enable FMU Co-Simulation endpoint" en las preferencias;
para señales discretas, como las de compuerta, se recomienda FMI 3.0 (Event Mode).
Opción que la spec no menciona: PLECS también exporta FMU de Model Exchange.
Exportarla requiere licencia de PLECS Coder.
Ejecutarla no requiere licencia.
El modelo se discretiza a paso fijo y no admite tiempo de muestreo variable.
Si el grupo tiene Coder, esto permite fijar referencias y pruebas en CI sin PLECS. Se evalúa en E3.
Respaldo si ninguna FMU sirve: un bloque DLL en el circuito que llama a un servidor Python por socket. Es lo que recomienda Plexim en su foro. Se documenta como alternativa y no se implementa salvo que E3 descarte las FMU.
Jobs: SQLite (librería estándar) + un proceso worker separado (python -m driveflow.jobs.worker). Redis/RQ queda fuera de alcance; requiere decisión.
Componentes de GUI nuevos (spec §9): se construyen sobre viz/components/ y viz/theme/tokens.py; no se crea un segundo sistema visual. PromotionGate se mantiene para modelos y ApprovalGate es nuevo, para controladores y defaults.
Dos defectos del notebook PLECS de Fund. (8_PE_Simulation_Automation/PlecsAutomation/Data acquisition.ipynb) que no deben copiarse:
en writerow se llama get_a_param_set(prelim) sin índice, así que todas las filas de Performance.csv se escriben con los parámetros de la combinación 0;
la URL 'http://localhost:1080/RPC2"' tiene una comilla de más;
además, usa rutas de Windows con \\ y varios hilos contra un servidor que solo atiende un comando bloqueante a la vez.
Mapa: funcionalidad → tarea → spec
#    Funcionalidad    Origen    Tarea(s)    Sección de la spec    Cruce con TODO_CLAUDE_CODE.md
1    PLECS como planta del Gym de SPADE    Fund. 8_PE_Simulation_Automation/PlecsAutomation/    E2, E3, E4, E5    §3 completo    V0 (modelo VSC), V-block (patrón de referencias)
2    Envoltorio gymnasium.Env + catálogo Environments    Fund. 7_Reinforcement_Learning/ (diseño de BuckAvgEnv)    E1    §8.5, §3.4 etapa 5E    —
3    Tuning Studio (PSO, BO, multiobjetivo)    Fund. 1_MHA/, 2_Classic_ML/gaussian_process_bayesian_optimization.ipynb    E7    §4    C1 (DPC), C2 (S-04)
4    RL Lab (DQN, DDPG)    Fund. 7_Reinforcement_Learning/*.ipynb    E8    §5    —
5    Planta Buck DC-DC    Fund. BuckAvgEnv + 9_Case_Studies_PE/Buck_Design/    E6    §8.1    V-block (equivalencia)
6    DAB + modulación con IA + TinyML    Fund. 9_Case_Studies_PE/DAB_Design/ + DAB_sample1.plecs    E9 (opcional)    §8.1    —
7    OOD en AI Diagnostics + ensembles/MDN    Fund. 4_Neural_Network/Multi_Modal_Distribution/, 3_Ensemble_Learning/    E10a    §8.2    A1 (lo implementa)
8    PINN de identificación de parámetros    Fund. 5_PIML/PINN/    E10b    §8.3    V0, A4
9    Pronóstico RUL de IGBT    Fund. 9_Case_Studies_PE/IGBT_Maintenance/    E11 (requiere decisión)    §8.4    —
10    Algorithm Guide    Sel. (solo como referencia de contenido)    E12    §8.6    —
—    Fundaciones (runs, jobs, componentes)    —    E0    §2, §6, §7, §9, §12, §13 Fase 0    —
—    Cierre    —    E13    §14, §20    P4
—    Regresor dc_motor (pendiente propio, no viene de los repos externos)    —    E14    §8.2 (estados), Monitoring existente    "Requieren una decisión" de TODO_CLAUDE_CODE.md; A1
Resumen y orden sugerido
Orden    ID    Tarea    Prioridad    Esfuerzo    Depende de    Requiere
1    E0    Fundaciones: contratos, run store, jobs, componentes, Experiment Runs    Alta    L    P0    —
2    E1    Entornos Gymnasium nativos (DC, VSC) + página Environments    Alta    M    E0    —
3    E2    PLECS por RPC: dataset + Plant Studio (My Plants, Dataset, Diagnostics)    Alta (P0 investigador)    L    E0    PLECS para la evidencia final
4    E3    Spike go/no-go: FMU de PLECS paso a paso    Alta (P0 investigador)    M    E1    fmpy (confirmar) + PLECS 5.0 con licencia
5    E6    Planta Buck nativa + página DC-DC Converters    Alta    M    E1    —
6    E4    PlecsGymEnv + asistente de 6 etapas + verificación y publicación    Alta (P0 investigador)    L    E3 = go, E1, E6    PLECS 5.0 con licencia
7    E5    Surrogate desde barridos PLECS + "Compare with SPADE"    Media    M    E2, E1    —
8    E7    Tuning Studio    Media    L    E0, E1    confirmar si se agregan pyswarms/pymoo
9    E8    RL Lab    Media    L    E1, E7 (ApprovalGate)    —
10    E10a    Estado "Unknown condition" en AI Diagnostics    Media    S    A1 cerrada    —
11    E10b    Pestaña Parameter Identification (PINN) en VSC / DPC    Media    M    V0    —
12    E12    Algorithm Guide    Baja    S    E1    —
13    E9    DAB por PLECS + modulación con IA (opcional)    Opcional    L    E2, licencia del .plecs    decisión
14    E11    Prognostics: RUL de IGBT con datos NASA    Opcional    M    E0    decisión + licencia
15    E14    Regresor dc_motor: detector de anomalías por residuo de pronóstico    Media    M–L    ninguna (independiente de E0–E12)    confirmar caso de uso y criterios
16    E13    Cierre: resumen, CI en verde, PR a dev    Alta    S    todas    —

Esfuerzo: S ≈ menos de media jornada · M ≈ 1 jornada · L ≈ 2 o más jornadas.

E0. Fundaciones
- [x] E0. Contratos, persistencia de runs, jobs y componentes base

Por qué: la spec exige que los trabajos largos (barridos PLECS, optimización, RL) sobrevivan a cambios de página y recargas. Hoy viz/adapters/session.py guarda como máximo 30 runs en st.session_state.

Hacer:

Inventario antes de editar (spec §21.1–2): firmas reales de viz/components/, viz/adapters/, viz/theme/. Deja la lista en el PR.
runs/schemas.py: PlantProfile, EnvironmentCapability, ExperimentSpec, Run, SignalSeries, ApprovalRecord y ErrorCode, como dataclasses con versión de esquema y validación (spec §2 y §3.7).
runs/store.py: SQLite en workspace/spade.db; artefactos en workspace/runs/<run_id>/. Cada Run guarda:
el hash de su config;
la semilla;
versiones y commit;
un comando reproducible, sin secretos ni rutas privadas.
jobs/: cola durable y worker en proceso aparte.
Estados (spec §6): queued → preparing → running → cancelling → completed | partially_completed | failed | cancelled.
resumable es un atributo, solo verdadero si existe un checkpoint real.
La ETA es nula hasta tener historial.
Lock por motor: un solo comando RPC a PLECS a la vez.
Componentes (spec §9): SourceBadge, ProvenanceFooter, ConnectionStatus, JobCard, RunComparison, ApprovalGate, FailurePanel, cada uno con los 5 estados (vacío, cargando, error, éxito, resultado negativo) y en ambos temas.
Página Workspace / Experiment Runs (spec §6, wireframe) y contador de jobs en el pie del sidebar (spec §1).
Navegación nueva en viz/app.py (spec §1). Las páginas aún no implementadas no se agregan: nada de enlaces a pantallas vacías.

Restricciones: no cambiar el comportamiento de páginas existentes salvo para agregar ProvenanceFooter. Las pruebas byte a byte de datagen no se tocan.

Pruebas: esquemas (ida y vuelta, versión); store (concurrencia básica); worker (lanzar, cancelar, matar el proceso y recuperar el estado real); AppTest de Experiment Runs vacía y con un job de prueba.

Criterio de cierre: se cumple el criterio de la spec §13 Fase 0. Un job de prueba (por ejemplo, una corrida corta de run_scenario) sobrevive al cambio de página y a la recarga, se cancela de forma honesta y deja logs y comando reproducible. Evidencia en docs/audit/expansion/e0_foundations.md.

E1. Entornos Gymnasium nativos
- [x] E1. gymnasium.Env para las plantas de SPADE + página Environments (funcionalidad #2)

Origen: Fund. 7_Reinforcement_Learning/RL_buck_control.ipynb (BuckAvgEnv). Se toma solo el diseño: observación normalizada, reset con condiciones aleatorias por semilla y recompensa por términos. Ese BuckAvgEnv ni siquiera hereda de gymnasium.Env; no se copia.

Hacer:

envs/native.py: DcMotorEnv y VscEnv, que envuelven DcMotorSystem y VscSystem sin modificar su física. Contrato Gymnasium completo (spec §3.7):
reset(seed, options) → (obs, info);
step(a) → (obs, reward, terminated, truncated, info);
espacios con unidades en info o en metadata.
La recompensa es configurable mediante una lista de términos (spec §3.4 etapa 5C): seguimiento, sobrecorriente, variación de la acción. El mismo esquema lo usarán E4 y E8.
envs/registry.py: gymnasium.register con IDs estables (SPADE/DcMotor-v0, SPADE/Vsc-v0) y un EnvironmentCapability por entorno.
Página Workspace / Environments (spec §8.5). "Copy Python snippet" solo para IDs que estén registrados de verdad.

Pruebas:

gymnasium.utils.env_checker.check_env pasa para cada entorno;
con acción igual a la del controlador PI de runner.py, el entorno reproduce bit a bit la trayectoria de run_scenario en el mismo escenario. Esto prueba que el envoltorio no altera la física.

Criterio de cierre: las pruebas en verde, con marcador simulation. La página muestra los dos entornos con su contrato.

E2. PLECS por RPC (rutas Dataset y Diagnostics)
- [ ] E2. Cliente RPC de PLECS, perfiles de planta y PLECS Plant Studio (rutas no Gym) (funcionalidad #1, parte a)

Origen: Fund. 8_PE_Simulation_Automation/PlecsAutomation/Data acquisition.ipynb: patrón plecs.load / plecs.simulate(model, {"ModelVars": …}) / plecs.close, con tiempo límite por corrida y registro de las que no convergen. Sin sus defectos (ver Decisión 6).

Hacer:

integrations/plecs/rpc_client.py, con xmlrpc.client de la librería estándar:
test_connection() devuelve versión y capacidades detectadas;
simulate(profile, model_vars, solver_opts) lee los Values/Time de las salidas (Outport) y aplica un tiempo límite por corrida;
lock por motor;
errores tipados (spec §3.4 etapa 2: Cannot reach RPC endpoint, Backend running on a different host, …).
integrations/plecs/manifest.py: manifiesto de interfaz (ModelVars y salidas, con unidades), declarado a mano y validado. No se promete leer automáticamente los bloques del circuito (spec §3.4 etapa 3).
PlantProfile con source_kind = plecs_rpc_batch, hash SHA-256 del .plecs y versionado; si el hash cambia, el perfil queda Stale.
Barrido como job: grilla o Latin hypercube con semilla. Las corridas fallidas quedan registradas y no se interpolan. La salida es un dataset en data/plecs/<profile>/ con su manifiesto.
Integración con datagen: plant_config_id = "plecs:<profile_id>" en Scenario, solo para la ruta dataset (el control vive dentro del circuito). Validación al estilo de S-03.
GUI (spec §3.14): PLECS Plant Studio con las pestañas My Plants, Create Plant (paso 1 Goal con las 3 tarjetas; solo la ruta Dataset queda habilitada en esta tarea), Connection & Diagnostics y la vista de barrido (spec §3.14.2). La tarjeta "Use PLECS as a Gymnasium plant" se ve, pero muestra Interactive stepping is not available with the detected backend. hasta que E4 esté cerrada.

Pruebas:

servidor XML-RPC falso (xmlrpc.server.SimpleXMLRPCServer en un hilo) que imita plecs.load, plecs.simulate y plecs.close, incluidos un tiempo límite, una no convergencia y una conexión caída;
regresión del defecto del notebook: cada fila del dataset lleva los parámetros de su propia combinación;
AppTest de la página sin PLECS configurado: todo navegable, sin excepciones.

Evidencia real: en una máquina con PLECS, correr un barrido pequeño (por ejemplo, un buck con 3×3 puntos) y guardar log, manifiesto y hash en docs/audit/plecs/e2_rpc_sweep.md. Si no hay PLECS disponible: "no ejecutado: sin PLECS".

Criterio de cierre: pruebas en verde sin PLECS; ruta Dataset funcional; nada en la GUI sugiere un Gym.

E3. Spike FMU paso a paso (go/no-go)
- [ ] E3. ¿Puede SPADE controlar un circuito PLECS paso a paso vía FMU? (funcionalidad #1, parte c)

Requiere confirmación: agregar fmpy (FMI 2.0/3.0 en Python) como dependencia opcional ([plecs]). También requiere una máquina con PLECS 5.0 y licencia. Sin eso, la tarea queda bloqueada y se documenta así.

Hacer:

El circuito mínimo es el buck de E6, construido en PLECS como subsistema atómico. Su entrada es el ciclo de trabajo (duty) y sus salidas son v_out e i_L.
Exportar una FMU de co-simulación FMI 3.0 (y 2.0 para comparar) y activar el endpoint en las preferencias de PLECS.
integrations/plecs/fmu_adapter.py: instanciar, inicializar, doStep(Ts) × N con una secuencia fija de acciones, leer salidas, reiniciar y cerrar.
Medir y reportar:
si la acción afecta la salida;
que el tiempo sea monótono;
si el reinicio es reproducible (dos episodios con la misma semilla y las mismas acciones son idénticos);
el tiempo de pared por paso, p50 y p99;
el comportamiento ante NaN y ante cierre del endpoint;
la comparación contra el buck nativo de E6 con la misma secuencia de acciones.
Si existe licencia de PLECS Coder, repetir con una FMU de Model Exchange (o la que FMPy pueda ejecutar como co-simulación) y comprobar que corre sin PLECS. Si es así, puede generar referencias para CI.
Documentar la opción de respaldo con el bloque DLL (Decisión 3) solo como texto.

Entregable: docs/audit/plecs/fmu_stepping_spike.{md,json} con veredicto GO / NO-GO / GO parcial (por ejemplo, "GO solo con Ts ≥ X ms por latencia") y comandos reproducibles.

Criterio de cierre: veredicto escrito y respaldado por un log. Un NO-GO es un resultado válido: se documenta como hallazgo y E4 queda bloqueada.

E4. PLECS como planta Gym (ruta principal del investigador)
- [ ] E4. PlecsGymEnv + asistente de 6 etapas + verificación + publicación (funcionalidad #1, completa)

Solo si E3 = GO.

Hacer:

envs/plecs_gym_env.py: PlecsGymEnv(gym.Env) sobre fmu_adapter, con el contrato de la spec §3.7.
step() avanza el mismo estado de la planta de t a t+Ts;
las acciones fuera de rango se rechazan o se recortan con registro;
terminated y truncated se separan;
las fallas del solver nunca se cuentan como éxito.
Reusar el esquema de recompensa y espacios de E1.
GUI: las etapas 2 a 6 de la spec §3.4, el panel Environment readiness, la máquina de estados (spec §3.8: DRAFT → … → PUBLISHED, más STALE, FAILED, UNSUPPORTED) y el microcopy de la spec §3.9.
Verificación (spec §3.4 etapa 6, los 12 checks) como job. Publish to Environments se habilita solo si la verificación está vigente para ese hash y esa configuración.
Al publicar, el entorno aparece en Environments (E1) con SIMULATED · PLECS, su hash y su verificación.
Prueba de punta a punta del ejemplo de la spec §3.6 (buck): publicar y correr PI de SPADE y una política aleatoria sobre la planta PLECS. Si la regulación con PI se ve razonable, puedes agregar el escenario a la comparación PI vs RL de E8.

Pruebas:

unitarias con un adaptador FMU falso;
check_env;
de integración marcadas slow + plecs, que se saltan si no hay PLECS. Nunca se marcan como aprobadas sin haberse ejecutado.

Criterio de cierre: se cumplen los criterios de la spec §3.11 y §20 "PLECS: creación de Gym". Evidencia en docs/audit/plecs/e4_plecs_gym.md, con el log de la máquina con PLECS.

E5. Surrogate y comparación SPADE vs PLECS
- [ ] E5. Planta sustituta desde barridos PLECS + vista "Compare with SPADE" (funcionalidad #1, parte b)

Hacer:

envs/surrogate_env.py: ajustar un modelo discreto a partir de un dataset de E2. Opciones, de la más simple a la más compleja:
identificación lineal (ZOH, como en V0);
una red Keras pequeña.
El split es por condición, nunca por ventana. Se reporta el error por canal y por condición, y el dominio de validez. Fuera de ese dominio se advierte.
Publicar como SURROGATE-Gym con su propia etiqueta de procedencia.
Vista Compare with SPADE (spec §3.14.4): la misma excitación sobre la planta nativa y sobre PLECS (o el sustituto), con ejes alineados, error por condición, casos fallidos visibles y sin afirmar equivalencia si las condiciones no son comparables.
Caso de validación: el buck nativo de E6 frente al buck de PLECS. Si hay un VSC en PLECS, también el VSC identificado de V0 frente a un VSC conmutado. Esto valida la fidelidad del modelo, no una máquina real.

Criterio de cierre: se cumplen la spec §3.14.3 y §20 "dataset/surrogate/comparación". Reporte en docs/audit/plecs/e5_surrogate_and_comparison.md.

E6. Planta Buck nativa
- [x] E6. sim/buck_system.py + página Simulation / DC-DC Converters (pestaña Buck) (funcionalidad #5)

Origen: Fund. BuckAvgEnv (modelo promediado: L, C, R, Vin, duty) y 9_Case_Studies_PE/Buck_Design/. Se reimplementa; no se copia.

Hacer:

BuckSystem con reset()/simulate(duty) (la misma interfaz que VscSystem) y τ configurable. Documentar si es promediado o conmutado.
Controlador PI de voltaje (reusar control/classical si aplica) y entorno SPADE/Buck-v0 en E1.
Agregar plant_config_id = "buck_v1" en Scenario y runner.py, sin cambiar las filas que generan las plantas existentes.
GUI (spec §8.1): página DC-DC Converters con la pestaña Buck (la de DAB se agrega solo con E9), siguiendo el patrón de Generate de las páginas existentes.

Pruebas:

contra la solución analítica en régimen permanente (Vout = D·Vin en conducción continua) y contra solve_ivp para el transitorio, con el error medido en el docstring;
las pruebas byte a byte de datagen siguen sin cambios.

Criterio de cierre: pruebas simulation en verde y página funcional. Este buck es la referencia para E3, E4 y E5.

E7. Tuning Studio
- [ ] E7. Optimización de sintonía de controladores (funcionalidad #3)

Origen: Fund. 1_MHA/Single_Objective_MHA/ (PSO con pyswarms), 1_MHA/Multi_Objective_MHA/ (pymoo) y 2_Classic_ML/gaussian_process_bayesian_optimization.ipynb (scikit-learn).

Dependencias:

la optimización bayesiana usa scikit-learn (GaussianProcessRegressor), que ya está en pyproject.toml;
PSO: preferir una implementación propia en numpy (unas 80 líneas, con pruebas sobre funciones de referencia);
multiobjetivo: preferir NSGA-II propio y simple, o pedir confirmación para pymoo.

Hacer:

optim/objectives.py: la función objetivo corre run_scenario (o un entorno de E1) sobre una grilla de sintonía y devuelve métricas (sobreimpulso, RMSE, tiempo de establecimiento, CPU). La grilla de evaluación es distinta y se fija antes de empezar; si se reusa la de sintonía, la GUI muestra The evaluation reused tuning conditions….
optim/pso.py, optim/bayes.py, optim/multiobjective.py: con semilla y como jobs de E0.
GUI (spec §4 completa): pasos 1 a 5, curva del mejor valor contra número de evaluaciones y frente de Pareto (con selección por clic, no solo hover). Save candidate no cambia nada; Use as default pasa por ApprovalGate y requiere además la confirmación de Luciano fuera de la app (regla de CLAUDE.md).
Primer caso de uso: el escalón pequeño de S-04 en el PI del motor DC. Coordinar con C2 de TODO_CLAUDE_CODE.md, que ataca el mismo síntoma con ponderación del setpoint.

Criterio de cierre: se cumple la spec §13 Fase 3 en la parte de optimización. Reporte en docs/audit/expansion/e7_tuning_s04.md, incluido el caso en que no se mejora la sintonía actual.

E8. RL Lab
- [ ] E8. DQN y DDPG en Keras + editor de recompensa + benchmark (funcionalidad #4)

Origen: Fund. 7_Reinforcement_Learning/RL_buck_control.ipynb (DQN: replay, target network, acciones discretas Δduty) y DDPG_buck_control.ipynb (DDPG: ruido Ornstein-Uhlenbeck, soft updates, penalización de la suavidad del duty). Están en PyTorch, así que se reimplementan en Keras.

Hacer:

control/rl/dqn.py y control/rl/ddpg.py, con semilla, checkpoints y entrenamiento como job de E0.
control/rl/policy_controller.py: envuelve una política entrenada con la misma interfaz de controlador que PI, MPC y DPC, para compararlos en run_scenario.
DQN solo con acciones discretas. Si se discretiza una acción continua, la transformación se guarda (spec §5 paso 1).
GUI (spec §5): entorno → recompensa (componente RewardEditor, con vista previa solo tras correr un episodio real) → entrenamiento (recompensa por episodio, pérdidas, violaciones, reproducción de episodios ejecutados) → benchmark contra PI, MPC y DPC en la misma grilla.
Primer caso: el buck de E6. Segundo caso: el VSC frente a DPC v3 y v4, en la grilla de measure_dpc_rmse_by_r_range.py.

Criterio de cierre:

reproducibilidad: misma semilla, misma curva de recompensa, dentro de la tolerancia declarada;
benchmark completo, publicado aunque RL pierda;
ninguna política queda como default sin aprobación.

Reporte en docs/audit/expansion/e8_rl_benchmark.md.

E9. DAB (opcional, requiere decisión)
- [ ] E9. DAB como planta PLECS + modulación TPS sugerida por IA (funcionalidad #6)

Antes de empezar:

confirmar la licencia de DAB_sample1.plecs (podría estar bajo la CC BY-NC del contenido que no es código);
confirmar con Luciano que se quiere un tercer convertidor.

Hacer:

El DAB entra como perfil PLECS por E2 (dataset) y, si E3 = GO, por E4.
Modelo de modulación inspirado en one_stop_AI_DAB_modulation.ipynb y TinyML.ipynb, reimplementado en Keras, por la vía existente de YAML → train_model.py → registro → TFLite. Dominio nuevo dab, sin mezclarlo con otros.
GUI (spec §8.1): pestaña DAB con selector de estrategia y diagrama de desfases. La sugerencia de la IA es AI SUGGESTION — REQUIRES VERIFICATION, y Compare suggested vs baseline corre simulaciones nuevas.

Criterio de cierre: a definir con Luciano cuando apruebe la tarea.

E10. IA avanzada en la GUI
- [x] E10a. Estado "Unknown condition" en AI Diagnostics (funcionalidad #7)

Depende de A1 (TODO_CLAUDE_CODE.md), que implementa ai/ood.py. Esta tarea no implementa el detector, salvo que se quiera agregar un ensemble o una red de densidad de mezcla (MDN), inspirados en Fund. 4_Neural_Network/Multi_Modal_Distribution/, como método candidato adicional dentro del reporte de A1 y bajo su misma regla anti-fuga.

Hacer (spec §8.2): tres salidas distinguibles: Known condition, Uncertain — low confidence y Unknown condition — REQUIRES VERIFICATION, con la regla de decisión de la spec. Si el modelo no tiene un detector validado, se muestra OOD detection unavailable y nunca un puntaje inventado. La hipótesis del agente de simulación se presenta como pista secundaria.

Criterio de cierre: pruebas AppTest de los tres estados + Input invalid + "detector ausente".

- [x] E10b. Pestaña Parameter Identification (PINN) en VSC / DPC (funcionalidad #8)

Origen: Fund. 5_PIML/PINN/pinn_ode.ipynb (PyTorch, se reimplementa en Keras). Depende de V0 (discretize_zoh y los parámetros L, R_f, C) y alimenta a A4.

Hacer:

ai/pinn/identification.py: estimar L, C y R_f desde ventanas de v_c, i_f y v_o, con un intervalo de incertidumbre cuyo método se declara (bootstrap o ensemble).
Validarlo sobre simulaciones con parámetros conocidos (incluido C degradado vía discretize_zoh) y reportar el error de recuperación.
GUI (spec §8.3) con el texto fijo Parameter estimate — not a validated fault diagnosis.

Criterio de cierre: reporte docs/audit/ai/pinn_identification.{md,json} con el error de recuperación por parámetro y por nivel de ruido.

E11. Prognostics: RUL de IGBT (opcional, requiere decisión)
- [ ] E11. Pronóstico probabilístico de vida útil de IGBT con datos NASA (funcionalidad #9)

Por qué requiere decisión:

son los primeros datos medidos de SPADE, pero de otro componente (IGBT), no de rodamientos ni del VSC promediado;
abre un dominio nuevo (igbt_aging) y no valida ninguna afirmación del diagnóstico de rodamientos.

Antes de empezar:

verificar la licencia y la procedencia del dataset de envejecimiento acelerado de IGBT de la NASA (https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging) y del .mat incluido en Fund.;
descargar a data/upstream/fund/igbt/;
documentar la fuente en DATA.md.

Hacer:

Una BiLSTM con salida gaussiana (media y log-varianza, pérdida NLL), reimplementada en Keras desde rul_prediction.ipynb, como YAML en configs/regressors/.
Split por dispositivo o serie, sin mezclar ventanas de una misma unidad entre train y test.
Reportar la cobertura empírica del intervalo del 90 %. Si no puede evaluarse, la GUI dice Coverage not validated.
GUI (spec §8.4) con la etiqueta MEASURED · EXTERNAL DATASET y el texto This dataset is separate from SPADE's simulated plant diagnostics.

Criterio de cierre: reporte docs/audit/ai/igbt_rul.{md,json} y su audit_findings.json con los límites conocidos.

E12. Algorithm Guide
- [ ] E12. Página Help / Algorithm Guide (funcionalidad #10)

Origen: Sel. (solo la idea del asistente What-Which-How; no se copia código porque no hay licencia).

Hacer (spec §8.6): tres preguntas (fase del ciclo de vida, tarea, origen de los datos) → familia recomendada, con su porqué, supuestos y límites → enlace a la página de SPADE que la implementa, o Not available in this version. La tabla de recomendaciones vive en un YAML del repo, con cita al paper de Li et al. (IEEE TIE, 2026).

Criterio de cierre: AppTest recorre todas las combinaciones sin enlaces rotos.

E14. Regresor para dc_motor: detector de anomalías por residuo
- [ ] E14.0. Confirmar el caso de uso y los criterios (requiere decisión de Luciano; no se programa antes)
- [ ] E14.1–E14.6. Implementación (solo después de E14.0)

Contexto:

dc_motor solo tiene clasificadores; los regresores existen solo en vsc_dpc (pc_full, rpi5_tcn y esp32_tiny).
Este regresor estaba bloqueado por falta de un caso de uso, no por un problema técnico.
Se trabaja en el worktree SPADE-dev (rama dev, con su propio .venv).
La línea base que informó Luciano es de 760 pruebas aprobadas. No la verifiqué en esta sesión; Claude Code debe re-medirla en P0.
Caso de uso recomendado: 1, detección de anomalías por residuo

Comparación de las tres opciones del contexto, a partir de lo que SPADE ya hace:

Opción    Valor    Por qué
1. Residuo de pronóstico    Alto    Aporta algo que hoy no existe: un detector que no depende de las clases del clasificador y que no necesita el simulador en tiempo de inferencia. El clasificador falla con fallas fuera de su vocabulario (AI-N12; según el contexto, el detector OOD tampoco detectó bien la falla de orden 1,3). El agente de simulación sí detecta fallas, pero necesita el simulador y el punto de operación, así que no corre en RPi5 ni en ESP32; un pronosticador pequeño sí.
2. Pronóstico de corriente para protección    Bajo    En SPADE el lazo PI en cascada ya limita la corriente; el motor llega a la cota de corriente limitada (S-07, VALIDATION_SUMMARY.md §2). Anticipar un sobrepaso que el controlador ya impide aporta poco en esta planta simulada, y las reglas del gateway ya cubren los umbrales.
3. Sensor virtual    Bajo    Par desde corriente: en un motor DC de excitación permanente el par es proporcional a la corriente (ψ·i), así que una red no agrega nada a la física. Vibración desde corriente y rpm: la vibración es sintética y la genera un modelo de SPADE; la red aprendería ese sintetizador, lo que no valida nada (es un razonamiento circular).
Qué se pronostica, con qué entradas y a qué horizonte
Entrenamiento: solo con ventanas sanas (normal) del dataset de grilla (generate_diagnosis_grid_dataset.py). El modelo aprende la dinámica sana y su error crece cuando aparece el rizado de par de una falla.
Objetivos: current_r y rpm, con pronóstico directo multihorizonte (build_direct_forecast_windows). La falla de rodamiento entra a esos canales por el rizado de par (vía MCSA). La vibración no es objetivo: tiene ruido aleatorio por semilla y no es pronosticable.
Entradas: el pasado de current_r, rpm y voltage_v. Hay un riesgo de diseño que hay que medir, no suponer: en lazo cerrado, el PI compensa el rizado moviendo voltage_v. Con voltage_v como entrada, el modelo podría "explicar" el rizado y ocultar la falla. Por eso se entrenan dos variantes, con y sin voltage_v, y se reportan ambas.
Ventana y horizonte: se parte de los valores de pc_full.yaml (ventana de entrada y horizon) y se ajustan a la dinámica del motor:
el horizonte debe ser más corto que medio período de la frecuencia característica de falla más alta de la grilla;
Claude Code calcula esa frecuencia desde sim/vibration y datagen/fault_injection.py para las 7 velocidades, y la anota en el reporte.
Si conviene decimar (por ejemplo, de 10 kHz a 2 kHz con un filtro antialias), las frecuencias de falla deben quedar bajo Nyquist.
Ninguno de estos valores se elige mirando ventanas con falla.
Puntaje de anomalía: RMSE del residuo por ventana, normalizado por canal con estadísticas de validación sana.
El umbral es el cuantil 99 del puntaje en validación sana, sin mirar fallas: es el mismo criterio de "no ajustar con datos de prueba ni OOD" de CLAUDE.md.
El cuantil exacto lo confirma Luciano en E14.0.
Métricas de éxito (fijadas antes de mirar resultados)

Todo se evalúa sobre ventanas simuladas, en condiciones excluidas del entrenamiento: velocidades no vistas (115, 190 y 250 rad/s, igual que en la evaluación del clasificador), severidad ×0,6, 100 % de ruido y deriva de planta.

#    Criterio    Meta propuesta    Si no se cumple
1    Pronóstico sano frente a la referencia de persistencia (split por condición)    RMSE ≤ 0,8 × persistence_test_rmse, por canal    Detener: un pronosticador que no le gana a la persistencia no sirve como detector. Se documenta como hallazgo
2    Detección de fallas conocidas (outer_race, inner_race, severidad ≥ ×1)    AUROC ≥ 0,9    Si AUROC < 0,7: resultado negativo, documentado como hallazgo AI-N1x; no se integra a la GUI
3    Fallas fuera del vocabulario del clasificador (bola, jaula, orden 1,3)    Reportar AUROC y tasa de detección con el umbral fijo, sin meta: es el valor diferencial, pero no se ajusta nada por ella    —
4    Falsas alarmas en sano bajo corrimientos benignos (ruido 100 %, deriva, velocidades no vistas)    ≤ 1 % de ventanas    Reportar por condición; nunca subir el umbral mirando estos datos
5    Arranque y escalones de referencia    Reportar la tasa de falsas alarmas en transitorios por separado    Si es alta: proponer filtrar por régimen (por ejemplo, no puntuar durante N ms tras un cambio de referencia) como una decisión aparte
6    Borde (si se exporta)    Paridad TFLite frente a Keras (report_tflite_parity.py), costo en host y 0 operaciones Flex    No promover a borde

Las metas numéricas de las filas 1, 2 y 4 son una propuesta: Luciano las confirma o ajusta en E14.0, antes de entrenar.

Implementación (después de E14.0)
E14.1 Datos:
extraer las adquisiciones normal del dataset de grilla;
test_groups por condición: velocidades no vistas y combinaciones de ruido y deriva;
las ventanas con falla se usan solo en la evaluación.
E14.2 Modelos:
YAML nuevos configs/regressors/dc_motor_residual_{pc,rpi5,esp32}.yaml (dominio dc_motor, sin mezclar filas de vsc_dpc);
entrenamiento con train_model.py;
RPi5 y ESP32 con --distill --teacher-run <pc>;
para RPi5, usar TCN (recurrent_type: none) para evitar las operaciones Flex.
E14.3 Puntaje:
ai/residual.py con fit_threshold(val_healthy) y score(window);
guarda el umbral, el cuantil y las estadísticas en metrics.json["residual_detector"].
E14.4 Evaluación:
experiments/report_residual_anomaly.py, que escribe docs/audit/ai/dc_motor_residual_anomaly.{md,json} con los 6 criterios, las dos variantes (con y sin voltage_v) y la comparación con el agente de simulación en los mismos escenarios;
si falla un criterio de corte, el entregable es el reporte del fallo.
E14.5 Interfaz (solo si se cumplen los criterios 1 y 2):
Monitoring: el residuo entra como cuarto detector del consenso ("Forecast residual"), con la etiqueta SIMULATED. Por la regla vigente, no dispara una alerta por sí solo, igual que el clasificador. Cambiar esa regla es cambiar un default y requiere confirmación.
AI Diagnostics: panel "Forecast residual", con pronóstico y señal real superpuestos, el puntaje frente a su umbral y la etiqueta AI SUGGESTION — REQUIRES VERIFICATION. El texto aclara que un residuo alto indica "comportamiento distinto al sano aprendido", no una clase de falla.
Model Registry: el bloque regressor de dc_motor aparece en los tres niveles.
E14.6 Cierre:
audit_findings.json del run con sus límites conocidos (solo simulado; transitorios; efecto de voltage_v);
actualizar VALIDATION_SUMMARY.md §7 y AI_TECHNICAL_AUDIT.md;
correr la suite completa y el smoke test;
no promover sin confirmación (promote_run.py).
Pruebas
Unitarias: el umbral se calcula solo con datos sanos de validación (la prueba falla si fit_threshold recibe filas con falla); el split por condición no comparte grupos entre train y test.
De regresión: el residuo de una ventana sana de entrenamiento queda bajo el umbral, y el de una ventana con rizado sintético grande queda sobre él.
AppTest: los paneles de Monitoring y AI Diagnostics en tres casos: sin modelo registrado (Residual detector not available), sano y anómalo.
E13. Cierre
- [ ] E13. Resumen consolidado y PR
Actualizar docs/VALIDATION_SUMMARY.md con:
una sección "Expansion (E0–E12)";
la tabla de pruebas;
qué se ejecutó con PLECS y qué no;
los hallazgos nuevos.
Recorrer la matriz de la spec §14 y los criterios de §20, marcando cada fila como aprobada, fallida o no ejecutada (con el motivo).
Capturas de cada pantalla nueva en sus 5 estados, en tema claro y oscuro, guardadas en docs/screenshots/expansion/.
CI en verde en dev. El PR de dev a main solo con confirmación de Luciano.
Fuera de alcance de esta lista
Ejecutar algo en hardware físico, o calificar como validación de una máquina real cualquier resultado de PLECS o del sustituto.
Automatizar LTspice o Simulink (Fund. 8_PE_Simulation_Automation/LTspiceAutomation, SimulinkAutomation). El cliente de E2 debería quedar con una interfaz que permita agregarlos después.
Modelado de pérdidas magnéticas (MagNet) y campo térmico de capacitores (Fund. 9_Case_Studies_PE/Magnetic_Modeling, 4_Neural_Network/Field_Data): poco relacionados con accionamientos.
Agentes LLM (Fund. 6_Agentic_AI): solo hay un README, sin código.
Redis, RQ o un despliegue multiusuario del job runner.
Decisiones que necesita Luciano
Decisión    Afecta a    Cuándo
Agregar fmpy como dependencia opcional [plecs]    E3, E4    antes de E3
Acceso a una máquina con PLECS 5.0 + licencia (y ¿PLECS Coder?)    E2 (evidencia), E3, E4    antes de E3
pyswarms/pymoo o implementación propia    E7    antes de E7
Hacer E9 (DAB) y la licencia de DAB_sample1.plecs    E9    después de E2
Hacer E11 (dominio IGBT con datos medidos)    E11    cuando quieras
Usar como default cualquier candidato de E7 o E8    E7, E8    al cerrar cada una
Caso de uso del regresor dc_motor (recomendado: residuo) y metas de E14 (cuantil del umbral, AUROC ≥ 0,9, falsas alarmas ≤ 1 %, RMSE ≤ 0,8 × persistencia)    E14    antes de E14.1
Integrar el residuo al consenso de Monitoring y, aparte, si puede disparar alertas por sí solo    E14.5    al cerrar E14.4
Promover los regresores dc_motor    E14.6    al cerrar E14
