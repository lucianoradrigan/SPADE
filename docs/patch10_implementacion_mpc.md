# Patch 10 — Implementación de MPC (resuelve discrepancia Fase A)

**Contexto de origen:** revisión crítica del repositorio, tarea de prioridad Media #4 de un
documento externo de "tareas de corrección priorizadas". `INSTRUCTIONS.md` describía la Fase A
como "motor DC + PI/MPC + módulo de vibración + inyección de fallas" y la marcaba "Completada",
pero `control/mpc/` solo contenía un `.gitkeep` — MPC nunca existió. Se decidió (respuesta
explícita del usuario, no una decisión unilateral) implementar MPC en vez de retirar la mención.

## 1. Qué se implementó

`src/driveflow/control/mpc/controller.py::MpcController` — un MPC lineal nativo (no un port de
nada) para `DcMotorSystem`, con la misma interfaz que `PICascadeController`
(`reset()`/`control(state, omega_ref)`/`control_torque(state, torque_ref)`), de modo que es un
`controller_type` intercambiable con `"PI"` sobre el mismo `plant_config_id="dc_perm_ex_v1"`
(a diferencia de DPC, que opera sobre un dominio físico completamente distinto — ver Sección 4).

**Modelo interno:** mismo modelo físico que ya usa el lazo de velocidad de `PICascadeController`
(ver docstring de ese módulo) — el lazo de corriente es mucho más rápido que la constante de
tiempo mecánica, así que la fricción (coeficientes `a`/`b`/`c` de `PolynomialStaticLoad`) se trata
como perturbación rechazada por realimentación, no como parte del modelo de diseño:

```
d(i)/dt     = (-r_a*i - psi_e*omega + u) / l_a
d(omega)/dt = (psi_e*i) / j_total
```

Discretizado por Euler hacia adelante al mismo `tau` que usa la planta real (`EulerSolver`).

**Formulación:** QP lineal con horizonte deslizante (`horizon=20` pasos, ~1.7 constantes de tiempo
eléctricas) — predicción completa de trayectoria vía matrices `Sx`/`Su` precalculadas, costo
cuadrático de seguimiento (`q_track`) + esfuerzo de control (`r_effort`) + una penalización suave
(cuadrática más allá del límite) sobre la corriente predicha (`q_current_penalty`), ya que a
diferencia de PI (que recorta explícitamente su `i_ref`), este controlador nunca forma un `i_ref`
explícito — la penalización actúa directamente sobre el estado predicho. Restricción dura de caja
en `u ∈ [0, u_sup]` (límite físico real del convertidor). Resuelto con `scipy.optimize.minimize`
(L-BFGS-B, gradiente analítico, warm-start entre pasos).

## 2. Resultado de calibración (verificado contra la planta real, no solo unitario)

| Referencia ω (rad/s) | error relativo asentado | máx \|i\| (A, límite 210A) |
|---|---|---|
| 50 | -0.04% | 178.6 |
| 100 | -0.04% | 178.8 |
| 150 | -0.04% | 178.8 |
| 250 | -0.04% | 179.1 |

Seguimiento de torque (`control_torque`): error relativo ~0.00% en 5/10/15 Nm. Ambos mejores que
la tolerancia de PI (rel=0.05-0.08) ya establecida en `tests/test_pi_controller.py`.

## 3. Hallazgo de rendimiento no trivial: threading de BLAS

Una primera versión sin mitigación tardaba **varios segundos por paso de control** (no ms) —
confirmado con `ps aux` mostrando ~660% CPU en operaciones matriciales de 20x20. Causa: numpy/BLAS
lanza multi-threading automático incluso para matrices minúsculas, y el overhead de creación/
sincronización de threads por llamada domina sobre el cómputo real. Solucionado envolviendo el
`scipy.optimize.minimize` en `threadpoolctl.threadpool_limits(limits=1)` (ya presente
transitivamente vía `scikit-learn`; se agregó como dependencia explícita en `pyproject.toml` por
ser ahora una dependencia directa real, no solo transitiva) — de varios segundos a ~7ms/paso, sin
necesidad de variables de entorno externas (`OMP_NUM_THREADS`, etc.), que no serían controlables
por quien use el paquete en producción/dashboard.

## 4. Efecto sobre Patch 5 (¿se reabre la comparación DPC vs. PI/MPC?)

**No.** Patch 5 concluyó que DPC (sobre un VSC) y PI/MPC (sobre el motor DC) no son comparables
bajo las mismas condiciones porque operan en dominios físicos distintos, sin variables de estado,
planta ni objetivo de control en común. Implementar un MPC real para el motor DC no cambia el
dominio de DPC ni crea ninguna variable compartida entre ambos — la conclusión de Patch 5 se
reconfirma, ahora con un MPC real en mano en vez de una mención sin implementar.

Lo que SÍ se vuelve posible por primera vez: una comparación PI vs. MPC real, con ambos sobre el
mismo `DcMotorSystem`, mismo estado, misma acción. `tests/test_mpc_controller.py::TestPiVsMpcComparison`
es esa comparación (antes no podía existir porque uno de los dos lados no estaba implementado).

## 5. Qué se actualizó para reflejar la realidad

- `datagen/scenario.py`: `_VALID_PAIRS` ahora incluye `("MPC", "dc_perm_ex_v1")`; docstring de
  `controller_type` actualizado.
- `datagen/runner.py`: `_run_dc_motor_scenario` despacha por `scenario.controller_type` vía
  `_DC_MOTOR_CONTROLLERS = {"PI": PICascadeController, "MPC": MpcController}` en vez de
  instanciar `PICascadeController` de forma fija.
- `tests/test_datagen.py`: `test_rejects_unimplemented_controller` ya no podía usar `"MPC"` como
  ejemplo de controlador no implementado (dejó de serlo) — se reemplazó por un nombre inventado, y
  se agregaron `test_accepts_mpc_dc_motor_pairing`/`test_rejects_mpc_vsc_pairing` +
  `TestRunScenarioMpc` (extremo a extremo vía `run_scenario`).
- `tests/test_mpc_controller.py`: nuevo, misma estructura/tolerancias que `test_pi_controller.py`
  (verificación de lazo cerrado contra la planta real, no solo "no explota").
- `README.md`: tabla de sistemas y layout de repositorio actualizados.

## 6. Qué NO se hizo (alcance explícitamente fuera de este patch)

- No se agregó selector de controlador PI/MPC en el dashboard (`viz/dashboard.py`) — la Fase A del
  dashboard sigue usando PI únicamente en la UI interactiva; MPC es utilizable hoy vía
  `Scenario`/`run_scenario`/`experiments/generate_diagnosis_dataset.py`, no desde la UI. Dejado
  como ítem de UI separado, no bloquea nada de lo anterior.
- No se implementó restricción dura de corriente (solo penalización suave) — una restricción dura
  requeriría un solver de programación cuadrática con restricciones lineales generales (p.ej.
  `trust-constr`), significativamente más lento por paso; la penalización suave ya mantiene la
  corriente bajo el margen de seguridad en las condiciones probadas (Sección 2), documentado como
  limitación de diseño, no como omisión.
