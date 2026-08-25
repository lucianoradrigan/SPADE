# Patch 7: definición de la Fase D — modelos sobre datos del dominio DPC/VSC

**Estado: definido, no implementado.** Este documento especifica el alcance de D.1 y D.2; ninguna
de las dos sub-fases tiene código, datos derivados, ni resultados todavía. No confundir con
`docs/macro_fase_B1_dpc.md`/`macro_fase_B2_dpc_deployment.md`, que sí documentan trabajo ya hecho
y validado.

## Por qué existe esta fase

Patch 5 (`docs/patch5_alcance_macrofase_B.md`) retiró el objetivo de comparar DPC contra PI/MPC
"bajo las mismas condiciones" — no tiene sentido físico, son plantas distintas. Eso deja abierta
la pregunta de qué se hace con el trabajo de la Macro-fase B más allá de haber portado y validado
el controlador. La respuesta: los datos que B ya generó (evaluaciones de v1/v2/v3, los rollouts de
augmentación estilo DAgger) sirven de insumo a dos modelos propios del dominio VSC, análogos en
espíritu a los de `paper_federative` pero sin reutilizar directamente su dominio físico (vibración
de rodamiento).

## D.1 — Regresor de degradación de seguimiento

**Depende de:** Macro-fase B (datos ya generados, sin nueva simulación). **Prioridad:** ejecutar
primero — costo bajo, no bloquea con nada pendiente.

- **Tarea:** predecir el error de seguimiento futuro (`|v_c − v_ref|`, horizonte corto) a partir de
  una ventana de contexto de `(v_ref, v_c, i_f)`.
- **Datos:** derivados de las trayectorias ya generadas en B (evaluaciones de checkpoints v1/v2/v3
  y los 8 rollouts de augmentación de `finetune_dpc_closed_loop.py`) — no requiere correr una
  nueva simulación, solo re-empaquetar lo que ya existe en `configs/`/los scripts de
  `experiments/`.
- **Arquitectura:** adaptar `build_envelope_forecaster()` de `paper_federative` (contexto pasado →
  magnitud futura), sustituyendo la envolvente de vibración por el error de seguimiento DPC. Mismo
  patrón arquitectónico, dominio de entrada distinto.
- **Entregable:** `models/regressors/dpc_tracking_forecaster.py`, dataset derivado documentado
  (sin nueva simulación), evaluación con holdout **por rollout completo** (no por muestra
  individual — evita que el holdout comparta trayectoria con el train, el mismo tipo de fuga que
  ya se evitó en la Macro-fase A al separar por bearing en vez de por muestra).

## D.2 — Clasificador de fallas de convertidor

**Depende de:** D.1 (ejecutar después) + veredicto del Paso 0 de separabilidad (condicionada, no
automática).

- **Alcance:** degradación de capacitor del filtro LC (ESR y capacitancia efectivos respecto al
  valor nominal `C = 15µF`). **Fuera de alcance:** fallas de conmutación — `VscSystem` es un
  modelo promediado discreto (`Adf`/`Bdf`), sin representación de dispositivos de conmutación
  individuales, no puede representar ese tipo de falla.
- **Paso 0 (obligatorio, no omitir):** parametrizar `VscSystem` con `C_efectivo` y `ESR_efectivo`
  variables; generar un conjunto reducido de corridas bajo condición nominal y 2-3 niveles de
  degradación; calcular separabilidad (AUC) sobre features de ventana de `(v_c, v_ref, i_f)` —
  mismo criterio de aceptación que ya se usó para el Módulo B de vibración
  (`docs/patch2_retiro_modulo_C.md`), no una decisión nueva de método.
  - **Si la separabilidad resulta baja (AUC cercano a 0.5): detener D.2 en este punto y
    documentar como limitación** — no proceder a construir un clasificador sobre una señal que el
    propio Paso 0 muestra que no es separable (el mismo criterio que ya retiró el Módulo C, ver
    `docs/patch2_retiro_modulo_C.md`).
- **Advertencia epistemológica, de consignación obligatoria si D.2 avanza:** no existe dataset
  real de referencia para degradación de capacitores, a diferencia de Paderborn para rodamientos
  (Macro-fase A). La validación de cualquier clasificador de esta sub-fase es exclusivamente
  contra la propia definición sintética de la falla (los niveles de `C_efectivo`/`ESR_efectivo`
  que el Paso 0 generó), no contra evidencia empírica independiente. Debe documentarse con el
  mismo nivel de explicitud que las limitaciones ya registradas para el Módulo B
  (`docs/addendum_vibracion_v1.md` Sec. 3/9, `docs/patch2_retiro_modulo_C.md`).
- **Entregable (condicionado al resultado del Paso 0):** `sim/vsc_faults.py`, dataset etiquetado
  sano/degradado, clasificador (arquitectura adaptada de `sensor_dscnn.py` o
  `gateway_resnet_se.py`), documentación de la limitación epistemológica de arriba.

## Ver también

- `docs/patch5_alcance_macrofase_B.md` — por qué DPC no se compara contra PI/MPC, y por qué esta
  fase existe en su lugar.
- `docs/macro_fase_B2_dpc_deployment.md` — de dónde salen los datos que D.1 reutiliza.
- `docs/patch2_retiro_modulo_C.md` — precedente del criterio de aceptación por separabilidad (AUC)
  que el Paso 0 de D.2 reutiliza, y precedente de "detener y documentar" cuando la señal no separa.
