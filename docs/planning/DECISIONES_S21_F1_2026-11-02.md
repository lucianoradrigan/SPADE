# SPADE · Respuesta a `ESTADO_FASE1_2026-11-02.md`: S-21, F1.1 y decisiones de implementación

2026-11-02 · Documento de planificación. Redactado por Claude (chat) para que Luciano lo apruebe o lo corrija y se lo pase a Claude Code. **Luciano aprobó todas las decisiones y las opiniones del §5 el 2026-11-02 a las 21:28.**

---

## 1. Lectura del estado

La ejecución va bien:
- se cerraron cinco tareas (F1.0, F1.4, F1.2, F1.3, F1.5 y F1.6) con la suite en verde (1599 passed, 9 xfailed, 0 failed);
- la regla de no invención quedó más estricta que en el plan;
- el modo objetivo entrega un **resultado negativo honesto**: sobrepaso 4,87 %, cumple; asentamiento 53,2 ms frente a 50, no cumple.

El tema importante es **S-21**.

## 2. S-21: el mecanismo está confirmado en el código y la magnitud cuadra

**El código.** `src/driveflow/sim/converters.py:391-393`, `ContOneQuadrantConverter._convert`:

```python
return self._current_action[0] if i_in[0] >= 0 else 1
```

**Qué hace:**
- si la corriente al **inicio** del período es negativa, aplica **tensión plena durante los 100 µs** del período;
- si la corriente es ≥ 0 y el PI pide frenar (duty 0), aplica 0 V durante todo el período, y la corriente puede **cruzar a negativo dentro del período**, porque la decisión no se reevalúa.

En un convertidor real de un cuadrante (chopper con diodo de libre circulación) la corriente **no puede ser negativa**: al llegar a 0, el diodo se bloquea, la corriente queda en 0 (conducción discontinua) y la máquina frena por fricción y carga.

**Magnitud estimada.** Es aritmética, no una medición. Con L_a = 19 µH, U = 65 V, τ = 100 µs y ω = 120 rad/s:

| ψ_e (V·s) | E = ψ_e·ω | ΔI en un período con u = 1 | ΔI en un período con u = 0 |
|---|---|---|---|
| 0,165 (stock GEM) | 19,8 V | **+238 A** | −104 A |
| 0,181 (PMG 132, desde T/I) | 21,7 V | +228 A | −114 A |
| 0,194 (PMG 132, desde U − RI) | 23,3 V | +220 A | −123 A |

Un período con corriente apenas negativa lanza la corriente a +220 A o más (par motor positivo). El siguiente pedido de freno la lleva a unos −100 A, y así sucesivamente. Es un **ciclo límite que inyecta energía**, coherente con lo que observó Claude Code: oscilación cada dos períodos, corriente hasta −160 A y velocidad que **sube** a 179–197 rad/s con referencia de 120.

Con una constante eléctrica τ_a = L_a/R_a ≈ 1,19 ms, un período de 0,1 ms es grueso para un evento de diodo. **Subpasos de Euler o RK45 no lo corrigen**, porque la decisión del convertidor sigue siendo por período (lección G7, bien aplicada por Claude Code).

**Es un comportamiento heredado de GEM.** El port es fiel y la física no lo es. Por eso la corrección debe ser un **modo opcional** con default idéntico, igual que en G3 y R2.1. Cambiar el default es una decisión aparte.

## 3. El punto que más preocupa: S-19 y los modelos promovidos

Claude Code dice dos cosas que se contradicen:
- **(a)** "el motor de stock con su fricción de stock no se ve afectado en las grillas actuales";
- **(b)** "S-19 probablemente es el mismo mecanismo". S-19 dice que el **15 % de los runs de la grilla de 5 clases** no sostiene la velocidad de referencia (hasta +72 %).

Si (b) es cierto, **los datasets con que se entrenaron los clasificadores promovidos** (dc_motor, PC, RPi5 y ESP32) **contienen corridas con este artefacto**. Eso sería un hallazgo de IA, no solo de simulación. Hay que resolver la contradicción **antes** de cualquier otra cosa de S-21. Por suerte es barato: los parquet de la grilla están en `data/`.

## 4. Decisiones propuestas

| # | Decisión | Estado |
|---|---|---|
| **S21-1** | **Primero, el diagnóstico de alcance (barato, sin cambiar código de planta).** Script `experiments/report_s21_dataset_scan.py`: en cada parquet de diagnóstico de `data/` (3 y 5 clases, Euler y RK45) y en los datasets que entrenaron a los clasificadores promovidos, contar por adquisición (a) cuántos períodos tienen `current_r` < 0, y (b) cuántas adquisiciones tienen velocidad media en el régimen > 2 % sobre la referencia. Cruzar con la lista de runs de S-19. Reportar por clase, velocidad y severidad, y qué fracción de las **ventanas de entrenamiento** cae en adquisiciones afectadas. Evidencia en `docs/audit/assistant/s21_dataset_scan.{md,json}`. | **Aprobada** (Luciano, 2026-11-02 21:28) |
| **S21-2** | **Referencia convergida y modo opcional (autorizado, sin cambiar defaults).** Agregar al convertidor un modo opcional, por ejemplo `Scenario.dc_converter_model: "gem" | "dcm"`, con **default `None` = `"gem"`**, byte a byte idéntico. En `"dcm"`, la decisión del convertidor se reevalúa **dentro del período** con subpasos y localización del cruce por cero: con u = 0 y la corriente llegando a 0, la corriente queda en 0 (conducción discontinua) hasta que el duty vuelva a ser > 0. Pre-registro: con subpasos ×10, ×100 y ×1000, el modo `"dcm"` converge (< 1e-4 NRMS en ω e i entre las dos discretizaciones más finas). Medir contra él el error del modo `"gem"` en los casos de S-21 y S-19, y en la velocidad nominal con carga (donde no debería haber diferencia). | **Aprobada** (Luciano, 2026-11-02 21:28) |
| **S21-3** | **Mitigación inmediata en el asistente (no cambia física).** Mientras S-21 esté abierto: `controller_design_check` y `describe_plant` devuelven una advertencia "one-quadrant converter: light-load deceleration not physical (S-21)" para motores con fricción 0b-D1 y para `dc_pmg132_v1`. `validate_draft` marca como `warning` (no bloquea) los experimentos de esos motores con referencia por debajo de la velocidad libre y carga < 1 N·m. El asistente lo dice al proponer. Estado de S-21: **abierto · mitigado en el asistente**. | **Aprobada** (Luciano, 2026-11-02 21:28) |
| **S21-4** | **Cambiar el default** del convertidor, regenerar datasets o re-entrenar: **no** en esta ronda. Se decide con la evidencia de S21-1 y S21-2. Si S21-1 muestra datos de entrenamiento afectados, se abre un hallazgo AI-N (el siguiente libre) y se trae la decisión a Luciano con opciones y costos. | **Aprobada** (Luciano, 2026-11-02 21:28) |
| **F1-A** | **Autorizar F1.1** (descargar 2–3 modelos de 5–9 GB cada uno; 16 GB de VRAM confirmados). La elección del modelo no depende de S-21. **Orden:** S21-1 primero (horas), después F1.1; S21-2 después de F1.1 o en paralelo si Claude Code lo prefiere. F1.7 y F1.8 se pueden intercalar. | **Aprobada** (Luciano, 2026-11-02 21:28) |

## 5. Decisiones de implementación de Claude Code (§4 y §5 del estado)

| # | Lo que hizo Claude Code | Opinión |
|---|---|---|
| 1 | Un default aceptado no se pasa a `Scenario` | **De acuerdo.** Mantiene la identidad byte a byte y evita activar la regla 0b-D1 por accidente |
| 2 | "1,146" o "1.500" ambiguos → preguntar | **De acuerdo como regla base.** Propuesta adicional, también determinista: si de las dos lecturas **solo una** cae dentro del rango válido del campo en `configs/plant_catalog.yaml`, se acepta esa y se informa ("lo interpreté como 1,146 V·s/rad; 1146 está fuera de rango"). La decide Python, nunca el modelo. Implementarla antes de F1.1, para que la comparación de modelos no se vea afectada por preguntas evitables |
| 3 | Parches todo-o-nada | De acuerdo |
| 4 | `preset` frente a `base` decidido por campo | De acuerdo |
| 5 | Herramientas `propose_experiments` / `propose_tuning` (no estaban en el plan) | **De acuerdo:** es la forma correcta de atar el "sí" a una propuesta concreta. Hay que agregarlas a la especificación (§2, C4/C5) y al plan (§5) |
| 6 | `is_affirmative` acepta "ok" escueto | De acuerdo, siempre que el mensaje sea **solo** la afirmación. "Ok, pero cambia la duración" no confirma (ya es así según el estado) |
| 7 | `suggest_value` nunca sugiere parámetros de máquina para otro motor | De acuerdo (P5). Consecuencia esperada: para un motor nuevo, el asistente pregunta L_a y J sin sugerir hasta que haya fuentes externas |
| 8 | Desde el chat solo se sintoniza el PI del motor DC | De acuerdo para la Fase 1 |
| 9 | `SIMULATED · AI-PLANNED` adelantado a F1.5 | De acuerdo |
| 10 | `dc_pmg132_v1` sin perfil en `NATIVE_PROFILES` ni métrica en `_condition_metrics` | **Corregirlo** en una tarea corta (`feat/ASSIST-1-pmg132-profile`): agregar el perfil y la métrica no cambia ningún default ni resultado. Registrarlo como hallazgo menor solo si algo visible dependía de eso (p. ej. *Experiment Runs* sin métrica para esos runs) |

## 6. Texto para pasarle a Claude Code

> Decisiones de Luciano del 2026-11-02 sobre `ESTADO_FASE1_2026-11-02.md`, ver `DECISIONES_S21_F1_2026-11-02.md`. Aprobadas S21-1 a S21-4, F1-A y las opiniones del §5.
>
> **Orden:**
> 1. S21-1 (barrido de datasets, sin tocar la planta). Si muestra datos de entrenamiento de modelos promovidos afectados, **detente y repórtame** antes de seguir.
> 2. La regla de desambiguación por rango (§5, punto 2) y la tarea corta del perfil de `dc_pmg132_v1` (§5, punto 10).
> 3. F1.1.
> 4. S21-2 y S21-3; F1.7 y F1.8 cuando convenga.
> 5. Frontend del avatar: según `ADENDA_FRONTEND_FASE1_2026-11-02.md` (FE-1 a FE-3 mientras espero la elección del modelo; FE-4 dentro de F1.10).
>
> **No cambies el default del convertidor** ni regeneres datasets (S21-4).
