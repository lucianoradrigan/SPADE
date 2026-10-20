# SPADE — Especificación integral de GUI / UX/UI

**Versión:** 3.0 · Documento consolidado  
**Fecha:** 20 de octubre de 2026  
**Idioma de toda la aplicación:** inglés (`en`)  
**Idioma de esta especificación:** español; los textos de la interfaz aparecen literalmente en inglés.  
**Estado:** propuesta técnica y de diseño para implementar; no representa funcionalidades ya desplegadas ni pruebas experimentales realizadas.  
**Base:** SPADE actual (`SPADE-main (2).zip`), alcance funcional solicitado, propuesta general de expansión y rediseño de PLECS Plant Studio.

> **Objetivo central:** que una persona investigadora pueda diseñar una planta, simularla, integrarla con Gymnasium, entrenar o comparar controladores y documentar sus resultados dentro de un único entorno consistente, reproducible y transparente.

## Guía de uso del documento

Este archivo es el **documento maestro** para diseño e implementación del frontend; reemplaza la necesidad de combinar manualmente los documentos de expansión general y PLECS por separado. Se puede compartir directamente con un equipo de desarrollo o con un asistente de programación. Contiene requisitos de producto, arquitectura de información, flujos, wireframes, componentes, estados, microcopy, interfaces técnicas, aceptación y plan de implementación.

**Cómo leerlo:**

- Capítulos **0–2**: alcance general, navegación y patrón transversal.
- Capítulo **3**: especificación **prioritaria y detallada** de PLECS como planta de Gymnasium; incluye dataset/surrogate, validación y comparador.
- Capítulos **4–8**: Tuning, RL, ejecución persistente, comparación y módulos restantes.
- Capítulos **9–15**: biblioteca de componentes, microcopy, accesibilidad, arquitectura, roadmap y pruebas.
- Capítulos **16–22**: sistema visual, inventario de pantallas, requisitos detallados de interacción, handoff para implementación y límites del alcance.

### Alcance y prioridad de implementación

| Área | Qué se diseñará | Prioridad |
|---|---|---|
| **PLECS Plant Studio** | Planteamiento por objetivos, conexión/capacidades, FMU y RPC, mapping I/O, definición Gym, verificación, perfiles, dataset, surrogate, comparación | **P0 / principal** |
| Runs y Jobs | Trabajos duraderos, cola, progreso, logs, cancelación, checkpoints cuando existan, comparación y procedencia | **P0 / habilitador** |
| Tuning Studio | PSO, optimización bayesiana y multiobjetivo con grilla fija y aprobación | P1 |
| RL Lab | Gymnasium, DQN/DDPG, edición de recompensas, entrenamiento y benchmarks | P1 |
| Otras plantas | Buck y DAB en vista compartida DC-DC | P1 |
| Inteligencia | OOD, PINN, pronóstico IGBT/RUL, registro de modelos | P2 |
| Environments y Help | Catálogo Gym, snippets verificados y selección guiada de algoritmos | P1–P2 |

### Decisiones de producto ya fijadas

1. Mantener **Streamlit + Plotly + Material icons + light/dark themes** y la división `view → adapter → backend numérico`.
2. Mantener SPADE como **workbench para investigadores e ingenieros**, no como HMI de operación industrial.
3. Conservar los módulos actuales, ampliando funciones sin romper navegación ni contratos existentes.
4. Priorizar **PLECS como planta de un Gymnasium real**. RPC batch, surrogate y FMU interactiva son capacidades distintas.
5. Nunca habilitar una capacidad en la GUI por mera apariencia: se exige verificación del backend.
6. Toda inferencia, comparación, gráfica, resultado negativo y procedencia deben quedar auditables.

---

## 0. Alcance, fuentes y decisiones

Esta especificación se construye a partir del alcance funcional aportado para SPADE y una inspección del repositorio `SPADE-main (2).zip`.

**Componentes existentes que se deben conservar**:

- `src/driveflow/viz/app.py`: Streamlit multipágina con `st.navigation` y Material icons.
- `src/driveflow/viz/views/`: vistas delgadas que invocan adapters.
- `src/driveflow/viz/adapters/`: frontera con librería Python.
- `src/driveflow/viz/components/page_header.py`: encabezados, badges, procedencia.
- `src/driveflow/viz/components/promotion_gate.py`: compuerta de validación de modelos.
- `src/driveflow/viz/components/model_comparison.py`: comparación actual de modelos.
- `src/driveflow/viz/adapters/session.py`: logs de runs en `st.session_state`, máximo 30; insuficiente para experimentos largos persistentes.
- `src/driveflow/viz/theme/tokens.py`: tokens de color light/dark, y `theme/plotly.py`: estilos Plotly.
- `src/driveflow/viz/views/transfer_learning.py`: wizard actual de 4 pasos que sirve como patrón.

**Decisión:** conservar Streamlit. Agregar servicio de jobs y almacenamiento duradero fuera de la sesión. Conservar `view → adapter → dominio`; nunca ejecutar física, RL o optimización dentro de un componente de UI. Plotly sigue siendo la base gráfica; el resto de componentes se compone con widgets Streamlit, CSS/tokens existentes y HTML seguro.

**Nota PLECS:** La RPC de PLECS Standalone es local al mismo computador donde corre PLECS; el servidor RPC está deshabilitado por defecto, el puerto de ejemplo es 1080 y la operación `simulate` es bloqueante. Los barridos RPC son viables. El control a cada paso **no** se implementa simplemente llamando RPC durante una ejecución bloqueante; se debe estudiar un acoplamiento diferente, como FMI Co-Simulation en una versión de PLECS que lo soporte. No se presentará Step-by-step como habilitado hasta comprobarlo mediante pruebas. Referencias: https://docs.plexim.com/plecs/latest/scripting/ y https://docs.plexim.com/plecs/latest/using-plecs/configuring-plecs/ .

### Principios no negociables

1. Ninguna gráfica numérica nace de datos simulados previamente cargados sin ejecutar. El botón `Generate` (o `Start ...` para entrenamiento/optimización) dispara el cálculo. Las vistas vacías contienen diagramas estructurales/explicaciones, no curvas inventadas.
2. Cada señal tiene procedencia visible: `SIMULATED`, `SIMULATED · PLECS`, `SIMULATED · SURROGATE`, `EXTERNAL FILE`, `MEASURED · EXTERNAL DATASET`. Los hashes y licencias se muestran cuando aplican. `CONNECTED / LIVE` queda reservado y nunca se emplea para el enlace RPC.
3. Toda inferencia se marca `AI SUGGESTION — REQUIRES VERIFICATION`; se muestra modelo, versión, confianza calibrada cuando existe y límites aplicables.
4. Las modificaciones de controladores, umbrales por defecto y modelos promocionados exigen acción humana inequívoca y registro de aprobación. Se reutiliza `PromotionGate` para modelos y se propone `ApprovalGate` para parámetros/controladores.
5. Resultados negativos, experimentos fallidos, divergencias y condiciones no confiables se ven con el mismo nivel de detalle que un éxito.
6. `Uncertain` (baja confianza para clases conocidas) no es sinónimo de `Unknown condition` (fuera de distribución).

## 1. Arquitectura de navegación

```text
SPADE · Engineering Workbench
├── Workspace
│   ├── Overview                        [existente, enriquecer]
│   ├── Experiment Runs                 [nuevo: cola + historial + comparaciones]
│   └── Environments                    [nuevo: catálogo Gymnasium]
├── Simulation
│   ├── PLECS Plant Studio              [nuevo: plantas Gym, datasets, surrogate, comparación]
│   ├── DC Motor                        [existente]
│   ├── PMSM / FOC                      [existente]
│   ├── VSC / DPC                       [existente, tab Parameter Identification]
│   └── DC-DC Converters               [nuevo, tabs Buck / DAB]
├── Design & Control
│   ├── Tuning Studio                   [nuevo]
│   └── RL Lab                          [nuevo]
├── Intelligence
│   ├── AI Diagnostics                 [existente, agregar OOD]
│   ├── Prognostics                    [nuevo, IGBT RUL]
│   ├── Transfer Learning              [existente]
│   └── Model Registry                 [existente]
├── Operations
│   ├── Monitoring                     [existente]
│   └── Alert History                  [existente]
└── Help
    ├── Documentation                  [existente]
    └── Algorithm Guide                [nuevo]

GLOBAL SIDEBAR FOOTER
  PLECS: [Not configured | Available | Error]   (solo RPC/simulador)
  Jobs: [N running]   [Open queue]
  Data: simulation/external datasets, never telemetry live
```

**Por qué:** Tuning y RL crean controladores, por lo que tienen su propia categoría Design & Control. Diagnóstico y pronóstico predicen estados y permanecen en Intelligence. Environments es transversal. `DC-DC Converters` evita duplicar diseño para Buck y DAB. Los detalles avanzados de cada planta se resuelven con plugins/fichas especializadas, no con 10 rutas independientes.

### Accesos cruzados

- Desde cualquier planta: `Generate` → `Compare runs` → `Send to Tuning Studio` o `Open in RL Lab` si es compatible.
- Desde PLECS: `Save plant profile` → `Open in Simulation / Tuning Studio / RL Lab / Dataset generation` según capacidades disponibles.
- Desde Tuning: `Save candidate` → `Evaluate on condition grid` → `Request approval` → `Use as default`.
- Desde RL: `Train` → `Compare with classical controllers` → `Save candidate` → `Request approval`.
- Desde AI Diagnostics: `Unknown condition` → detalles OOD → evidencia secundaria de agente de simulación → exportar run.
- Desde Model Registry: versiones enlazadas con sus runs y datasets de origen.

## 2. Patrón común de todas las pantallas experimentales

**Cabecera persistente:** breadcrumb (`Simulation / ...`), título, resumen, source badge, selector global de `Plant profile`, `Experiment draft`, e indicadores de entorno/datos. **Zona principal:** un panel izquierdo de configuración (colapsable) y un panel derecho ancho de resultados. **Zona inferior:** pestañas `Signals`, `Metrics`, `Compare`, `Provenance`, `Logs` según tipo de página.

**CTA explícita** según tarea: `Generate`, `Start sweep`, `Fit surrogate`, `Start tuning`, `Start training`, `Run evaluation`. Ningún cambio de dropdown dispara automáticamente la simulación o el entrenamiento.

**Ciclo de resultados**: `No runs yet` → `Queued` → `Running` → `Completed | Partially completed | Failed | Cancelled`. Todo resultado conserva los parámetros usados al iniciar, aunque el usuario edite el formulario después.

**Entidades para adapters**:

```text
PlantProfile {id, kind, engine, file_hash?, signals[], parameters[],
              action_space, observation_space, capabilities[], validation}
ExperimentSpec {id, plant_profile_id, mode, controller_spec?, dataset_refs[],
                conditions_grid, objectives[], constraints[], seed, algorithm, budget}
Job {id, experiment_id, state, progress?, eta_seconds?, checkpoint?, worker_id?,
     queued_at, started_at?, finished_at?, log_ref?, cancel_requested}
Run {id, experiment_id, job_id?, completed_at, config_hash, artifacts[],
     source_provenance[], metrics[], failure_reason?, command, environment}
SignalSeries {name, unit, samples_ref, t_ref, origin, run_id, hash?, model_ver?,
              confidence_metadata?, source_uri?, license?}
ApprovalRecord {subject, baseline_run, candidate_run, gate_report, approver,
                user_confirmed_at, decision, config_before_hash, config_after_hash}
```

Todos los datos de estas entidades se validan en backend. La UI solo muestra o construye propuestas tipadas.


## 3. PLECS Plant Studio — circuito como planta Gymnasium (prioridad principal)

### 3.1 Propósito y corrección del diseño anterior

Este capítulo detalla el flujo principal de PLECS y reutiliza los patrones de `Jobs`, `Runs`, `SourceBadge`, `Comparison`, `ApprovalGate` y `Plotly` descritos en el documento. Es la especificación de experiencia de usuario para la integración PLECS–Gymnasium, no una afirmación de que el backend ya esté implementado.

**Solicitud central del investigador:** convertir circuitos PLECS en **plantas de un entorno Gymnasium de SPADE** para evaluar y entrenar controladores, sin confundir esto con lanzar una simulación completa o generar un dataset.

**Error de UX que se evita:** un wizard `Connect → Import → Map I/O → Choose coupling` no cubre por sí solo los requisitos del entorno RL (action/observation spaces, reloj, `reset`, `step`, recompensa, límites, fin de episodio, reproducibilidad). Tampoco debe presentar una conexión RPC como prueba de que existe co-simulación interactiva.

#### Decisiones de producto

- Renombrar la pantalla `PLECS Plant Studio` a **`PLECS Plant Studio`**, dentro de **Simulation**.
- Priorizar el caso **`Use PLECS as a Gymnasium plant`**; ofrecer `Generate simulation datasets` y `Build a fast surrogate plant` como rutas relacionadas, pero diferenciadas.
- Incorporar un **asistente guiado de seis etapas**, con configuración progresiva y resumen de preparación siempre visible.
- Mantener un camino **Beginner / Advanced** dentro de cada etapa, no duplicar pantallas ni esconder la información física crítica.
- Permitir **Save draft** en cualquier momento; **Publish as Gym environment** solamente tras pasar verificaciones exigibles de la ejecución elegida.
- No crear gráficas de ejemplo como si fueran resultados reales; todos los gráficos de desempeño requieren presionar `Run validation`, `Generate` o `Start training` con backend disponible.
- **No llamar LIVE** al vínculo con PLECS: `PLECS engine available` solo se refiere a un simulador externo.
- Mostrar toda configuración de muestra como `EXAMPLE CONFIGURATION — NOT EXECUTED`.

### 3.2 Modelo mental del usuario: primero el objetivo

Pantalla inicial con pregunta **`What would you like to do with your PLECS circuit?`** y tres tarjetas:

| Opción visible | Qué promete | Backend técnico | Salida reutilizable |
|---|---|---|---|
| **Use PLECS as a Gymnasium plant** — `Recommended for closed-loop RL` | SPADE suministra acciones en cada intervalo de control y recibe observaciones, reward y terminación | FMU Co-Simulation / adaptador de paso compatible, **tras verificarlo realmente** | `PLECS-Gym` con `reset()` / `step()` |
| **Generate simulation datasets** | Barridos, grillas y parámetros variados; control interno del circuito | PLECS Standalone RPC (`simulate`) | Dataset de simulación; **no Gym interactivo** |
| **Build a fast surrogate plant** | Ajustar sustituto a partir de runs PLECS y utilizarlo como entorno rápido | RPC para generar datos + entrenamiento del surrogate | `SURROGATE-Gym`; error y dominio de validez visibles |

Regla crítica: **no hacer downgrade silencioso** del primer modo a RPC o surrogate. Si el adaptador interactivo no está disponible, explicar el bloqueo y ofrecer como opciones separadas los otros dos modos; nunca declarar `Gym ready`.

### 3.3 Sidebar y navegación

```text
Simulation
  DC Motor
  PMSM / FOC
  VSC / DPC
  DC-DC Converters
  PLECS Plant Studio  ← nuevo punto principal

Workspace
  Environments        ← aquí aparecen perfiles publicados
  Experiment Runs

Design & Control
  Tuning Studio       ← puede seleccionar plantas PLECS compatibles
  RL Lab              ← entrenar/evaluar sobre Gym environment publicado
```

La página principal tendrá tabs `My Plants | Create Plant | Compare with SPADE | Connection & Diagnostics`; las mismas entidades pueden abrirse por enlaces contextualizados en `RL Lab`, `Tuning Studio` y `Environments`.

### 3.4 Flujo guiado de seis etapas

```text
[1. Goal] → [2. Engine] → [3. Circuit] → [4. Signals] → [5. Gym behavior] → [6. Validate & publish]
    │          │             │               │                 │                         │
  Intention   Capability    File+provenance   I/O contract      Time, reset, reward       Real test report
```

Una columna derecha fija con **`Environment readiness`** muestra: `Engine`, `Circuit`, `Action mapping`, `Observation mapping`, `Gym dynamics`, `Validation`, `Publishability`. Cada entrada indica `Not configured`, `Needs attention`, `Validated`, `Failed`, o `Not supported` con icono y texto (no solo color). La UI mantiene el progreso al navegar y deja acceder a cada etapa para corregir fallos.

#### Etapa 1 — Goal: decidir sin tecnicismos

**Pregunta:** `What is your goal?` con tres tarjetas; el modo preferido queda visible pero no promete conectividad. Debajo: `What will SPADE control?` con presets opcionales: `DC-DC voltage regulation`, `Motor speed regulation`, `Current control`, `Custom research task`. Son **plantillas para campos de configuración, no circuitos preimportados**.

**UX:** recomendación contextual según tarea y versión disponible. Botón `Continue to engine setup`. Tarjeta `Why does Gym need step-by-step control?` explica en tres líneas `action → plant advances → observation → reward`.

#### Etapa 2 — Engine: conexión y capacidades, no simple indicador de puerto

- **Modo Gym interactivo:** preguntar por un **FMU Co-Simulation exportado desde un atomic subsystem** de PLECS, versión FMI, archivos compatibles con SO, servicio endpoint y runtime/licencia. Ofrecer guía paso a paso `Prepare a PLECS Co-Simulation FMU` con enlaces oficiales.
- **Modo dataset:** probar **PLECS Standalone RPC** en el host que ejecuta SPADE, dirección sugerida `http://localhost:1080`, disponibilidad y permisos. El navegador y el backend no son el mismo host por definición; un PLECS instalado en el PC del usuario no sirve automáticamente a un SPADE remoto.
- **Modo surrogate:** requiere una fuente de datos de barrido; puede usar PLECS RPC para generarla.
- **Diagnóstico accionable:** `Cannot reach RPC endpoint`, `Co-Simulation endpoint disabled`, `PLECS runtime missing`, `Unsupported FMU version`, `Missing compatible platform binary`, `License unavailable`, `Backend running on a different host`.
- **Output de esta etapa:** un `EngineCapabilityReport` con las capacidades que fueron **detectadas**, cuáles no se pudieron comprobar y sus acciones correctivas. No inferir versión/licencia del circuito por nombre del archivo.

**Regla de habilitación:** `Interactive control: Available` solo si el **adaptador FMU** ha pasado un chequeo real de inicialización y avance, no por detectar el servidor RPC.

**Referencia verificable:** PLECS 5.0 incorpora exportación de FMU Co-Simulation (FMI 2.0/3.0); la FMU de co-simulación es **tool-coupling**, exige PLECS instalado y licencia en runtime, y el endpoint de co-simulación debe habilitarse. Para señales discretas con discontinuidades de compuerta, la documentación recomienda FMI 3.0/Event Mode cuando sea compatible. Ver https://docs.plexim.com/plecs/latest/functional-mockup-interface/ . RPC de PLECS Standalone está deshabilitado por defecto y solo admite clientes del mismo host, ver https://docs.plexim.com/plecs/latest/scripting/ .

#### Etapa 3 — Circuit: importar un modelo ejecutable

- `Upload .fmu` **para modo interactivo**, con validación de `modelDescription.xml`, tipo de FMU, causality de variables, versión FMI, binarios de plataforma y referencia de exportación PLECS. **No suponer que subir un .plecs genera una FMU.** Mostrar instrucciones para exportar el atomic subsystem en PLECS.
- `Upload .plecs` o `Select local circuit` **para batch sweep RPC**, registrando ruta controlada, hash SHA-256, modelo PLECS y variables accesibles por el adaptador si realmente pueden enumerarse. **No prometer introspección universal de bloques**: si el parser no lo soporta, permitir manifiesto manual validado.
- `Import an existing plant profile` cuando el usuario tiene una configuración ya validada.
- Mostrar `Circuit identity`, `File hash`, `Model type`, `Available inputs / outputs`, `Known parameters`, `Validation summary`, `What SPADE still needs`.
- Previsualizar esquema **solo si existe imagen real** o vista provista por un adaptador; no inventar bloques con fines estéticos.
- Crear perfil versionado con identificador estable. Cambio de archivo o hash invalida validaciones dependientes: estado `Stale — Revalidate`.

#### Etapa 4 — Signals: una traducción visual del circuito a Gym

Dividir en cuatro grupos semánticos (sin abrumar con nombres de API al principio):

1. **`What can SPADE control?` — Actions**: duty cycle, torque reference, gate commands, voltage reference, etc. Selector de canal FMU, unidad, tipo discreto/continuo, rango permitido, saturación y valor seguro ante error.
2. **`What can SPADE observe?` — Observations**: tensión, corriente, velocidad, error, estado térmico, etc. Selector múltiple con orden explícito, unidad, límites de plausibilidad, escala/normalización, observaciones disponibles al instante correcto.
3. **`What changes during an episode?` — References & disturbances**: consignas, carga, variaciones de tensión, condiciones iniciales. Definir qué cambia al iniciar un episodio y qué cambia dentro de él.
4. **`What must SPADE protect?` — Safety signals**: sobrecorriente, sobretensión, salida del dominio del solver. Umbrales, reglas de terminación y manejo de acciones inválidas.

Tabla editable:

| PLECS channel | Role | SPADE signal | Unit | Bounds | Timing | Status |
|---|---|---|---|---|---|---|
| `(choose actual FMU input)` | Action | `duty_cycle` | dimensionless | `[0, 1]` *illustrative* | Hold Ts | `Unmapped` |
| `(choose actual FMU output)` | Observation | `v_out` | V | user-defined | End of Ts | `Unmapped` |
| `(choose actual FMU output)` | Observation | `i_L` | A | user-defined | End of Ts | `Unmapped` |

Campos de ejemplo señalados claramente como **`EXAMPLE MAPPING — NOT FROM THE IMPORTED FILE`** hasta existir datos reales. `Auto-suggest mapping` puede proponer nombres compatibles, **pero requiere confirmación**. Incluir `Validate signals` y alertas: dirección equivocada, unidades incompatibles, señales ausentes, acción no aplicada, observaciones no finitas, orden inconsistente.

La cabecera del mapeo debe presentar la traducción humana `SPADE writes → PLECS simulates → SPADE reads`, con un ejemplo visible de la dimensión de `Box(...)` o `Discrete(...)` resultante.

#### Etapa 5 — Gym behavior: configuración indispensable que faltaba

**Pantalla con cinco pestañas o acordeones**, con `Basic` por defecto y `Advanced` desplegable:

##### A. Control timing (`When does SPADE act?`)

- `Controller sample time (Ts)`, unidades y comprobación con la configuración del solver/FMU.
- `Number of control steps per episode` o duración física equivalente.
- `Action hold` (p.ej. Zero-order hold), retardo de acción, interpolación solo cuando sea implementable y explícita.
- Distinguir el **paso de control** del paso interno (fijo/adaptativo) del solver PLECS; no deben confundirse ni asumirse idénticos.
- Convención temporal: agente ve `obs(t)`, decide `action(t)`, PLECS avanza a `t+Ts`, devuelve `obs(t+Ts)`.
- Para compuertas discretas: atención a eventos, compatibilidad FMI 3.0, discontinuidades y restricciones de muestreo. Si la FMU no satisface el método seleccionado, bloquear publicación.

##### B. Episode reset (`How does a new experiment begin?`)

- `reset(seed)` reinicializa instancia y estados definidos; opcionalmente aleatoriza condiciones con semilla registrada.
- Elegir `Fixed initial conditions`, `Randomized within bounds`, `Scenario presets`.
- Distinguir `initial states`, `parameters at initialization`, `disturbances during episode` y sus mecanismos efectivos de cambio.
- Si el adaptador no puede reiniciar de forma fiable, el perfil **NO** se publica como Gym interactivo.

##### C. Reward function (`What should the controller learn?`)

Editor sin fórmulas obligatorias para principiantes:

| Term | Semantic role | Weight | Form |
|---|---|---:|---|
| Tracking error | Minimize difference from reference | configurable | `-w_e * (reference-output)^2` |
| Unsafe current | Avoid overcurrent | configurable | `-w_i * max(0, abs(i)-limit)^2` |
| Action variation | Reduce abrupt switching/control | configurable | `-w_u * (u_t-u_(t-1))^2` |
| Efficiency (optional) | Reduce losses | configurable | `-w_l * estimated_loss` (only if signal/model available) |

Permitir activar/desactivar términos, validar unidades/escalas y mostrar el desglose **solo después de evaluarlo con valores reales**. Registrar límites de penalización y evitar doble conteo inadvertido. En avanzado, expresión personalizada validada/sandboxed sin ejecución arbitraria.

##### D. Safety & episode end (`When should an episode stop?`)

- `terminated = True` para una condición terminal propia de la tarea, como violación de seguridad que define el experimento.
- `truncated = True` para límite de horizonte, timeout externo o pérdida del simulador, diferenciando la causa mediante `info`.
- `solver failure` se registra explícitamente y nunca como éxito; decidir política (truncamiento controlado o fallo de run) según gravedad, conservando trazas.
- Acción fuera de límites: decidir `Reject action` o `Clip and log`; hacer visible qué se aplicó realmente.
- En fallos de comunicación: abortar/aislar instancia y registrar estado; no reutilizar silenciosamente valores anteriores.

##### E. Environment export (`What will RL Lab receive?`)

Presentar un contrato visible, construido desde la configuración: `observation_space`, `action_space`, `reset()`, `step()`, `termination rules`, `metadata`, ejemplo de uso Python, capacidad `train/evaluate`, y modo de simulación/procedencia.

Contrato conceptual:

```python
# Example API contract, not a runnable instance without a backend adapter
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

Para verificación de estructura usar `gymnasium.utils.env_checker.check_env`; además se requieren tests de integración FMU, reproducibilidad y dinámica bajo acciones. Documentación: https://gymnasium.farama.org/api/env/ .

#### Etapa 6 — Validate & publish: prueba obligatoria antes de usar RL

Panel **`Environment verification`** con checks reales y drill-down:

1. `Circuit file and hash match`.
2. `Backend capability supports interactive stepping`.
3. `FMU instance initializes and closes correctly`.
4. `Reset can be repeated and state is well defined`.
5. `Observation and action spaces validate`.
6. `Action changes affect the designated PLECS inputs` (si una acción no produce efecto ante prueba controlada, advertir y pedir verificación; no inferir de una sola muestra).
7. `A multi-step episode advances physical simulation time monotonically`.
8. `Sampling and state synchronization meet configured assumptions`.
9. `Reward terms use available signals and finite values`.
10. `Termination, truncation, invalid action and solver-error handling are tested`.
11. `Seed and scenario reproduction check`.
12. `Performance benchmark` documenta tiempo de pared por step/episodio y factores que limitan entrenamiento (sin ETA inventado).

**Botón real:** `Run verification` inicia job persistente; en el prototipo, este botón permanecerá claramente no conectado. Tras la prueba mostrar `Validated`, `Failed`, `Partially validated`, `Not supported` con razones, comandos y logs. El usuario puede **guardar un Draft aunque falle**; `Publish to Environments` se habilita solo si la compatibilidad real fue verificada.

**Acciones después de publicar:** `Open in RL Lab`, `Run baseline episode`, `Compare against native SPADE plant`, `Save version`, `Export Python snippet`. El perfil se muestra en `Workspace → Environments`, incluyendo el motor PLECS vs surrogate, el hash, validaciones, última ejecución y una indicación de que no es hardware físico.

### 3.5 Layout: progresión clara y ayudas en contexto

#### Wireframe A — Pantalla de bienvenida

```text
┌ SPADE / Simulation / PLECS Plant Studio ───────────────────────────────────────┐
│ Create an environment from a circuit.             PLECS: Not configured       │
│                                                                                │
│ What would you like to do with your PLECS circuit?                            │
│ ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────────┐          │
│ │ Use as Gym plant │ │ Generate dataset │ │ Build surrogate         │          │
│ │ (recommended)    │ │ Run many cases   │ │ Train fast / verify     │          │
│ │ Closed-loop      │ │ Open/batch loop  │ │ Approximate model       │          │
│ └──────────────────┘ └──────────────────┘ └────────────────────────┘          │
│                                                                                │
│ How it works: SPADE action → PLECS advances → Observation + Reward            │
│ Optional: [Motor speed] [Buck voltage] [Custom task]                          │
│ [Save draft]                                          [Continue →]             │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### Wireframe B — Wizard con estado permanente

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ PLECS Plant Studio     Example configuration  [DRAFT] [SIMULATED · PLECS]    │
│ 1 Goal — 2 Engine — 3 Circuit — 4 Signals — 5 Gym behavior — 6 Validate       │
├────────────────────────────────────────────────────┬─────────────────────────┤
│ MAIN CONFIGURATION                                 │ ENVIRONMENT READINESS   │
│                                                    │ Engine       Needs test │
│  (formularios orientados a tarea)                  │ Circuit      No file    │
│                                                    │ Actions      Missing    │
│  Task explanation + guidance in plain English     │ Observations Missing    │
│  Advanced settings ▸                              │ Dynamics     Missing    │
│                                                    │ Verification Not run    │
│                                                    │                         │
│ [Back] [Save draft]                     [Next]     │ Why not ready? ▾       │
└────────────────────────────────────────────────────┴─────────────────────────┘
```

#### Wireframe C — Mapping intuitivo

```text
┌ What SPADE controls ──────────────────┐ ┌ What SPADE observes ──────────────┐
│ Action source       [FMU input ▼]     │ │ v_out              [FMU output ▼]│
│ Control name        [duty_cycle ]    │ │ i_L                [FMU output ▼]│
│ Allowed range       [0] ... [1]      │ │ [+ Add observation]              │
│ Unit                [1 ▼]            │ │ Order: [v_out] [i_L]             │
└───────────────────────────────────────┘ └───────────────────────────────────┘
 [Reference] [Disturbance] [Safety]  [Show raw channel metadata]
 Contract preview: Action Box(shape=...)  Observation Box(shape=...)
 Blocking issues: 2 unmapped signals. [Validate mapping]
```

#### Wireframe D — Gym behavior, episodio y recompensa

```text
┌ Gym behavior ────────────────────────────────────────────────────────────────┐
│ [Timing] [Reset] [Reward] [Safety] [Contract]                              │
│ Timing:  Ts [       ] s     Episode length [     ] steps                    │
│          Action hold [Zero-order hold ▼]    Solver notes ▸                   │
│ Reward:  [x] Tracking error            Weight [     ]                       │
│          [x] Overcurrent penalty       Weight [     ]                       │
│          [x] Action smoothness          Weight [     ]                       │
│          [ ] Loss / efficiency term     Requires valid loss signal          │
│ Stop:    [x] Safety violation   [x] Horizon reached                        │
│ Contract preview: reset(seed) → obs; step(action) → obs,reward,terminated,...│
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Wireframe E — Validación y activación

```text
┌ Environment verification ────────────────────────┬ Next actions ─────────────┐
│ [Run verification]  Status: NOT TESTED            │ [Save Draft]             │
│ — FMU init/reset                                 │ [Publish to Environments]│
│ — One controllable step                          │ (disabled until checks)  │
│ — 10-step continuity                              │ [Open in RL Lab]         │
│ — Reward and safety                               │ (disabled until publish)│
│ — Determinism and limits                          │                          │
│ Reports: logs, versions, timing, errors           │                          │
└───────────────────────────────────────────────────┴──────────────────────────┘
```

### 3.6 Ejemplo didáctico completo: Buck voltage control (solo plantilla)

**Caso ilustrativo, no resultados numéricos obtenidos:** un investigador dispone de un convertidor Buck implementado en PLECS. Quiere entrenar un controlador RL para regular `Vout` ante perturbaciones de carga.

1. Escoge `Use PLECS as a Gymnasium plant` y template `DC-DC voltage regulation`.
2. El asistente verifica que el adaptador de FMU Co-Simulation puede ejecutar el modelo; si no puede, se detiene la publicación y muestra cómo preparar la exportación en PLECS.
3. Importa un `.fmu` que **realmente exponga** una entrada de control (por ejemplo duty ratio) y salidas `Vout` e `iL`. No suponer que los nombres aparecen así en cualquier circuito.
4. Mapea `Action = duty_cycle`, `Observations = [Vout, iL]`, `Reference = Vref`, `Disturbance = Rload` cuando los canales correspondientes existan.
5. Define límites de `duty`, `Ts`, horizonte, inicialización, términos de tracking/penalización por corriente y reglas para detener una simulación peligrosa/inestable.
6. Ejecuta una validación donde SPADE aplica acciones diferentes al circuito en los intervalos designados y comprueba que el estado y el tiempo evolucionan correctamente.
7. Publica **solo después de verificaciones reales**. Entonces `RL Lab` utiliza el entorno. El entrenamiento RL se ejecuta como job y produce runs reproducibles.
8. Compara PI vs RL sobre **idénticos escenarios** y publica también resultados negativos; el perfil sigue etiquetado como `SIMULATED · PLECS`.

Un ejemplo de parámetro como `duty ∈ [0,1]` es solo una plantilla de configuración, no un valor descubierto del circuito. Los valores de corriente, tensión, pérdidas y tiempos de ejecución no se rellenan hasta ejecutar el modelo.

### 3.7 Modelos de backend y servicios sugeridos

Conservar `View → Adapter → Domain/Runner`:

```text
src/driveflow/viz/views/plecs_plant_studio.py
src/driveflow/viz/adapters/plecs_adapter.py
src/driveflow/viz/components/plant_goal_cards.py
src/driveflow/viz/components/plecs_readiness.py
src/driveflow/viz/components/gym_signal_mapper.py
src/driveflow/viz/components/gym_behavior_editor.py
src/driveflow/viz/components/gym_validation_report.py
src/driveflow/integrations/plecs/rpc_client.py
src/driveflow/integrations/plecs/fmu_adapter.py
src/driveflow/integrations/plecs/diagnostics.py
src/driveflow/environments/plecs_gym_env.py
src/driveflow/environments/plant_registry.py
src/driveflow/experiments/job_runner.py
src/driveflow/experiments/run_store.py
```

**Todas son rutas propuestas, no afirmaciones de archivos existentes.** Conservar la separación de los adapters actuales de `src/driveflow/viz/adapters/` y componentes de `src/driveflow/viz/components/`.

#### Entidades de dominio

```yaml
PlecsPlantProfile:
  profile_id: generated-id
  source_kind: plecs_fmu_co_sim  # or plecs_rpc_batch / surrogate
  circuit_hash: sha256
  fmu_hash: sha256-or-null
  plecs_version: detected-or-unknown
  fmi_version: detected-or-null
  endpoint_capabilities: []
  action_mapping: []
  observation_mapping: []
  reference_mapping: []
  disturbance_mapping: []
  action_space: null
  observation_space: null
  controller_sample_time_seconds: null
  reset_strategy: null
  episode_horizon_steps: null
  reward_terms: []
  safety_rules: []
  validation_run_id: null
  version: 1
  status: draft
  provenance: {origin: 'SIMULATED · PLECS'}

EnvironmentCapability:
  can_simulate_batch: false
  can_step_interactively: false
  can_reset_episode: false
  can_train_rl: false
  can_serialize_config: true
  validated_at: null
  diagnostic_messages: []
```

Estas capacidades provienen de pruebas concretas; no del tipo de archivo únicamente.

#### Contrato Gymnasium propuesto

```python
class PlecsGymEnv(gym.Env):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space

    def reset(self, *, seed=None, options=None):
        # establish/reinitialize a verified plant instance
        # configure initial state/scenario using recorded seed
        # return (observation, info)
        ...

    def step(self, action):
        # validate and apply action
        # advance *same physical simulation* from t to t+Ts
        # read causal observations; compute reward, terminal/truncation
        # return (obs, reward, terminated, truncated, info)
        ...

    def close(self):
        # close FMU instance/resources safely
        ...
```

**Important:** un adaptador RPC que reinicia una simulación desde t=0 en cada llamada `step` NO cumple el contrato de planta interactiva, aunque la firma Python se parezca a Gym. Si se usa un surrogate, el tipo del entorno será `SIMULATED · SURROGATE` y se muestra fidelidad frente al circuito original.

### 3.8 Estado persistente, trabajos y errores

**State machine:** `DRAFT → IMPORTED → MAPPED → SPECIFIED → VERIFYING → VALIDATED → PUBLISHED`; pueden ocurrir `FAILED`, `UNSUPPORTED`, `STALE`, `DISCONNECTED`. Se puede reabrir un borrador aunque no haya PLECS disponible; publicar es una acción distinta y requiere verificación de versión y artefactos.

- `Test engine`, `Run verification`, `Generate sweep`, `Fit surrogate`, `Train agent` y `Run evaluation` generan trabajos persistentes cuando la operación pueda prolongarse.
- Mostrar `Queued`, `Running`, `Cancelling`, `Partial`, `Completed`, `Failed`, `Cancelled`. No prometer `Resume` si el job/backend no tiene checkpoint compatible.
- PLECS puede serializar el acceso a una instancia/licencia. Implementar locking y colas por motor/perfil para evitar carreras entre usuarios y episodios.
- Mostrar duración estimada únicamente con benchmarks de runs previos comparables; de otro modo `ETA unavailable`.
- Conectividad no es telemetría: `PLECS available` (herramienta), **nunca** `Connected / Live`.
- Cada run guarda configuración exacta, RNG seed, hashes, versiones, controlador, reward spec, historial de fallos, métricas y comando reproducible.
- Nunca cambiar de modelo PLECS a surrogate sin aceptación expresa.
- Al actualizar un `.plecs`/`.fmu` o el mapping: invalidar verificación y mostrar `Revalidation required`.

### 3.9 Microcopy inglés listo para interfaz

| State or action | Exact UI copy |
|---|---|
| Start | `Create an environment from a PLECS circuit` |
| Goal | `What would you like to do with your PLECS circuit?` |
| Primary | `Use PLECS as a Gymnasium plant` |
| Primary explanation | `Train or evaluate a controller that interacts with PLECS at each control interval.` |
| Dataset | `Generate simulation datasets` |
| Surrogate | `Build a fast surrogate plant` |
| Setup | `Prepare a Co-Simulation FMU in PLECS` |
| Not available | `Interactive stepping is not available with the detected backend.` |
| Local RPC | `PLECS RPC is available only when SPADE's backend runs on the same computer.` |
| No file | `No circuit has been imported yet.` |
| No outputs | `This model does not expose the signals required by your selected task.` |
| Mapping | `What can SPADE control?` / `What can SPADE observe?` |
| Action | `Choose the input that the controller can change.` |
| Reset | `How should each episode begin?` |
| Reward | `What should the controller learn?` |
| Timing | `How often should SPADE apply a control action?` |
| Safety | `When should an episode stop?` |
| Status | `Not tested`, `Needs attention`, `Validated`, `Failed`, `Not supported` |
| Validate | `Run environment verification` |
| Draft | `Save draft` |
| Publish | `Publish to Environments` |
| Disabled publish | `Finish verification before publishing this Gymnasium environment.` |
| Not Gym | `A completed PLECS sweep does not provide an interactive Gymnasium plant.` |
| Stale | `The circuit or mapping changed. Re-run verification.` |
| Provenance | `SIMULATED · PLECS` / `SIMULATED · SURROGATE` |
| Negative | `Validation failed. The detailed report is saved with this run.` |
| No artificial graphs | `No simulation has been executed for this configuration.` |
| Example | `EXAMPLE CONFIGURATION — NOT EXECUTED` |

### 3.10 Accesibilidad y responsive

- Desktop: flujo horizontal de 6 etapas, contenido 65–75%, panel readiness 25–35%. Sidebar reutiliza los tokens de SPADE.
- 13" laptop: readiness compacta `3/7 requirements met` con expansión por clic; mapping se hace lista vertical editable en lugar de tabla estrecha.
- Tablet/móvil: stepper vertical, tarjetas de objetivo apiladas, `Back` y `Continue` abajo; no depender de drag-and-drop (orden de observaciones por botones ↑ ↓).
- `Not supported`, `Failed` y `Validated` se distinguen por iconos+texto, no solo por rojo/verde. Contraste AA, foco visible y labels reales.
- Feedback técnico junto al campo correspondiente, con instrucciones de corrección. Errores de solver y trazas expandidas bajo `Advanced diagnostics`, sin esconder estado principal.
- Teclado, lector de pantalla, persistencia de borradores y advertencia antes de perder datos de un formulario.

### 3.11 Tests y criterios de aceptación

**UX & navigation**
- [ ] La opción `Use as Gym plant` se diferencia inequívocamente de `Generate dataset`.
- [ ] La persona puede guardar un draft incompleto y retomarlo después de cerrar sesión.
- [ ] La columna de readiness explica *por qué* no se puede publicar y ofrece enlace a la etapa responsable.
- [ ] Todos los elementos visibles en la app están en inglés y admiten modo light/dark.
- [ ] Ninguna gráfica de resultados se dibuja sin una ejecución real desde `Generate`/`Run`.

**Functional Gym**
- [ ] `gymnasium.utils.env_checker.check_env` pasa para perfil validado.
- [ ] `reset(seed=...)` genera un estado inicial controlado y documenta los límites de determinismo.
- [ ] `step(action)` aplica realmente la acción al FMU y avanza sin reiniciar inadvertidamente el estado del solver.
- [ ] El tiempo físico es estrictamente creciente y el contador de pasos se mantiene consistente.
- [ ] `observation_space` y `action_space` coinciden en forma, límites y dtype con los datos efectivos.
- [ ] Recompensa, `terminated`, `truncated` y `info` se calculan y auditan de forma independiente.
- [ ] Acciones inválidas, caída del simulador, no convergencia, NaN y saturación tienen políticas explícitas y ensayadas.
- [ ] No se publican perfiles interactivos solo por poder hacer RPC `simulate`.

**PLECS capability, provenance & reliability**
- [ ] FMU 2.0/3.0 soportada solo en función de capacidades reales del runtime importador.
- [ ] Se prueban restricciones de licencia, SO y endpoint de PLECS antes de publicar.
- [ ] Batch/Surrogate/Gym quedan identificados como motores distintos y no intercambiables.
- [ ] Cambiar hash de circuito, FMU o mapeos invalida certificación previa.
- [ ] Las fallas se guardan con logs y resultado explícito; no se ocultan en promedios.
- [ ] El runner maneja exclusión mutua / cuotas y conserva consistencia al cancelar.
- [ ] El entrenamiento en RL Lab nunca se etiqueta como hardware verified o LIVE.

### 3.12 Fases de implementación sugeridas

1. **Foundation:** `PlantProfile`, `Readiness`, navegación, borradores/versiones, provenance y motor de jobs.
2. **PLECS batch proof:** RPC local, importar circuito de forma segura, barridos reproducibles, sin prometer Gym.
3. **FMU stepping spike (go/no-go):** una FMU pequeña exportada de PLECS 5.0, `reset/step`, 10 pasos con secuencia de acciones, prueba de SO/licencia/latencia. Bloquear publicación si falla.
4. **Gym wizard:** mapeo, timing, reward, reset, safety, panel de validación y checks de capacidad.
5. **Publish/integrate:** plant registry, RL Lab y Tuning Studio condicionados a capacidades reales.
6. **Surrogate:** ajuste sobre dataset, límites del dominio y evaluación externa frente a PLECS; nunca cambiar la fuente sin aprobación.
7. **Benchmark:** tiempo de pared, consistencia, acciones discretas, semillas, múltiples episodios, pruebas adversariales y accesibilidad.

### 3.13 Referencias técnicas consultadas

- PLECS 5.0 — [Functional Mockup Interface](https://docs.plexim.com/plecs/latest/functional-mockup-interface/). FMU Co-Simulation PLECS 5.0, endpoint, runtime/licencia, FMI 2/3.
- PLECS — [Simulation Scripts and RPC](https://docs.plexim.com/plecs/latest/scripting/). RPC en mismo host, configuración por defecto y simulaciones por script.
- Gymnasium — [Env API](https://gymnasium.farama.org/api/env/). `reset`, `step`, spaces, `terminated`, `truncated`, `close`.

**Material del producto revisado:** código de SPADE adjunto `SPADE-main (2).zip` y especificación de expansión previamente redactada. La propuesta no supone que ya exista una integración PLECS en el repositorio.

### 3.14 Pantallas adicionales indispensables para PLECS Plant Studio

Las seis etapas definen el alta de un entorno, pero **no** cubren por sí solas el ciclo de vida posterior. Se añaden cuatro vistas en la misma página principal: `My Plants`, `Create Plant`, `Compare with SPADE`, `Connection & Diagnostics`. Tres rutas internas complementarias (`Dataset`, `Surrogate`, `Gym`) comparten identidad/versiones de planta, no simulaciones o modelos silenciosamente intercambiables.

#### 3.14.1 My Plants — biblioteca y estado de perfiles

**Objetivo:** reabrir plantas, crear versiones y llegar a RL Lab sin reimportar el modelo.

| Elemento visible | Comportamiento esperado |
|---|---|
| `Search plants` | Filtra por nombre, ID y etiqueta; no modifica la planta. |
| `Source` | Distingue `Native SPADE`, `PLECS FMU`, `PLECS RPC dataset`, `Surrogate`. |
| `Readiness` | `Draft`, `Needs attention`, `Validated`, `Published`, `Stale`, `Unsupported`. |
| `Capabilities` | `Batch sweep`, `Interactive stepping`, `Can reset`, `RL compatible`; obtenidas del backend. |
| `Last verification` | Fecha, identificador del run y perfil/hash usado. |
| `Open configuration` | Retoma el wizard en el paso más relevante. |
| `Create new version` | Duplica configuración editable; nueva versión no hereda validación si cambian artefactos o mapeos. |
| `Open in RL Lab` | Solo si existe entorno publicado y realmente compatible. |
| `Compare with SPADE` | Abre comparación con selecciones de planta prellenadas, sin ejecutarla automáticamente. |

**Estados:** sin plantas: `No PLECS plant profiles yet. Create your first plant to get started.`; planta obsoleta: `This profile changed since its last verification. Revalidation is required.`; conexión ausente: se puede navegar, consultar resultados y editar borradores; se bloquean solo acciones que necesitan el simulador.

#### 3.14.2 Ruta Dataset — simulaciones por barridos completos

La ruta se inicia desde `Generate simulation datasets` y **no exige** configurar reward/reset de Gym. Flujo: `Select circuit → Select variables → Choose sweep method → Review budget → Start sweep → Inspect results → Export dataset`.

**Formulario:** circuito/.plecs accesible por el backend; nombre del experimento; parámetros que variarán con unidad y rango; método `Grid` o `Latin hypercube`; número de condiciones o muestras; semilla; tiempo simulado por ejecución; variables de salida; exclusiones/errores; límite de recursos; política de convergencia. `Start sweep` crea un job persistente.

**Resultado:** tabla de condiciones completadas/fallidas, curvas originadas en simulaciones ejecutadas, distribución de tiempos de pared, lista de errores del solver, metadata del circuito/versión y exportación de tabla/manifest. No interpolar las simulaciones fallidas. Nunca marcar el dataset como entorno interactivo.

```text
PLECS Plant Studio / Generate simulation datasets
┌────────────────────────────┬─────────────────────────────────┐
│ CIRCUIT AND SWEEP          │ RESULTS                         │
│ Circuit [Select profile ▾] │ Status: Not generated           │
│ Parameters [+ Add row]     │ [Plot shown after Start sweep]  │
│ Method [Grid / LHS ▾]      │ [Conditions: completed / failed]│
│ Seed [ ]  Budget [ ]       │ [Failures and logs]             │
│ [Start sweep]              │ [Export dataset / manifest]     │
└────────────────────────────┴─────────────────────────────────┘
```

#### 3.14.3 Ruta Surrogate — sustituto con dominio de validez explícito

Flujo: `Select existing PLECS dataset → Choose target signals → Select surrogate family → Fit surrogate → Validate vs held-out PLECS runs → Register surrogate → Use as surrogate Gym (if validated)`.

**Campos:** dataset fuente, esquema I/O, familia del aproximador implementado, split por escenarios/condiciones para evitar filtración, presupuesto, semillas y criterios mínimos de aceptación. **El frontend no prescribe una familia numérica sin backend**: solo lista algoritmos realmente disponibles.

**Paneles de evaluación:** error por canal, error por condición, latencia de inferencia, cobertura del dominio, condiciones fuera del rango conocido, detalle de fallos y diferencia con modelo PLECS en grilla independiente. Resaltar `SIMULATED · SURROGATE` y el error de ajuste/validación. No declarar la sustitución equivalente a PLECS, ni ejecutar cambios sin selección explícita.

**Salida:** perfil de surrogate versionado y, **solo si el runtime es compatible**, Gym `SURROGATE-Gym` con contrato `reset/step` propio y sus limitaciones.

#### 3.14.4 Compare with SPADE — pantalla de fidelidad

**Propósito:** comparar las respuestas de planta nativa SPADE y circuito PLECS a la misma excitación y condiciones **solo cuando las señales sean físicamente comparables**.

**Entrada obligatoria:** perfiles y versiones; señales emparejadas; unidades; excitación/referencia; condiciones de carga; muestreo y eje temporal; condiciones iniciales equivalentes; métricas a calcular. Al no coincidir físicamente las condiciones, mostrar aviso y permitir inspección sin afirmar equivalencia.

```text
┌───────────────────── SPADE vs PLECS ─────────────────────────────────┐
│ Native plant [Select ▾]         PLECS plant [Select ▾]                │
│ Signal map [Review]   Condition grid [Review]   [Run comparison]      │
├─────────────────────────────┬────────────────────────────────────────┤
│ Signal overlay              │ Error per condition                    │
│ [Empty until executed]      │ [Empty until executed]                  │
│ Current / Voltage / Speed   │ [RMSE / peak error / mismatch]         │
├─────────────────────────────┴────────────────────────────────────────┤
│ [Metrics table] [Heatmap] [Failed cases] [Provenance] [Logs]          │
│ Source A: SIMULATED   Source B: SIMULATED · PLECS                     │
│ Run IDs, hashes, engine/version, aligned time grid, Copy command      │
└──────────────────────────────────────────────────────────────────────┘
```

**Interacciones:** hover sincronizado de curvas; elección de señales desde una lista con unidades; selección de condición en tabla y resaltado del intervalo; exportación de resultados y definición exacta de RMSE/error; datos faltantes como `Missing`, divergencia como `Failed`, solver sin convergencia identificado. Comparaciones negativas se muestran como resultados completos, no se ocultan.

**Estado vacío exacto:** `No comparison has been generated yet. Select compatible plants and run the comparison.`

#### 3.14.5 Connection & Diagnostics — sin ambigüedad entre conexión y simulación

El diagnóstico muestra por separado:

- `RPC connectivity`: reachability, host local del backend, respuesta real y error de configuración.
- `FMU import support`: versión FMI, OS/platform binaries, herramientas/runtime disponibles, licencia cuando sea verificable.
- `Step capability`: capacidad real de inicializar, actuar, avanzar en tiempo y reiniciar.
- `Circuit readiness`: mapeos, límites, guardado de perfiles.
- `Environment readiness`: aprobación de tests para Gym. **No se deriva automáticamente de RPC connectivity**.

La pantalla propone acciones según error, por ejemplo `Open setup instructions`, `Copy diagnostics`, `Retry check`, `Save report`. No incluye credenciales ni paths privados en reportes compartibles.

### 3.15 Tabla de capacidades y controles dependientes

| Función en UI | RPC batch disponible | FMU stepping verificado | Surrogate Gym validado |
|---|---:|---:|---:|
| `Start sweep` | Sí | No necesariamente | No |
| `Fit surrogate` | Con dataset válido | Con dataset válido | No aplica |
| `Run PLECS step test` | **No** | Sí | No |
| `Publish PLECS Gym` | **No** | Sí | **No** |
| `Publish surrogate Gym` | **No** | No | Sí |
| `Train agent on PLECS` | **No** | Sí | **No** |
| `Train agent on surrogate` | **No** | No | Sí |
| `Compare with native SPADE` | Sí, si señales/condiciones alineadas | Sí, si señales/condiciones alineadas | Sí, con etiqueta propia |

`Sí` significa **habilitable una vez validados los demás prerrequisitos**, no que la función esté implementada ahora. Toda capacidad requiere un informe de compatibilidad del backend; la UI no debe inferirla por extensión del archivo.

### 3.16 Recorridos completos observables por el investigador

**Recorrido 1 — PLECS → Gym → RL:** `Choose goal` → `Verify FMU stepping backend` → `Upload FMU` → `Map I/O` → `Configure time/reset/reward/safety` → `Run verification` → `Publish environment` → `Open RL Lab` → `Train` → `Compare PI vs RL` → `Save candidate` → `Request human approval`. Cada transición conserva referencias a perfiles y runs, no datos binarios solo en memoria de Streamlit.

**Recorrido 2 — PLECS → Dataset → Surrogate:** `Configure RPC` → `Import circuit` → `Select parameter sweep` → `Generate` → `Inspect failed cases` → `Fit surrogate` → `Compare against held-out PLECS conditions` → `Register surrogate` → `Train/evaluate on surrogate` → `Verify selected policy with PLECS when actual interactive coupling is available`.

**Recorrido 3 — PLECS ↔ SPADE:** `Select compatible native/SPADE profiles` → `Review signal map / units / initial conditions` → `Run both simulations` → `Align timelines` → `Inspect overlays, error heatmap and failed cases` → `Export a reproducible report`.

### 3.17 Estados de validación y bloqueo dentro del wizard

| Paso | CTA principal | Validación bloqueante | Comportamiento ante error |
|---|---|---|---|
| Goal | `Continue` | Ninguna, salvo seleccionar objetivo | Explicar rutas sin sugerir Gym listo |
| Engine | `Test capabilities` | Para Gym: stepping/reseteo debe existir para poder publicar, pero puede continuarse en draft | `Not supported` + enlaces al setup |
| Circuit | `Import circuit` | Artefacto y metadatos validados; hash verificable | Conservar draft, sin falsos canales |
| Signals | `Validate mapping` | Acción controlable, observaciones legibles, unidades/límites | Errores por fila, ir al canal problemático |
| Gym behavior | `Validate definition` | Tiempo, reset, reward finita, safety y termination definidas | Mensajes junto a cada pestaña |
| Verify | `Run verification` | Pruebas del backend; hardware/software disponible | Run fallido registrado, bloqueo de publish |
| Publish | `Publish to Environments` | Validación vigente para hash/versiones/config | Deshabilitado con causa y acceso directo a corrección |

**Regla:** `Save draft` nunca equivale a `Publish`; y publicar no equivale a desplegar un controlador en hardware.


## 4. Tuning Studio — optimización de controladores

### Paso 1. Select system

Elegir `Plant`, versión de perfil (nativo, PLECS, surrogate), controlador (PI, MPC, DPC u otros compatibles), condiciones de carga/referencia/velocidad y grilla de evaluación independiente. Mostrar capabilities; bloquear combinaciones incompatibles.

### Paso 2. Search space

Tabla de parámetros: nombre, valor baseline, lower bound, upper bound, distribución/escala lineal-log, fijo/optimizable; validación de rangos y unidades, constraints técnicas (corriente máxima, potencia, estabilidad, duty cycle, etc.).

### Paso 3. Objectives & method

Métricas: overshoot, RMSE, settling time, CPU/runtime cost y restricción personalizada respaldada por un evaluator. Señalar si son objetivos minimizados o maximizados. Método `PSO`, `Bayesian Optimization`, `Multi-objective`. Presupuesto: nº evaluaciones, seed, stop criteria. Estimación solo tras benchmark.

### Paso 4. Execute

`Start tuning` encola job persistente. UI enseña `Best-so-far vs evaluations`, tasa de fallos, violaciones de restricciones, Pareto (multiobjective), log y control de cancelación. Hover sobre punto de Pareto muestra parámetros, objetivos y enlace a respuesta temporal **que realmente haya sido calculada y almacenada**.

### Paso 5. Review & approve

Comparar baseline vs candidato sobre **idéntica** grilla fijada antes de entrenar; para evaluación honesta debe haber un conjunto no usado al optimizar o un aviso `Evaluation uses tuning conditions`. Métricas por condición, tendencias negativas, condiciones divergentes, heatmap, incertidumbre/semillas según disponibilidad. `Save candidate` no modifica defaults. `Use as default` requiere `ApprovalGate`, confirmación explícita y registro inmutable.

### Wireframe Tuning Studio

```text
┌──────────────────────────── Tuning Studio ────────────────────────────┐
│ Design & Control / Tuning Studio    Plant: [VSC / DPC ▾]              │
│ [Setup] [Objectives] [Run progress] [Review]                         │
├────────────────────────────────┬─────────────────────────────────────┤
│ CONFIGURATION                  │ RESULTS                             │
│ Controller [PI ▾]              │ [No tuning run yet]                 │
│ Variables                      │                                     │
│ Kp   [min] — [max]             │  Best objective vs evaluations      │
│ Ki   [min] — [max]             │  [display after execution]          │
│ Objectives                     │                                     │
│ [x] RMSE    [x] Overshoot      │  Pareto frontier                    │
│ [x] Settling time              │  [display for multi-objective runs] │
│ Constraints                    │                                     │
│ Current < [A]                  │  Hover: objectives, params, trace   │
│ Algorithm [Multi-objective ▾]  │  [Selected candidate details]       │
│ Budget    [evaluations]        │                                     │
│ [ Start tuning ]               │                                     │
├────────────────────────────────┴─────────────────────────────────────┤
│ BASELINE vs CANDIDATE  [Condition grid ▾]                            │
│ [Metric table] [Signal overlay] [Error heatmap] [Failed conditions]  │
│ [Save candidate] [Use as default — requires human approval]         │
│ Provenance: [source] [run ID] [config hash] [Copy command]            │
└──────────────────────────────────────────────────────────────────────┘
```

## 5. RL Lab — control mediante aprendizaje por refuerzo

### Paso 1. Environment

Seleccionar entorno Gymnasium con especificación explícita: observaciones (señal, unidad y shape), acciones (discretas o continuas, límites, saturaciones), timestep, recompensas y criterios de terminación. Mostrar enlace `Copy Python snippet` y `Preview environment contract`. DQN solo habilitado en acciones discretas; DDPG en continuas. Si se plantea discretización para DQN, solicitar configuración y guardar la transformación.

### Paso 2. Reward editor

Editor de términos con peso, escala, fórmula, sentido y unidad: tracking error, overcurrent penalty, smooth action penalty. Vista previa de fórmula normalizada y contribuciones explicadas. Antes de entrenar no se inventan curvas: una `Preview on baseline episode` requiere ejecutar el episodio y genera un run de evaluación.

### Paso 3. Training

Seleccionar DQN/DDPG, seed, episodios, longitud, budget, eval interval, checkpoints. `Start training` encola job. Entrenamiento: recompensa por episodio, pérdida critic/actor cuando corresponde, tasas de violaciones de restricciones, episodios de validación y enlace a `Playback actual evaluation episode` cada N episodios. El “playback” utiliza datos de runs ejecutados; no es animación decorativa.

### Paso 4. Benchmark

Comparar con PI/MPC/DPC solo cuando compatibles con esa planta y la misma grilla. Mismas perturbaciones, referencias, límites, timestep y métricas. Mostrar divergencias, desempeño inferior y sensibilidad a la semilla. `Save policy candidate` no despliega la política; aprobación humana previa a cualquier default.

### Wireframe RL Lab

```text
┌────────────────────────────── RL Lab ────────────────────────────────┐
│ Design & Control / RL Lab   [Environment] [Reward] [Training] [Eval]  │
├──────────────────────────────┬───────────────────────────────────────┤
│ ENVIRONMENT                  │ REWARD COMPOSITION                    │
│ Plant       [Buck ▾]         │ Tracking     weight [ ]               │
│ Actions     [Continuous]     │ Overcurrent  weight [ ]               │
│ Algorithm   [DDPG ▾]         │ Action Δ     weight [ ]               │
│ Episodes    [   ]            │ R = Σ normalized weighted terms      │
│ Seed        [   ]            │ [Preview on baseline episode]        │
│ [Start training]             │ [No evaluated episode yet]            │
├──────────────────────────────┴───────────────────────────────────────┤
│ Training: [Episode reward] [Loss] [Constraint violations] [Replay]   │
│ Evaluation: [PI] [MPC] [DPC] [RL] on SAME condition grid             │
│ [Failed/divergent conditions] [Save candidate] [Request approval]    │
└──────────────────────────────────────────────────────────────────────┘
```

## 6. Sistema de ejecución persistente — Job Runner y Experiment Runs

**Necesidad:** mantener colas PLECS, optimización y RL aunque cambie la pantalla, se haga un rerun Streamlit o se cierre la pestaña. `st.session_state` solo conserva navegación/inputs transitorios; no se usa como fuente de verdad de ejecuciones.

**Backend mínimo sugerido:** `job_service` persistente con base SQLite en desarrollo local, archivos/artifacts bajo directorios propios por `run_id`, worker separado del proceso Streamlit. Para concurrencia multiusuario o cargas avanzadas, cola Redis/RQ (o equivalente) + DB transactional. Un solo PLECS RPC activo por modelo/instancia, respetando exclusión. `JobAdapter` consulta/suscribe a progreso; Polling moderado o fragmento auto-refresh **sin** relanzar jobs.

**Estados:** `draft → queued → preparing → running → cancelling → completed | partially_completed | failed | cancelled`; `resumable` como atributo (no estado), solo si hay checkpoint real y backend soporta recuperación. Si se pierde RPC durante una simulación bloqueante, `Cancel requested` hasta confirmación de parada, no prometer cancelación inmediata. ETA nulo hasta tener historial suficiente; mostrar `ETA unavailable` sin estimación fabricada.

**Trazabilidad:** identificador universal, UTC, hash completo de configuración, versión del entorno y simulador, seed, commit si disponible, hashes de datasets/modelos, señales y unidades, errores, comando/manifest reproducible. No imprimir secretos, rutas sensibles ni URLs con credenciales en `Copy command`.

### Wireframe queue drawer + Runs page

```text
┌────────────────────────────── Experiment Runs ───────────────────────┐
│ [All] [Running] [Completed] [Failed] [Cancelled]    Search [       ] │
│                                                                      │
│ JOB ID        TYPE           PLANT     STATE       PROGRESS ACTION    │
│ [loaded jobs from DB only; no fabricated examples]                  │
│                                                                      │
│ ┌──────────────────────────── Selected job ────────────────────────┐ │
│ │ Progress [████░░...] (if measurable)  ETA: unavailable           │ │
│ │ Phase: ...    Completed simulations: ... / ...                  │ │
│ │ [Cancel request] [Resume if checkpoint exists] [Open run]       │ │
│ │ LOGS ─────────────────────────────────────────────────────────   │ │
│ │ [real timestamped worker events]                                │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│ Run details [Config] [Metrics] [Artifacts] [Logs] [Provenance]       │
│ [Compare selected] [Export manifest] [Copy reproducible command]    │
└──────────────────────────────────────────────────────────────────────┘

GLOBAL (sidebar)  Jobs: 2 running  → opens compact queue drawer/panel
```

## 7. Comparación universal de ejecuciones — Run Comparison

**Inputs:** runs seleccionados, grid frozen, signal map, unit conversion, timebase alignment, baseline, metric definitions. **Outputs:** tabla general, overlays, heatmap `condition × model/controller`, tabla de divergence/errors, diff de config y señales, export reproducible.

**Validaciones previas**: no comparar perfiles con canales incompatibles o grids dispares; si faltan condiciones, mostrar `Not comparable`/`Missing condition`, nunca imputar una mejora; mostrar mismo conjunto de test en ambos. Si solo una parte coincide, ofrecer intersección de condiciones y cuantificar cuántas quedaron fuera.

**Heatmap:** tooltips contienen condición, métricas, tipo de run, origen y si la simulación convergió; patrón y texto para divergencia, no solo rojo.

## 8. Nuevas plantas y módulos AI

### 8.1 Buck y DAB

Ruta única `Simulation / DC-DC Converters` con tabs `Buck` y `DAB`, componentes base compartidos: `PlantParameters`, `ControllerParameters`, `OperatingConditions`, `GenerateButton`, `SignalChart`, `MetricCards`, `SourceBadge` y `RunComparison`. Tab `Modulation` solo si planta lo admite. Para DAB, selector de estrategia (incluida TPS cuando el modelo soporte triple phase shift), diagrama de desfases real derivado de los parámetros seleccionados y señales ejecutadas. Una predicción de modulación nunca se ejecuta automáticamente: `AI SUGGESTION — REQUIRES VERIFICATION` y `Compare suggested vs baseline` dispara nuevas ejecuciones bajo las mismas condiciones. PLECS profile válido puede ser motor de la planta.

### 8.2 OOD en AI Diagnostics

Tres categorías de salida **mutuamente distinguibles**:

1. `Known condition` — clase soportada y decisiones de confianza/umbral válidas.
2. `Uncertain` — confianza insuficiente entre clases entrenadas.
3. `Unknown condition — REQUIRES VERIFICATION` — OOD score supera umbral calibrado de OOD, independientemente de la máxima confianza para una clase conocida.

**Regla de decisión sugerida:** si entrada inválida: `Input invalid` y no predecir. Si OOD flag: `Unknown condition`. Si no OOD pero baja confianza: `Uncertain`. Si ambos checks pasan: `Known condition`. Si no se dispone de detector OOD validado: el tercer estado se mantiene deshabilitado (`OOD detection unavailable`) y jamás se inventa un score.

### Wireframe Unknown condition

```text
┌──────────────────────────── AI Diagnostics ───────────────────────────┐
│ Input: [Run / Uploaded file]  [Analyze]  Source: EXTERNAL FILE        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ [?] UNKNOWN CONDITION — REQUIRES VERIFICATION                    │ │
│ │ The input differs from the conditions represented in training.   │ │
│ │ OOD score: [from model]    OOD threshold: [calibrated threshold]  │ │
│ │ Known-class confidence: [if calibrated]  Model: [version]         │ │
│ │                                                               │ │
│ │ Secondary simulated-agent hypothesis: [if supported]            │ │
│ │ Not a confirmed diagnosis. [Inspect signals] [Export evidence]   │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│ [Signal views] [OOD method and validation] [Provenance]               │
└───────────────────────────────────────────────────────────────────────┘
```

`Uncertain` emplea etiqueta y explicación diferentes: `The classifier cannot distinguish reliably among its known classes.` No mostrar “healthy” en verde si OOD está activo.

### 8.3 PINN Parameter Identification

Dentro de `Simulation / VSC / DPC`, tab `Parameter Identification`. Elegir origen de datos (simulada / PLECS / external file), ventana de ondas, rango temporal, estimador y parámetros `L`, `C`, `R_f`. Mostrar nominal vs estimado, intervalo e **identificación del método de incertidumbre** (bootstrap, ensemble u otro, si implementado). Tendencia longitudinal de C solo si las ventanas son comparables y datos disponibles. Texto permanente: `Parameter estimate — not a validated fault diagnosis.` Los valores nominales no son el ground truth cuando se evalúa contra mediciones reales.

### 8.4 IGBT Prognostics

Nueva `Intelligence / Prognostics`. Origen primario de dataset de envejecimiento acelerado de NASA: `MEASURED · EXTERNAL DATASET`, ficha con fuente, identificador, licencia comprobada, dispositivos, condiciones de ensayo y política de splits. Elegir `Device / Series`, graficar señales de envejecimiento, inferencia RUL y banda `Estimated 90% prediction interval` **solo si el método realmente produce ese nivel**; reportar su cobertura evaluada o `Coverage not validated`. Superponer vida real solo cuando sea conocida en conjunto de evaluación; impedir contaminación entre entrenamientos y series de la misma unidad. Texto: `This dataset is separate from SPADE's simulated plant diagnostics.` No asumir una licencia si no está documentada: `License: verification required`.

### 8.5 Environments

Tabla `Workspace / Environments` con columna de planta, versión, fuente, observación (shape+unidades), acción (type/bounds), recompensa default, límites físicos, termination conditions, compatible algorithms y `Copy Python snippet`. El botón copia código que instancia el ID realmente registrado en Gymnasium, no uno ficticio. Entornos aún no disponibles: `Not registered` sin botón funcional.

### 8.6 Algorithm Guide

Wizard de tres preguntas (`Life-cycle phase`, `Task`, `Data source`) devuelve familia recomendada, *why / assumptions / limitations* y link a la página que la implementa. Mostrar `Not available in this version` para sugerencias sin implementación y nunca simular enlaces rotos.

## 9. Biblioteca de componentes y estados

Crear sobre patrones existentes en `viz/components/`:

| Componente | Props / responsabilidad | Empty | Loading | Error | Success | Negative result |
|---|---|---|---|---|---|---|
| `SourceBadge` | `origin, run_id, dataset_hash, license?, tool_ver?` | `No data source selected` | `Checking source…` | `Source cannot be verified` | `SIMULATED · PLECS` etc. | `Origin known; run failed` |
| `ConnectionStatus` | `provider, host, health, capabilities, last_checked` | `PLECS not configured` | `Testing connection…` | `PLECS RPC unavailable` | `PLECS available (RPC)` | `Connected; required capability unavailable` |
| `JobCard` | `job_id, type, phase, progress?, ETA?, log_ref, cancellable, resumable` | `No jobs yet` | `Job queued` / `Running` | `Job failed — view log` | `Job completed` | `Completed with unsuccessful conditions` |
| `RunComparison` | `runs, grid, metric_specs, units, align` | `Select at least two runs` | `Loading saved results…` | `Runs are not comparable` | `Comparison ready` | `Candidate did not improve` |
| `RewardEditor` | `terms, weights, limits, preview_run?` | `Add at least one reward term` | `Evaluating reward preview…` | `Invalid reward scale` | `Reward definition valid` | `Unsafe reward / high constraint penalty` |
| `PlantSelector` | `profiles, capabilities, requirements` | `No plant profiles available` | `Loading profiles…` | `Profile invalid` | `Plant compatible` | `Plant out of surrogate envelope` |
| `ParameterBoundsEditor` | `parameters, units, min/max, constraints` | `No tunable parameters` | `Validating bounds…` | `Lower bound exceeds upper` | `Bounds valid` | `Search range violates physical limit` |
| `ApprovalGate` | `decision, delta, human_actor, evidence` | `No candidate selected` | `Checking evidence…` | `Approval record not saved` | `Human approval recorded` | `Approval blocked due to worse safety limits` |
| `ProvenanceFooter` | `origin, run, hash, command, license?` | `No run generated` | `Preparing manifest…` | `Reproducibility manifest unavailable` | `Copy command` | `Failed run with traceback reference` |
| `FailurePanel` | `condition, failure_kind, error, suggested_action` | `No failures recorded` | `Checking failed runs…` | `Failure records unavailable` | `All conditions completed` | `Divergence / no convergence / violated constraint` |

**Estilos:** usar `theme/tokens.py`, Material icons existentes, estados con icono + texto. Componente `PromotionGate` existente permanece para modelos; no esconder criterios que bloquean promoción.

## 10. Microcopy (todas las cadenas visibles en la aplicación van en inglés)

**Page headings:** `PLECS Plant Studio`; `Experiment Runs`; `Environments`; `Tuning Studio`; `RL Lab`; `DC-DC Converters`; `Parameter Identification`; `Prognostics`; `Algorithm Guide`.

**Connect/import:**

- `PLECS is not configured on this machine.`
- `Enable the RPC server in PLECS Standalone and try again.`
- `PLECS RPC must be reachable from the machine running SPADE.`
- `Connection established. This indicates simulator availability, not measured telemetry.`
- `Could not connect to PLECS. Check the RPC setting and port.`
- `Circuit uploaded. Review the model variables and output ports before mapping.`
- `Automatic circuit inspection is unavailable. Enter the interface manifest manually.`
- `Required control action is not mapped.`
- `Incompatible units: expected A, received V.`
- `Co-simulation requires a compatible coupling adapter. RPC batch mode is available.`
- `Surrogate outside validated operating range. Use a PLECS simulation for this condition.`

**Jobs/runs:**

- `Nothing has been generated yet. Configure an experiment and select Generate.`
- `Job queued. You can leave this page while the worker runs.`
- `Running experiment — results will appear after computation.`
- `ETA unavailable until enough evaluations are completed.`
- `Cancellation requested. The current solver call may need to finish.`
- `Resume unavailable — no recoverable checkpoint was saved.`
- `Run completed with failed conditions. Inspect the failure table.`
- `This controller diverged under the selected condition.`
- `No improvement over the baseline was measured.`
- `These runs do not share the same evaluation grid.`
- `The evaluation reused tuning conditions; results are not an independent generalization estimate.`
- `Copy reproducible command` / `Open run` / `Compare runs` / `View log`.

**AI:**

- `AI SUGGESTION — REQUIRES VERIFICATION`
- `Known condition` / `Uncertain — low confidence` / `Unknown condition — REQUIRES VERIFICATION`.
- `The input differs from the conditions represented in training.`
- `The classifier cannot distinguish reliably among its known classes.`
- `Secondary hypothesis from simulated evidence — not a confirmed diagnosis.`
- `OOD detector unavailable for this model version.`
- `Parameter estimate — not a validated fault diagnosis.`
- `Estimated 90% prediction interval (coverage not independently validated).`
- `MEASURED · EXTERNAL DATASET` / `Dataset source` / `License: verification required`.

**Approval:**

- `Save as candidate` / `Request approval` / `Use as default`.
- `Human approval required. Review safety constraints and evaluation results.`
- `The candidate performs worse on at least one required condition.`
- `I reviewed the evidence and approve this change.`
- `Approval blocked. No default has been changed.`

## 11. Accesibilidad y responsive

- Respetar contraste WCAG AA como criterio: texto normal al menos 4.5:1, texto grande/UI esencial al menos 3:1; verificar colores en claro y oscuro con pruebas automatizadas y visuales, no asumir que los tokens actuales cumplen sin medir.
- Los estados requieren icono + texto, no exclusivamente rojo/verde. En heatmap, emplear símbolos/patrones `failed`, `not comparable`, `partial` y tabla textual accesible.
- Todos los inputs tienen label, help text y error asociado; recorrido completo por teclado; orden lógico de foco; visible focus; gráficos con título, unidades, tabla exportable y summary textual.
- Progreso: `aria-live` moderado en actualizaciones importantes; evitar re-render agresivo que haga perder foco. Respeto a `prefers-reduced-motion`, incluso en playback de RL y animación de motor.
- Pantalla >= 1200 px: sidebar + panel configuración 35% / resultados 65%; heatmap/series lado a lado cuando haya ancho.
- Pantalla 768–1199 px: navegación colapsable, panel de configuración arriba, resultados abajo, tabs para densidad.
- Pantalla < 768 px: wizard vertical, tarjetas de modo apiladas, inputs ancho completo, barra CTA inferior sin tapar contenido, leyendas de gráfico reducibles, tablas con scroll horizontal etiquetado, reemplazar hover-only de Pareto por selector de filas o puntos.
- Touch targets >= 44x44 px; no confiar en hover para descubrir los detalles del Pareto en móvil.
- Ningún valor numérico debe acortarse hasta perder unidades ni indicadores de dispersión/intervalo.

## 12. Mapa técnico de integración en el repositorio

```text
src/driveflow/
├── viz/
│   ├── app.py                               + new navigation entries
│   ├── pages/
│   │   ├── circuits_plecs.py                NEW
│   │   ├── experiment_runs.py               NEW
│   │   ├── environments.py                  NEW
│   │   ├── dc_dc_converters.py              NEW
│   │   ├── tuning_studio.py                 NEW
│   │   ├── rl_lab.py                        NEW
│   │   ├── prognostics.py                   NEW
│   │   └── algorithm_guide.py               NEW
│   ├── views/
│   │   ├── plecs_plant_studio.py             NEW
│   │   ├── plecs_comparison.py              NEW (subview/tab)
│   │   ├── experiment_runs.py               NEW
│   │   ├── environments.py                  NEW
│   │   ├── dc_dc_converters.py              NEW
│   │   ├── tuning_studio.py                 NEW
│   │   ├── rl_lab.py                        NEW
│   │   ├── prognostics.py                   NEW
│   │   ├── algorithm_guide.py               NEW
│   │   ├── ai_diagnostics.py                EXTEND OOD
│   │   └── vsc_dpc.py                       EXTEND PINN tab
│   ├── adapters/
│   │   ├── plecs_adapter.py                 NEW
│   │   ├── environment_adapter.py           NEW
│   │   ├── tuning_adapter.py                NEW
│   │   ├── rl_adapter.py                    NEW
│   │   ├── converter_adapter.py             NEW
│   │   ├── prognostics_adapter.py           NEW
│   │   ├── ood_adapter.py                   NEW
│   │   └── jobs_adapter.py                  NEW
│   ├── components/
│   │   ├── source_badge.py                  NEW/REFACTOR
│   │   ├── connection_status.py             NEW
│   │   ├── job_card.py                      NEW
│   │   ├── run_comparison.py                NEW
│   │   ├── reward_editor.py                 NEW
│   │   ├── plant_selector.py                NEW
│   │   ├── parameter_bounds.py              NEW
│   │   ├── approval_gate.py                 NEW
│   │   └── provenance_footer.py             NEW
│   └── theme/tokens.py                      EXTEND provenance/status tokens
├── experiments/                              NEW durable runs and profiles
│   ├── schemas.py
│   ├── run_store.py
│   └── plant_profiles.py
├── jobs/                                     NEW worker + queue facade
│   ├── schemas.py
│   ├── service.py
│   ├── worker.py
│   └── persistence.py
├── integration/                              NEW external PLECS coupling
│   ├── plecs_rpc.py
│   └── fmi_adapter.py                       ONLY when validated
└── [domain packages]
    └── physicists implement simulations, control, surrogate,
        optimizer, RL, OOD, PINN and RUL independently of UI
```

**Contracts before UI**: `PlantCapability`, `ExperimentSpec`, `Job`, `Run`, `SignalSeries`, `ApprovalRecord`, `ErrorCode`. Los adapters deben ofrecer funciones puras/servicios idempotentes `validate_*`, `submit_*`, `get_*`, `list_*`, `cancel_*`, `resume_*`, `compare_*`. El worker no usa Streamlit. Exigir capacidad (`available`, `unsupported`, `requires_configuration`) en lugar de presentar controles rotos.

## 13. Fases y criterios de aceptación

### Fase 0 — Fundaciones (indispensable)
- Persistir Job/Run/Experiment/PlantProfile fuera de la sesión; conservar compatibilidad con runs existentes de la sesión, migrando o marcando legacy según sea viable.
- SourceBadge, ProvenanceFooter, ConnectionStatus, JobCard, RunComparison, ApprovalGate con light/dark.
- Contratos de datos y grilla frozen; job worker desacoplado; historial con error por condición.
- **Aceptar cuando**: lanzar job, cambiar de página/cerrar pestaña, regresar y ver estado real, ejecutar cancelación honesta, recuperar artefactos/logs, copiar comando y conservar hash.

### Fase 1 — PLECS Batch + validación de stepping y comparación
- Asistente PLECS de 6 etapas con tres rutas según objetivo. Habilitar ruta de dataset mediante RPC local, perfiles versionados, mapeo I/O, barridos reproducibles, resultados parciales y comparación SPADE vs PLECS. La ruta Gym interactiva solo se habilita si el adaptador de stepping supera su validación; no depende del éxito de RPC.
- **Aceptar cuando**: conexión real, circuit file hash, runs reproducibles, comparación con tiempos alineados, visualización de fallas, y fallback si falta PLECS. No declarar co-simulation disponible.

### Fase 2 — Nuevas plantas + entornos
- Buck/DAB con `Generate` y controles especializados, catálogo Gymnasium con snippets reales; adapters registran capabilities. Surrogate solo después de dataset y validación.
- **Aceptar cuando**: cada plot proviene de run efectivo; PLECS/native/surrogate comparten contratos; selección incompatible se bloquea.

### Fase 3 — Optimización + RL
- Tuning Studio, optimizadores, reward editor, worker de entrenamiento, comparación sobre misma grilla, ApprovalGate.
- **Aceptar cuando**: Pareto/convergencia se genera a partir de runs; malas soluciones permanecen visibles; DQN/DDPG respetan action spaces; ninguna promoción cambia default sin aprobación.

### Fase 4 — IA avanzada
- OOD condicionado a validación, PINN con límites de incertidumbre, IGBT RUL medido con procedencia/licencia y evaluación independiente.
- **Aceptar cuando**: Unknown ≠ Uncertain, inferencias etiquetadas, RUL no se confunde con simulación ni con diagnóstico de rodamientos, intervalos informan método/cobertura.

### Fase 5 — Ampliación y endurecimiento del acoplamiento avanzado
- Mejorar escalabilidad y eficiencia de Co-simulation mediante FMI/u otro backend **si la prueba técnica de stepping ya confirmó viabilidad**; ejecutar pruebas de sincronización, latencia por paso, estabilidad y reproducibilidad.
- **Aceptar cuando**: la API soporta control real a cada paso, handshake y timestamps verificados, recuperación de errores, y se muestra `SIMULATED · PLECS`, nunca `LIVE`.

## 14. Matriz de pruebas de UI y aceptación

| Caso | Esperado |
|---|---|
| Cambiar de página durante RL | Worker continúa; estado persistido, sin entrenamiento duplicado |
| Recargar browser | Recupera jobs y runs desde almacenamiento persistente |
| PLECS RPC no configurado | Error accionable; resto de SPADE funcional |
| PLECS en máquina remota | Bloqueo y explicación del requisito local |
| Importar modelo sin puertos necesarios | Wizard bloquea ejecución, mapping editable |
| Barrido parcialmente fallido | Mostrar éxitos/fallos por condición, no imputar datos |
| Surrogate fuera del rango | Advertencia/block según política registrada; enlazar al barrido real |
| Stepwise sin FMI validado | `Unavailable`, botón disabled, razón visible |
| DQN con acciones continuas | Bloquear y explicar necesidad de acción discreta |
| Convergencia sin mejora | `No improvement`, sin promoción encubierta |
| Comparación con grids dispares | Bloqueo o intersección explícita, no ranking engañoso |
| OOD activado con confianza alta en clase conocida | Priorizar `Unknown condition` |
| Detector OOD ausente | No generar tercera clase ficticia |
| PINN con incertidumbre no validada | Método declarado / aviso de limitación |
| RUL dataset sin licencia verificada | Etiqueta explícita de verificación pendiente |
| Cambiar default sin gate | Operación negada por backend, no solo botón deshabilitado |
| Modo oscuro/claro/móvil/teclado | Contraste, foco, unidad, legibilidad, sin hover obligatorio |
| Copiar comando reproducible | Incluye config+seed+versiones, sin secretos, rutas seguras |

## 15. Decisiones de UX que conviene mantener constantes

- `Generate` siempre es una acción explícita, no un side effect de widget.
- `Simulated` identifica origen; `Available` identifica **conectividad a software**; son dimensiones separadas.
- Plant/Experiment/Run forman una jerarquía consistente en toda la app.
- Un nuevo módulo primero registra sus capacidades y luego aparece como seleccionable en las páginas compatibles.
- El sistema evita afirmar que se ha validado con hardware físico, ha probado un software que falta o conoce una vida útil real donde no existe ground truth.
- Mantener result charts visibles y reproducibles tanto para éxitos como para fallos.
- Ninguna herramienta de IA ejecuta cambios de control por sí sola; las propuestas se guardan como `candidate` y pasan por aprobación humana.

---

**Secuencia sugerida:** fundamento persistente → prueba técnica de FMU stepping y RPC batch → wizard y verificación Gym → publicación y RL. Si no existe stepping compatible, la ruta dataset sigue siendo válida sin confundirse con el objetivo Gym.



## 16. Sistema visual y lineamientos de diseño

### 16.1 Personalidad visual

SPADE debe verse como una **herramienta científica profesional**, no como un panel de telemetría industrial o una plantilla genérica. Usar lenguaje visual limpio, espacios amplios, alta legibilidad de gráficos, jerarquías claras y énfasis en evidencia experimental. Debe respetar **los tokens de tema existentes** (`viz/theme/tokens.py`) en claro/oscuro; no definir colores hex aislados en cada nueva vista.

- Una acción principal por pantalla o etapa (`Generate`, `Run verification`, `Start training`). Secundarias en estilo discreto (`Save draft`, `Export`, `Copy command`).
- Componentes densos (tabla de señales, matrices, logs) usan contenedores diferenciados y títulos de sección; evitar tarjetas sobre tarjetas innecesarias.
- Estados con icono + texto. Un resultado correcto que no mejora el baseline se clasifica como `Completed · No improvement`, no como fracaso de infraestructura.
- Gráficas Plotly con unidades en ejes, leyenda explícita, explicación textual de métricas y series siempre derivadas de un `Run` existente.
- Los formularios avanzados no abruman a quien importa su primera planta: usar **basic-first / advanced-on-demand** y ayuda contextual en inglés.

### 16.2 Layout recomendado

```text
┌──────────────────────────── TOP / PAGE HEADER ─────────────────────────────────┐
│ Breadcrumb  ·  Title  ·  SourceBadge/Capabilities  ·  Contextual actions         │
├──────────── SIDEBAR ────────┬──────────────────── MAIN CONTENT ────────────────┤
│ Workspace                   │ Primary task, tabs and configuration              │
│ Simulation                  │ ┌───────────────────┬────────────────────────┐   │
│ Design & Control            │ │ Inputs/wizard     │ Results/readiness       │   │
│ Intelligence                │ │ Contextual help   │ Empty until run exists  │   │
│ Operations                  │ └───────────────────┴────────────────────────┘   │
│ Help                        │ [Signals] [Metrics] [Comparison] [Logs] [...]   │
│ PLECS status / Jobs          │ Source, run ID, hash, Copy reproducible command  │
└──────────────────────────────┴───────────────────────────────────────────────┘
```

**Ancho desktop:** panel de configuración 30–40 % y resultados 60–70 % para vistas numéricas. En PLECS wizard, contenido 65–75 % y readiness 25–35 %. En pantallas angostas, config sobre resultados y readiness en acordeón. El sidebar se colapsa; la función queda accesible por título y breadcrumb.

### 16.3 Controles y patrones

| Elemento | Tratamiento |
|---|---|
| Stepper | Etapas numeradas, nombre corto, estado textual, navegación atrás sin perder cambios |
| Source badge | Indica procedencia, **nunca** actividad telemétrica |
| Job status | Estado y última actualización persistidos, progreso indeterminado si no medible |
| Data tables | Ordenables, exportables; nunca silenciosamente promedian errores o casos fallidos |
| Time-series Plotly | Ejes etiquetados, señales/unidades correctas, exportación y resumen accesible |
| Pareto | Click o selección de fila, no solo hover; muestra valores y respuesta realmente calculada |
| Validation checklist | Cada test da resultado, causa, detalle técnico desplegable y acción sugerida |
| Documented limitations | Visibles junto al resultado, no únicamente en Documentation |
| Dangerous changes | Require human approval, record hashes and final decision |

## 17. Inventario global de pantallas

| Área | Pantalla / vista | Tipo | Pantallas/acciones interiores |
|---|---|---|---|
| Workspace | Overview | Mantener + ampliar | Últimos runs, acceso a experimentos, capacidades y documentación |
| Workspace | Experiment Runs | Nueva | Cola, historial, detalle, logs, comparación y manifest |
| Workspace | Environments | Nueva | Catálogo, detalle, checks, snippet Python |
| Simulation | **PLECS Plant Studio** | **Nueva · P0** | My Plants, Create Plant (6 etapas), Compare with SPADE, Connection & Diagnostics; rutas Dataset/Surrogate/Gym |
| Simulation | DC Motor | Mantener + integrar | Plant selector, Generate, runs, comparación |
| Simulation | PMSM / FOC | Mantener + integrar | Controller setups, Generate, métricas y runs |
| Simulation | VSC / DPC | Ampliar | Simulación, control, Parameter Identification PINN |
| Simulation | DC-DC Converters | Nueva | Buck, DAB, modulación, señales y métricas |
| Design & Control | Tuning Studio | Nueva | Setup, bounds, objectives, worker job status, Pareto, compare, approval |
| Design & Control | RL Lab | Nueva | Environment, reward, train, evaluation, approval |
| Intelligence | AI Diagnostics | Ampliar | Known, Uncertain, Unknown/OOD, evidencias |
| Intelligence | Prognostics | Nueva | Dataset NASA, serie, degradación, RUL e incertidumbre |
| Intelligence | Transfer Learning | Mantener | Wizard existente adaptado a runs persistentes |
| Intelligence | Model Registry | Mantener | Modelos, versiones, promoción y evidencia |
| Operations | Monitoring | Mantener | Simulación/archivos como origen declarado |
| Operations | Alert History | Mantener | Histórico y procedencia, sin falsas etiquetas LIVE |
| Help | Documentation | Ampliar | Instalación, glosario, capacidades, API y troubleshooting |
| Help | Algorithm Guide | Nueva | Preguntas guiadas, recomendación explicada y enlace válido |

## 18. Reglas de interacción y persistencia entre pantallas

### 18.1 Contexto que viaja en enlaces

Enlaces como `Open in RL Lab`, `Send to Tuning Studio`, `Compare this run` transportan **IDs versionados**, nunca instancias numéricas en `st.session_state`. Los adapters recuperan y validan el perfil/run; si ya cambió, mostrar `The selected profile is no longer valid. Choose a current version.`

Cada CTA que ejecute código costoso exige confirmación de configuración: `Experiment summary` con perfil, controlador, condiciones, algoritmo, semilla, presupuesto y restricciones. Luego se crea una especificación inmutable; editar el formulario no cambia el job en marcha.

### 18.2 Estado global del usuario

- `Draft`: editable y guardado por identificador; no presume backend disponible.
- `ExperimentSpec`: snapshot versionado de lo que se solicitó ejecutar.
- `Job`: estado persistente y logs, sin depender de la página activa.
- `Run`: artefacto de una ejecución concreta, exitoso o fallido, con trazabilidad.
- `Published Plant`: perfil con capacidades y validación; no equivale a modelo promovido ni a hardware.

Streamlit puede cachear visualizaciones y conservar preferencias transitorias, pero **no es la fuente de verdad** para trabajos ni publicaciones.

### 18.3 Mensajes antes de salir de una vista

- Si cambió un draft sin guardarlo: `You have unsaved changes. Save your draft before leaving?`.
- Si hay job en marcha: navegación permitida, `This job will continue while you navigate SPADE.`.
- Si no existe checkpoint: no mostrar `Resume`; ofrecer `Create new run from saved configuration`.
- Si PLECS desaparece: conservar resultados anteriores y drafts, bloquear nuevos experimentos afectados.
- Si cambia el hash/versión de una FMU: `Validation is stale. Re-run environment verification.`.

### 18.4 Ayuda contextual específica para PLECS

Cada paso aporta una explicación corta, `Learn more` y una acción de corrección. El paso Engine tiene instrucciones separadas para `Local SPADE + local PLECS` y `SPADE hosted elsewhere`; no suponer que localhost del navegador y del backend son equivalentes. La preparación de FMU se ilustra mediante un checklist textual si no existe una imagen real. Los errores técnicos detallados se muestran bajo `Advanced diagnostics`, conservando a la vista la causa legible.

## 19. Comportamiento de GUI para errores y resultados negativos

| Situación | Cómo se ve en la GUI | Qué acción ofrece |
|---|---|---|
| Archivo inválido/corrupto | `The imported file cannot be validated.` + nombre/tipo, sin mostrar canales inventados | `Replace file`, `View diagnostics` |
| RPC responde, pero sin stepping | `RPC available · Interactive stepping unavailable` | `Learn about FMU Co-Simulation`, `Generate dataset instead` |
| Error de unidades | Error en la fila afectada, sin borrar configuración | `Edit mapping` |
| Reset no reproducible | Test `Failed`, bloqueo de Gym publication | `Inspect reset policy`, `Re-run test` |
| Acción no altera la planta | Warning o failed según test/observabilidad, con evidencia de acciones/lecturas | `Inspect action mapping` |
| NaN, solver error, divergence | Resultados/condiciones fallidas visibles y log adjunto | `View failed condition`, `Export report` |
| ETA desconocida | `ETA unavailable`, progreso indeterminado o evaluaciones reales | `Open log` |
| OOD no entrenado/validado | `OOD detection unavailable` | `See supported diagnostics` |
| Surrogate fuera de rango | Aviso de extrapolación y limitación | `Run PLECS comparison` |
| Candidato empeora baseline | `No improvement` con métricas por condición | `Inspect differences`, no promoción implícita |
| Intervalo RUL sin cobertura validada | `Estimated interval · Coverage not validated` | `Open evaluation details` |
| Cancelación solicitada | `Cancellation requested` hasta que el worker confirme | `View current phase` |

## 20. Criterios de aceptación adicionales por historia de usuario

### PLECS: creación de Gym

- [ ] Desde SPADE, la persona entiende la diferencia entre `Gym`, `Dataset` y `Surrogate` sin abrir documentación externa.
- [ ] Puede comenzar sin PLECS conectado, guardar un borrador y reabrirlo después.
- [ ] La UI distingue `.plecs` (sweep/script) y `.fmu` (interfaz candidata a co-simulación) correctamente.
- [ ] Detecta variables reales y limita auto-mapping a propuestas verificables; no inventa puertos ni un esquema eléctrico.
- [ ] El panel readiness identifica requisitos restantes y navega al campo correspondiente.
- [ ] Los contratos acción/observación, dimensiones, tiempos, reset, reward, safety y terminación son auditables.
- [ ] Solo un adaptador capaz de aplicar una acción **y avanzar el mismo estado de la planta** habilita publicación Gym interactiva.
- [ ] El usuario obtiene un reporte persistente de todos los tests incluso si fracasan.
- [ ] Un Gym publicado se muestra en Environments/RL Lab como entorno **simulado**, con versión y hash.

### PLECS: dataset/surrogate/comparación

- [ ] Barridos se pueden ejecutar de forma persistente con métricas y fallas por condición.
- [ ] Un dataset producido por RPC nunca aparece como planta Gym por error.
- [ ] Ajuste de surrogate usa partición declarada, evalúa frente a condiciones separadas y muestra error por condición.
- [ ] Cada curva de SPADE vs PLECS proviene de un run real, con alineación temporal y unidades registradas.
- [ ] Comparaciones con condiciones incompatibles no afirman mejora ni equivalencia.

### Plataforma completa

- [ ] Navegación nueva convive con páginas existentes sin duplicar módulos de simulación.
- [ ] Jobs sobreviven cambios de página y refresh; cancelación/reanudación tienen semántica real.
- [ ] Run Comparison reutilizado en Tuning, RL, plantas nativas y PLECS.
- [ ] Toda predicción IA indica `AI SUGGESTION — REQUIRES VERIFICATION` cuando aplica, con versión y confianza calibrada si existe.
- [ ] No hay uso engañoso de `LIVE`, `production ready`, `hardware verified`.
- [ ] Light/dark, teclado, accesibilidad y responsive probados al menos en desktop, laptop, tablet y móvil.
- [ ] Cambios de defaults y promoción de modelos pasan por aprobación humana persistida.

## 21. Instrucciones de implementación para agente de programación

> Objetivo de esta sección: convertir la especificación en tareas verificables para Codex/Claude o un equipo, **sin suponer que todas las clases y conectores ya existen**.

1. **Inspeccionar la versión actual del repositorio** antes de editar archivos. Identificar rutas y firmas reales en `src/driveflow/viz/` y evitar reemplazos masivos de vistas existentes.
2. Elaborar un **inventario de componentes reutilizables** (`page_header`, `promotion_gate`, estilos, plots, adapters). Reusar patrones y no crear un segundo sistema visual.
3. Definir y probar entidades `PlantProfile`, `EnvironmentCapability`, `ExperimentSpec`, `Job`, `Run`, `Provenance` y errores tipados. Implementar esquema/versionado/persistencia antes de formularios complejos.
4. Crear `JobService` en worker separado, con cola durable, estado consultable, cancelación honesta y export de logs/manifest. No almacenar jobs en `st.session_state`.
5. Crear **PLECS Plant Studio** con tabs y wizard de seis etapas, primero usando capacidades reales o estados `Not supported`; no fingir conexión o simulación. Todas las etiquetas de la aplicación son inglesas.
6. Implementar primero RPC batch y su comparación; el acoplamiento Gym por FMU requiere spike técnico verificable. Si falla, **mantener publicado solo Batch/Dataset** sin presentar Gym ready.
7. Implementar mapeo I/O, editor de reward y Safety como componentes reutilizables; probar validación de unidades, acción, bounds, causalidad, sincronización y errores.
8. Crear validación de entorno (Gym checker + stepping/reset físico simulado + fallas adversas). **Solo** publicar perfiles con prueba vigente para sus hashes/versiones.
9. Añadir Environments y acceso desde RL Lab; Tuning/RL se conectan a entornos según `capabilities`, no por nombre de planta.
10. Añadir Run Comparison, DC-DC, OOD, PINN, RUL, algoritmo guía, pruebas responsive/accesibilidad y documentación de instalación en fases posteriores.
11. Ejecutar tests y reportar exactamente qué se probó, qué falta y qué requiere licencia/PLECS local. No generar scores ni series ficticias para hacer bonita la pantalla.
12. Crear un PR o conjunto de commits por fase; pruebas mínimas por feature y capturas para cada estado (`empty`, `loading`, `error`, `success`, `negative`).

### Definición de terminado (DoD) para una pantalla

Una pantalla se considera completada cuando cumple: navegación y deep-link; formulario con defaults explicitados; validación inline; estado vacío; estados de carga y error; resultado calculado o `Not available`; source badge y provenance; accesibilidad/teclado; light/dark; sin ejecuciones involuntarias; logs/test reproducible y pruebas automatizadas de estados principales.

## 22. Lo que esta especificación no autoriza afirmar

- La existencia de una integración PLECS funcional, exportación FMU automatizada, runtime FMU licenciado o stepping ya compatible con el repositorio.
- Que un archivo `.plecs` subido desde el browser pueda necesariamente ejecutarse en un servicio remoto, ni que PLECS instalado en la computadora cliente esté disponible en el servidor.
- Que una llamada RPC por step equivalga a co-simulación verdadera.
- Que exista un modelo surrogate confiable sin partición de validación ni un estimador de error.
- Que una clasificación/estimación de IA sea diagnóstico clínico/industrial confirmado o que se haya validado sobre hardware real.
- Que todos los algoritmos de la guía estén implementados; los no disponibles se mostrarán como tales.
- Que `Resume` o `ETA` existan para todos los jobs, ni que un checkbox en frontend sustituya el enforcement del backend.

### Referencias de integración técnica

- [PLECS — Scripting and RPC](https://docs.plexim.com/plecs/latest/scripting/)
- [PLECS — Functional Mockup Interface](https://docs.plexim.com/plecs/latest/functional-mockup-interface/)
- [Gymnasium — Env API](https://gymnasium.farama.org/api/env/)
- Repositorio SPADE proporcionado (`SPADE-main (2).zip`).

---

**Documento maestro.** Para implementación, conservar este archivo como especificación de referencia y registrar cambios de alcance/versiones en un changelog separado.
