# SPADE · Agente Asistente "PhD. Radrigan": especificación de frontend e integración

Versión 1.0 · 2026-11-02 · Documento de planificación para Luciano (en español). El código, los identificadores y los textos de la interfaz van en inglés.

**Se apoya en:**
- `ESPEC_AGENTE_ASISTENTE_2026-11-02.md`: qué hace el agente, capacidades C1–C17 y reglas;
- `PLAN_AGENTE_ASISTENTE_2026-11-02.md`: arquitectura;
- `INSTRUCCIONES_CLAUDE_CODE_FASE1_2026-11-02.md`: decisiones B1–B7 y la Fase 1, en curso.

Este documento **no cambia** esas decisiones. Lo que las tocaría queda en §9 como decisión pendiente.

**Verificado en el código y en el archivo** (2026-11-02):
- **Streamlit:** SPADE usa la versión 1.66, que trae `st.components.v2.component(name, html=, css=, js=, isolate_styles=)` (componentes con HTML, CSS y JS inline, sin paso de build), `st.fragment(run_every=…)`, `st.chat_message(avatar=<imagen>)` y `st.chat_input(accept_file=True)`.
- **Recursos visuales existentes:** `src/driveflow/viz/assets/` (hoy solo `motor_base.webp`) y el tema en `src/driveflow/viz/theme/` (`tokens.py`, `css.py`).
- **La imagen del avatar** es un **JPG RGB de 1024 × 1028 px sobre fondo negro** (el 68 % de los píxeles es casi negro). **No tiene transparencia** (ver §1.1).

---

## 0. Ajustes a la propuesta, con su motivo

| Propuesta original | Ajuste | Motivo |
|---|---|---|
| Visión con GPT-4o o Gemini Pro Vision | **Modelo con visión local en Ollama**, elegido con evidencia como en F1.1 | Decisión **B1** (2026-11-02): ningún dato sale de la máquina. Los datasheets y las fotos de planta pueden ser confidenciales. Usar la nube requiere revertir B1 (§9, decisión F1) |
| Componentes React o Vue | **Componentes v2 de Streamlit** con HTML/CSS/JS sin framework | Sin toolchain de Node, sin paso de build, sin dependencias de npm y sin cambios en el instalador de Windows. La GUI de SPADE es Streamlit |
| Funciones `configurar_motor`, `extraer_parametros`, `ejecutar_simulacion` | Se usan las herramientas ya definidas (§4), con nombres en inglés | Regla del repo (los identificadores van en inglés) y para no duplicar herramientas |
| Avatar "completamente aislado de cualquier fondo" | Hay que **producir** la versión con transparencia y separada en capas (§1.1) | El archivo actual es JPG sobre negro |
| "Confirmación: guiño o sonrisa de aprobación" | El guiño se dispara **solo cuando el usuario confirma**, nunca al mostrar resultados | Una animación de "aprobación" sobre un resultado sugiere que está validado, y eso choca con las reglas de evidencia (§1.4) |

---

## 1. Diseño del avatar

### 1.1 Recursos que hay que producir (desde la imagen actual)

La imagen sirve como referencia de estilo, pero **no se puede animar tal como está**: es un render plano en JPG, con el contorno cian y el texto de neón fundidos sobre el negro. Se necesita:

| # | Recurso | Formato | Para qué |
|---|---|---|---|
| A1 | **Figura recortada con canal alfa**, sin el texto, conservando el brillo cian del contorno como semitransparencia | PNG maestro de 2048 px + WebP con alfa a 512 y 1024 px | Base de todo; funciona en tema claro y oscuro |
| A2 | **El texto "PhD. Radrigan" como texto real**, no como imagen | CSS (neón con `text-shadow`) y una fuente con licencia OFL incluida en el repo | Escala, cambia con el tema, se puede animar y se lee con lector de pantalla |
| A3 | **Recorte de cabeza** para las burbujas del chat | WebP con alfa, 96 y 192 px | `st.chat_message(avatar=…)` |
| A4 | **Variantes de pose** del mismo personaje, con la misma cámara y la misma luz: (a) neutra mirando al chat; (b) mano en la barbilla; (c) sosteniendo un papel; (d) guiño | Igual que A1 | Estados que CSS no puede hacer sobre un raster (mover la mano, cerrar un ojo) |
| A5 | **Capas superpuestas** en SVG: brillo del microchip del cuello, línea de escaneo, partículas o líneas de datos | SVG | Animación por CSS sobre A1 y A4 |
| A6 | **Variante para tema claro** del contorno: cian más oscuro o contorno doble | Igual que A1 | El cian sobre blanco pierde contraste |
| A7 | **Archivo fuente con capas** (PSD, SVG o el proyecto del render) | Fuente | Para un rig en Rive en la etapa UI-4 |

**Recortar A1 desde el JPG actual es posible pero difícil:** el brillo cian es semitransparente sobre negro, y un recorte por luminancia lo deja sucio. Conviene **pedir la exportación con transparencia** a quien generó la imagen, o **regenerarla** con fondo transparente y la misma referencia.

**Consistencia de A4:** si las poses se generan por separado, hay que aceptar que entre ellas haya pequeñas diferencias de cara y ropa. Para evitarlas está la opción A7 + Rive (§1.3).

**Presupuesto de peso:** menos de 250 KB por estado en WebP, y menos de 1 MB en total para el MVP.

**Ubicación:** `src/driveflow/viz/assets/assistant/`.

**Licencia del avatar.** SPADE es MIT, pero el avatar es la imagen de Luciano. Propuesta: el avatar queda **excluido de la licencia MIT**, con su licencia propia en `licenses/avatar/` (por ejemplo, "all rights reserved, redistribution only as part of SPADE"). Además, se puede reemplazar por configuración (`configs/assistant.yaml: persona.avatar_dir`), para que un fork use otro sin tocar el código. **Decisión F3.**

### 1.1b Prototipo con la imagen actual (2026-11-02)

Se construyó un prototipo interactivo (artifact "PhD. Radrigan Assistant") que demuestra que el MVP **se puede hacer ya con la imagen actual**, sin esperar a A1–A7:
- **Transparencia real a partir del JPG:**
  - el fondo negro conectado a los bordes pasa a alfa 0 (relleno por inundación, `scipy.ndimage.label`);
  - el halo cian cercano al borde (≤ 14 px) pasa a alfa parcial, proporcional a su brillo, con despremultiplicado;
  - el pelo y la barba oscuros quedan opacos, porque no tocan el fondo.

  Funciona sobre fondos oscuros. Para el tema claro sigue haciendo falta A6.
- **El texto horneado se recorta** (la imagen termina en la camisa, con un desvanecido inferior) y "PhD. Radrigan" se dibuja como texto neón en CSS (A2).
- **La mano como capa propia:** se separa por color (todo lo que no es la camisa morada dentro de la zona de la mano) y gira desde la muñeca, que queda cubierta por la manga. **Saludo real** al empezar cada respuesta.
- **Los dos momentos clave** que pidió Luciano:
  - **cuando alguien pregunta:** `listening` mientras escribe (inclinación hacia el chat) y `thinking` / `using_tool` después de enviar (puntos de pensamiento, chip del cuello que pulsa, anillo de barrido);
  - **cuando el agente responde:** `speaking`, con saludo al empezar, un pequeño latido de la figura **por cada palabra que aparece**, brillo del contorno y barras de voz en el indicador de estado.
- **Peso:** ≈ 35 KB la capa base y ≈ 11 KB la mano, en WebP con alfa, a 560 px.
- **Limitaciones del recorte automático:** quedan restos mínimos de contorno en el corte de la mano, y no hay parpadeo ni boca animada (requieren capas de origen, A7). Para producción se recomienda que estas capas salgan de la fuente original, pero el método sirve como MVP.

### 1.2 Disposición: el avatar "detrás" del chat, no un ícono

```
┌────────────────────────────── page "Assistant" (wide) ──────────────────────────────┐
│ ┌──────────── col A (≈28 %) ─────────────┐ ┌──── col B (≈44 %) ────┐ ┌─ col C (≈28 %) ─┐ │
│ │  status chip: "Thinking…"              │ │ chat messages          │ │ DRAFT            │ │
│ │                                        │ │  ◖ assistant bubble    │ │ plant · preset   │ │
│ │        ╭───────╮                       │ │    ◀── tail points     │ │ experiments      │ │
│ │       (  A1/A4 )   ← figure, sticky,   │ │        to the figure   │ │ cost · [Run]     │ │
│ │        ╰───┬───╯     bottom-anchored,  │ │  ▶ user bubble         │ │                  │ │
│ │     ┌──────┴──────┐  overlaps col B    │ │                        │ │ (review card when│ │
│ │     │  shoulders  │  edge by ~8 %      │ │                        │ │  a datasheet is  │ │
│ │     └─────────────┘                    │ │ [chat_input + attach]  │ │  being extracted)│ │
│ │   PhD. Radrigan  (A2, neon CSS text)   │ │                        │ │                  │ │
│ └────────────────────────────────────────┘ └────────────────────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Figura fija** (`position: sticky; bottom: 0`) en la columna A, anclada abajo y de mayor tamaño que un ícono (alto ≈ 60–70 % del viewport). Invade unos píxeles la columna del chat, para que parezca que asoma detrás de las burbujas.
- **La burbuja del asistente** tiene una "cola" dirigida hacia la figura, y el último mensaje se alinea a la altura de su cabeza.
- **Chip de estado** sobre la figura, con texto: "Thinking…", "Reading datasheet…", "Waiting for your confirmation", "Simulation running: 3/5". **El estado nunca se comunica solo con animación.**
- **Pantallas angostas** (menos de 900 px): la figura pasa a un medio cuerpo de unos 120 px en la cabecera del chat, y el borrador queda como pestaña bajo el chat.
- **Spike requerido (UI-1):** comprobar que un componente v2 con `isolate_styles=True` puede solaparse visualmente con la columna vecina. Si el aislamiento de estilos lo impide, la alternativa es dibujar la figura y el contenedor del historial **dentro del mismo componente**, dejando nativos solo `chat_input` y el panel del borrador.

### 1.3 Estados y micro-interacciones

**Técnica por etapa:**
- **MVP (UI-2):** las poses A4 se intercambian con un fundido de 150–250 ms, y las capas A5 se animan con CSS. Solo se animan `transform` y `opacity`.
- **Etapa UI-4, opcional:** un rig en **Rive** a partir de A7. Requiere agregar el runtime de Rive como archivo JS incluido en el repo (MIT), sin CDN, porque SPADE funciona sin internet. Es una dependencia de frontend: **decisión F4.**
- **Lottie no se recomienda** para un personaje con look de render 3D: funciona bien en vector plano, no en este estilo.

| Estado | Lo dispara | Visual (MVP) | Texto del chip | Duración |
|---|---|---|---|---|
| `idle` | Sin actividad | Pose (a). Respiración: `translateY` de ±2 px, ciclo de 6 s. **Parpadeo** con una capa SVG de párpados sobre A1, cada 4–7 s con jitter; si no se puede alinear bien, sin parpadeo | — | Continuo |
| `listening` | El usuario escribe (en el componente propio es detectable; con `st.chat_input` nativo no hay evento de tecleo, así que el estado se omite o el input va dentro del componente) | Pose (a) con leve inclinación hacia el chat (`rotate` de 1–2°) | — | Mientras escribe |
| `thinking` | Llamada al LLM en curso | Pose (b), **mano en la barbilla**. **El microchip pulsa** (A5, opacidad 0,4 → 1, ciclo de 1,2 s) | "Thinking…" | Mínimo 400 ms, para evitar parpadeos |
| `using_tool` | Ejecución de una herramienta | Pose (b), con una línea de datos cian que recorre el borde de la figura | Nombre legible de la herramienta: "Checking the plant…", "Estimating cost…" | Lo que dure |
| `extracting` | `read_datasheet` (Fase 2) | Pose (c), **sosteniendo un papel**. Línea de escaneo cian (A5) que baja por el papel y **líneas de datos que salen de las gafas** hacia la columna del borrador | "Reading datasheet (page 2/4)…" | Lo que dure |
| `awaiting_confirmation` | Hay una propuesta validada | Pose (a) mirando hacia el botón **Run** (o una flecha suave hacia él) | "Waiting for your confirmation" | Hasta que el usuario responde |
| `confirmed` | **El usuario** confirmó y el trabajo se encoló | Pose (d), **guiño**, una sola vez | "Launched" | 900 ms y vuelve |
| `running` | Trabajo en el worker | Pose (a), con una barra de progreso de neón bajo el chip (`st.fragment(run_every="2s")`, que consulta `get_job_status`) | "Simulation running: 3/5" | Lo que dure |
| `results_ready` | `summarize_results` terminó | Pose (a), gesto neutro (sin sonrisa de aprobación) | "Results ready (simulated)" | 2 s y vuelve a idle |
| `error` | Error de herramienta o de validación | Pose (b), chip en ámbar | El error en una línea | Hasta el siguiente turno |
| `unavailable` | Ollama no responde o falta el modelo | Figura en gris al 50 %, sin animación | "Assistant unavailable: local model not running" + cómo arrancarlo | Mientras dure |

**Accesibilidad:**
- con `prefers-reduced-motion: reduce`, se desactivan respiración, parpadeo, líneas y pulsos, y solo cambia la pose;
- `alt="PhD. Radrigan, SPADE assistant (AI)"`;
- el chip va en `aria-live="polite"`;
- contraste del texto del chip ≥ 4,5:1 en ambos temas;
- las animaciones se pausan con `document.visibilitychange` cuando la pestaña está oculta.

### 1.4 Reglas de identidad (persona)

- **Siempre identificado como IA.** Junto al nombre va la etiqueta "AI assistant · local model" (y el modelo y su digest en un tooltip). Primer mensaje fijo: "I'm SPADE's AI assistant. I set up and run **simulations**; my suggestions require your verification."
- **No habla como Luciano.** El nombre es la mascota del asistente. El modelo no escribe en primera persona como si fuera el investigador ("en mi experiencia…", "yo recomiendo como PhD…"). Esto va en el *system prompt* y en el benchmark de F1.11 como caso de prueba.
- **La animación nunca contradice la evidencia.** Ni guiño ni sonrisa sobre resultados; el éxito de una simulación no es una validación. Las etiquetas `AI SUGGESTION — REQUIRES VERIFICATION` y `SIMULATED` siguen visibles aunque el avatar ocupe espacio.

---

## 2. System design

```
 USER (browser)
   │ text / file (st.chat_input accept_file)       click [Run] / "yes"
   ▼                                                    │
 ┌─ Streamlit page: viz/pages/assistant.py ────────────────┼───────────────────────────────┐
 │  views/assistant/chat.py      views/assistant/draft_panel.py   views/assistant/review.py│
 │  components/radrigan_avatar.py  (st.components.v2.component: html+css+js, assets/)     │
 │        ▲ props: {state, label, progress, theme, reduced_motion}                        │
 │        │                                                                               │
 │  adapters/assistant_adapter.py  ── subscribes to AgentEvents, maps → AvatarState ──────┘
 └────────┬───────────────────────────────────────────────────────────────────────────────
          │ AgentEvent stream (in-process callback; persisted in assistant_messages)
          ▼
 ┌─ driveflow.assistant (Phase 1, in progress) ──────────────────────────────────────────┐
 │  core/loop.py ── OllamaClient (localhost) ── model (tools) / vision model (Phase 2)   │
 │  core/policy.py (tools visible per state; run_* need UI confirmation flag + draft hash)│
 │  draft.py (provenance)   numbers.py (no-invention)   tools.py                        │
 └────────┬──────────────────────────────────────────────────────────────────────────────┘
          ▼ existing SPADE code
   plant_catalog.describe_plant · Scenario · ExperimentSpec · JobService.submit
   jobs.handlers.scenario_sweep / controller_tuning · RunStore (SQLite + parquet)
   validation.step_metrics · runs.compare
```

**Contrato de eventos** (nuevo, `assistant/events.py`): el bucle emite eventos y la UI solo los traduce. Así la UI no depende del modelo, y los tests de UI usan eventos grabados.

```json
{"type": "turn_started"}
{"type": "llm_call", "phase": "start" | "end", "tokens": 812}
{"type": "tool_call", "phase": "start" | "end", "tool": "estimate_cost", "ok": true}
{"type": "extraction", "phase": "start" | "progress" | "end", "page": 2, "pages": 4}
{"type": "proposal_ready", "draft_hash": "…", "n_experiments": 5, "est_wall_s": 40}
{"type": "user_confirmed", "draft_hash": "…", "message_seq": 7}
{"type": "job", "job_id": "…", "state": "running", "done": 3, "total": 5}
{"type": "results_ready", "run_id": "…"}
{"type": "error", "where": "tool:validate_draft", "message": "…"}
{"type": "runtime_unavailable", "reason": "ollama not reachable at 127.0.0.1:11434"}
```

**Correspondencia evento → estado del avatar** (`adapters/assistant_adapter.py`, función pura y con test):
- `llm_call:start` → `thinking`;
- `tool_call:start` → `using_tool`;
- `extraction:*` → `extracting`;
- `proposal_ready` → `awaiting_confirmation`;
- `user_confirmed` → `confirmed`, luego `running`;
- `job` → `running`;
- `results_ready` → `results_ready`;
- `error` → `error`;
- `runtime_unavailable` → `unavailable`;
- `turn_ended` → `idle`.

Si llegan varios eventos a la vez, el orden de prioridad es: `unavailable` > `error` > `confirmed` > `extracting` > `using_tool` > `thinking` > `running` > `idle`.

**Modelo de ejecución de Streamlit:** una llamada al LLM bloquea el script. Por eso:
1. se pinta el avatar en `thinking` (con `st.empty()` y luego `.container()`) **antes** de la llamada;
2. el bucle llama a un callback por cada evento, y el callback actualiza ese contenedor;
3. el progreso del trabajo se consulta con `st.fragment(run_every="2s")`, sin bloquear la página.

---

## 3. Componentes de frontend

Todo vive en `src/driveflow/viz/`, respetando las capas que ya usa SPADE: view → adapter → dominio.

| Componente | Archivo | Tipo | Props / responsabilidad |
|---|---|---|---|
| `radrigan_avatar` | `components/radrigan_avatar.py` + `components/radrigan_avatar/{avatar.html,avatar.css,avatar.js}` | Componente v2 | `state`, `label`, `progress (0–1)`, `theme`, `reduced_motion`, `compact`. Solo presentación: no conoce al agente |
| `chat_view` | `views/assistant/chat.py` | Nativo (`st.chat_message`, `st.chat_input(accept_file=…)`) | Historial con el avatar A3 en las burbujas; adjuntos solo desde la Fase 2 |
| `draft_panel` | `views/assistant/draft_panel.py` | Nativo | Planta, preset con chips de origen, experimentos, costo y botón **Run** (que pone la marca de confirmación) |
| `extraction_review` | `views/assistant/review.py` | Nativo + `st.data_editor` | Revisión de lo extraído (§5). Fase 2 |
| `proposal_card` | `views/assistant/proposal.py` | Nativo | Tabla de experimentos expandidos, supuestos, costo, y botones **Run** / **Edit** |
| `job_progress` | `views/assistant/progress.py` | `st.fragment(run_every=…)` | Progreso, condiciones ok/fallidas y enlace a *Experiment Runs* |
| `results_summary` | `views/assistant/results.py` | Nativo + Plotly (ya en el extra `viz`) | KPIs de `summarize_results`, gráficos desde los parquet y etiqueta SIMULATED |
| `persona_header` | `components/persona_header.py` | Nativo + CSS del tema | Nombre en neón (A2), etiqueta "AI assistant · local model" y tooltip con el modelo |
| `assistant_adapter` | `adapters/assistant_adapter.py` | Python puro | Eventos → estado del avatar; sesión ↔ `st.session_state` |

**Esbozo del componente del avatar:**

```python
# src/driveflow/viz/components/radrigan_avatar.py
from pathlib import Path
import streamlit as st

_DIR = Path(__file__).parent / "radrigan_avatar"
_avatar = st.components.v2.component(
    "spade_radrigan_avatar",
    html=(_DIR / "avatar.html").read_text(encoding="utf-8"),
    css=(_DIR / "avatar.css").read_text(encoding="utf-8"),
    js=(_DIR / "avatar.js").read_text(encoding="utf-8"),
)

STATES = ("idle", "listening", "thinking", "using_tool", "extracting", "awaiting_confirmation",
          "confirmed", "running", "results_ready", "error", "unavailable")

def radrigan_avatar(state: str, label: str = "", progress: float | None = None, *, compact: bool = False, key: str = "avatar"):
    if state not in STATES:
        raise ValueError(f"unknown avatar state {state!r}")
    return _avatar(data={"state": state, "label": label, "progress": progress, "compact": compact,
                         "assets": asset_urls()}, key=key)
```

La forma exacta de pasar datos y recursos a un componente v2 (`data=…`, URLs de recursos estáticos) se confirma en el spike UI-1 contra la API de 1.66. Arriba está la intención, no la firma verificada.

```css
/* avatar.css (extract) */
.figure { position: sticky; bottom: 0; transition: opacity .2s; }
.pose { position: absolute; inset: 0; opacity: 0; transition: opacity .2s ease; }
.pose.active { opacity: 1; }
.state-idle .figure { animation: breathe 6s ease-in-out infinite; }
.state-thinking .chip-glow { animation: pulse 1.2s ease-in-out infinite; }
.state-extracting .scanline { animation: scan 1.6s linear infinite; }
@keyframes breathe { 50% { transform: translateY(-2px); } }
@keyframes pulse { 0%,100% { opacity: .4; } 50% { opacity: 1; } }
@keyframes scan { from { transform: translateY(0); } to { transform: translateY(100%); } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
```

---

## 4. Herramientas (lo que el agente puede "llamar")

Son las herramientas ya definidas en el plan y en la Fase 1. Los nombres de la propuesta original quedan mapeados así:

| Nombre propuesto | Herramienta real | Implementa sobre | Efecto | Fase |
|---|---|---|---|---|
| `extraer_parametros` | `read_datasheet(file_id, pages?)` | Modelo con visión local (Ollama), esquema forzado con convenciones y variantes (`RELEVAMIENTO_DATASHEETS` §5) | Lectura | 2 |
| — | `derive_parameters(plant, nameplate)` | `assistant/physics/nameplate.py` + `consistency.py` | Lectura | 2 |
| `configurar_motor` | `update_draft(patch)` + `set_plant(plant_config_id)` | `assistant/draft.py`; valida con `describe_plant` y `Scenario` | Borrador (**no** cambia nada en SPADE) | 1 |
| — | `recommend_plant`, `describe_plant`, `suggest_value` | `datagen/plant_catalog.py`, `configs/plant_catalog.yaml` | Lectura | 1 |
| — | `validate_draft`, `estimate_cost` | `Scenario(**kw)`, `_validate_scenario_sweep`, `configs/sim_cost.yaml` | Lectura | 1 |
| `ejecutar_simulacion` | `run_experiments()` | `ExperimentSpec("scenario_sweep")` → `RunStore.save_experiment` → `JobService.submit` | **Ejecuta**; requiere la confirmación del usuario y el hash del borrador | 1 |
| — | `describe_tunable`, `update_objective`, `run_tuning`, `check_requirements` | `optim/`, `controller_tuning` | `run_tuning` ejecuta (con confirmación) | 1 |
| — | `get_job_status`, `summarize_results` | `JobService`, `validation.step_metrics`, `runs.compare` | Lectura | 1 |
| — | `explain_spade`, `search_docs` | `configs/assistant_catalog.yaml`, `docs/` | Lectura | 1 |
| "guardar paquete de experimento" | No es una herramienta: lo hace `run_experiments` | `experiments` + `runs` + `workspace/runs/<id>/condition_NNN.parquet` + `assistant_launches.draft_snapshot` (con procedencia) | — | 1 |
| — | `export_report(session_id)` | Reporte reproducible (C11) | Escribe un archivo | 3 |

Todas son Python. **No hay herramientas en C++**: SPADE no expone ninguna, y no hace falta.

---

## 5. Revisión de lo extraído antes de usarlo (Fase 2)

La tarjeta `extraction_review` aparece en la columna C cuando termina `read_datasheet` + `derive_parameters`. **Nada pasa al borrador sin pasar por aquí.**

```
┌ Extracted from: PMG-132-e.pdf · AI SUGGESTION — REQUIRES VERIFICATION ───────────────┐
│ Motor type: DC permanent magnet  (evidence: "PMDC", "60 V DC")          [change ▾]    │
│ Variant:   ( ) 24 V  ( ) 48 V  (•) 60 V  ( ) 72 V        ← only when several columns  │
├──────────────────────┬──────────┬──────┬──────────────┬────────────┬──────────────────┤
│ Parameter            │ Value    │ Unit │ Origin       │ Confidence │ Where            │
├──────────────────────┼──────────┼──────┼──────────────┼────────────┼──────────────────┤
│ Rated voltage        │ 60       │ V    │ ● read       │ high       │ p1 table [crop]  │
│ Rated current        │ 110  ✎   │ A    │ ● read       │ ⚠ medium   │ p1 table [crop]  │
│ Armature resistance  │ 0.016    │ Ω    │ ● read       │ high       │ p1 [crop]        │
│ Flux constant ψe     │ 0.181    │ V·s  │ ◆ derived    │ —          │ T_n / I_n  (ⓘ)   │
│ Rotor inertia        │ ⛔ missing│ kg·m²│ —            │ —          │ [enter] [suggest]│
├──────────────────────┴──────────┴──────┴──────────────┴────────────┴──────────────────┤
│ Consistency checks:  ✓ P = T·ω (±2 %)   ⚠ Kt vs Ke differ 12 % → [use Kt] [use Ke]     │
│ Required for "DC speed control": 1 missing (rotor inertia)                            │
│                                 [Use these parameters]  (disabled until resolved)     │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

- **Columna "Where":** página y fila, más una **miniatura del recorte** si el modelo devuelve una región confiable. Si no la devuelve, solo la página. No se inventan coordenadas.
- **Colores de origen:** leído, calculado, sugerido, default y supuesto, con los mismos colores que el panel del borrador. **Hay leyenda; el color nunca es la única señal.**
- **Edición en línea con `st.data_editor`:** un valor editado pasa a origen `user` y queda registrado (antes → después).
- **Variantes** (estilo A del relevamiento): se elige antes de ver la tabla. Si el usuario dio el código de artículo, la variante se preselecciona y se dice por qué.
- **Conflictos** (Kt frente a Ke, etiqueta sospechosa como "Thermal resistance (Ra)"): se resuelven con botones explícitos, nunca en silencio.
- **"Use these parameters"** se habilita cuando no faltan valores **requeridos** para el objetivo elegido (matriz del relevamiento §4). Solo **llena el borrador**: lanzar sigue requiriendo la propuesta y **Run**.
- **Mientras tanto, el avatar** muestra `extracting` y después `awaiting_confirmation` (mirando la tarjeta).

---

## 6. Datos y almacenamiento

No hay nada nuevo frente al plan §8 y F1.8: `assistant_sessions`, `assistant_messages` (incluye los eventos de §2) y `assistant_launches`. Los adjuntos van a `workspace/assistant/<session_id>/` con su SHA-256.

**El estado del avatar no se guarda:** se deriva de los eventos.

---

## 7. Hoja de ruta del frontend

El orden está **alineado con las fases ya acordadas**: la Fase 1 (chat sin adjuntos) está en curso con Claude Code, y la extracción real de datasheets es la Fase 2. La interfaz de extracción se puede **diseñar y probar ahora con datos grabados**, sin esperar al modelo con visión.

| Etapa | Contenido | Depende de | Se puede empezar |
|---|---|---|---|
| **UI-0. Recursos** | A1–A6 (A7 si se elige Rive), licencia del avatar en `licenses/avatar/`, guía de uso, revisión en tema claro y oscuro | Decisiones F2 y F3 | **Ya** (es trabajo de diseño, no de código) |
| **UI-1. Spike** | Componente v2 con los 11 estados, alimentado por eventos **simulados**; verificar el solapamiento con la columna vecina, el tema, `reduced-motion`, la pausa con la pestaña oculta y el peso de los recursos; página de demo oculta | UI-0 (con los recursos actuales sirve un borrador) | **Ya**, en paralelo a la Fase 1 |
| **UI-2. Página Assistant con avatar** | Es la tarea F1.10 de la Fase 1, ampliada con `radrigan_avatar`, el contrato de eventos (§2), `persona_header` y `proposal_card` | F1.9 (CLI) + UI-1 | Cuando la Fase 1 llegue a F1.10 |
| **UI-3. Revisión de extracción** | `extraction_review` (§5), con **datos grabados** de los datasheets del relevamiento y del generador sintético; adjuntos en `chat_input` | UI-2 | El diseño y los tests con datos grabados, antes de la Fase 2; la conexión real, en la Fase 2 |
| **UI-4. Pulido** | Rig de Rive desde A7 (si F4 = sí), animaciones finas (parpadeo real, líneas de datos que salen de las gafas), `results_summary` completo y reporte (C11) | Fase 2–3 | Después |

**Criterios de aceptación de la interfaz** (para UI-2 y UI-3):
- todos los estados se pueden alcanzar desde eventos grabados (`AppTest`);
- la página no se rompe sin Ollama (estado `unavailable`);
- con `reduced-motion` no hay animación;
- contraste del chip ≥ 4,5:1 en los dos temas;
- recursos del MVP < 1 MB;
- no se modifica ninguna página existente;
- el smoke `--launch` sigue en verde;
- **ninguna animación de aprobación sobre resultados** (test que recorre los eventos `results_ready`).

---

## 8. Instrucciones para Claude Code (cuando llegue a F1.10)

**No hay que cambiar ahora las instrucciones de la Fase 1**, que están en ejecución. Cuando Claude Code llegue a F1.10, este documento se le pasa como adenda:
- F1.10 incluye UI-1 y UI-2;
- el contrato de eventos de §2 se implementa en `assistant/events.py`;
- `core/loop.py` emite esos eventos mediante un callback opcional. **Si ya está escrito sin eventos, se agregan sin cambiar su comportamiento**;
- los recursos se usan desde `src/driveflow/viz/assets/assistant/`. Si todavía no están A1–A6, se usa un **marcador de posición** (silueta neutra) y la página igual funciona. **No se commitea el JPG actual como recurso final.**

---

## 9. Decisiones pendientes

| # | Decisión | Recomendación |
|---|---|---|
| **F1** | Visión: ¿local (B1) o se permite la nube (GPT-4o, Gemini) para datasheets? | **Local.** La nube, solo como opción explícita, desactivada por defecto y con aviso. Implicaría revertir B1 en parte |
| **F2** | ¿Quién produce A1–A7 (recorte con alfa, poses, capas, variante clara)? ¿Exportar desde la fuente o regenerar? | Pedir la **exportación con transparencia** desde la fuente original. Si no existe, regenerar con fondo transparente usando la imagen actual como referencia |
| **F3** | Licencia del avatar (excluido de MIT) y posibilidad de reemplazarlo por configuración | Sí a ambas |
| **F4** | ¿Rig en Rive (runtime JS incluido, MIT) en UI-4, o solo poses + CSS? | Poses + CSS para el MVP; Rive se decide después de ver UI-2 |
| **F5** | Nombre visible: "PhD. Radrigan" con la etiqueta "AI assistant · local model" | Sí. Mantener siempre visible la etiqueta de IA |
