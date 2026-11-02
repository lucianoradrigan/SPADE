# SPADE: instrucciones para Claude Code. Ronda "Agente Asistente", Fase 1 (chat sin adjuntos, LLM local)

> **Versión final, aprobada por Luciano el 2026-11-02.** Claude Code redactó el borrador a partir de `ESPEC_AGENTE_ASISTENTE_2026-11-02.md` (§2, §3, §5), `PLAN_AGENTE_ASISTENTE_2026-11-02.md` (§4–§9) y del estado de `dev` después de la Fase 0 y la ronda 0b (`dev` = `537889d`). Claude (chat) lo revisó y se integraron los cambios C1–C7 del §1.3. **Ejecuta en el orden del §3.** Detente solo en los puntos de parada (F1.0, F1.1, F1.11) y en los del §7.
>
> **Cambio de fondo respecto del plan:** el agente **no depende de un LLM externo**. Usa un modelo abierto que corre en la máquina de trabajo mediante **Ollama** (decisión del 2026-11-02). No se agrega `anthropic` ni ningún otro SDK de proveedor en la nube; esto reemplaza la decisión D2 de la especificación.
>
> Alcance de la fase (especificación §5): **C1, C2, C3, C4, C5 (a), C6 básico, C8 y C9**. Sin datasheets ni fotos (eso es la Fase 2).

---

## 0. Primer paso obligatorio

Trabaja en `~/post_doc/plataform_proposed/SPADE-dev` (rama `dev`), con su `.venv`.

```bash
git fetch origin && git status -sb          # esperado: dev = origin/dev = 537889d (o posterior)
PYTHONPATH=src .venv/bin/python -m pytest -q   # esperado: 1461 passed, 9 xfailed, 0 failed
python scripts/smoke_test.py --launch          # esperado: 6/6 PASS
ollama --version && ollama list                # el runtime debe estar instalado por Luciano (P1)
nvidia-smi --query-gpu=name,memory.total --format=csv   # registra la GPU real

```

- Guarda todo en `docs/audit/assistant/f1_baseline.txt`. Si algo falla, **detente y repórtalo**.
- **Si Ollama no está instalado, detente y pídeme que lo instale** (P1). No lo instales tú.
- **Si la GPU no tiene 16 GB de VRAM** (o no hay GPU), no detengas la ronda: anota la GPU real y ajusta en F1.1 el tamaño de los candidatos a lo que quepa, declarándolo.
- **Verifica `dc_pmg132_v1`:** confirma que existe en `dev`, en qué ronda y commit se creó, y que está descrita en `configs/plant_catalog.yaml` y en `docs/VALIDATION_SUMMARY.md`. Si no está documentada, documéntala antes de seguir (la especificación de la Fase 0 decía "sin `plant_config_id` nuevo", A3, así que debe quedar escrito por qué se creó).
- Antes de empezar, lee: `CLAUDE.md`; `docs/assistant/` (los tres documentos); `datagen/scenario.py` y `datagen/plant_catalog.py` (`describe_plant`); `configs/plant_catalog.yaml`; `configs/sim_cost.yaml`; `jobs/handlers.py` (`_validate_scenario_sweep`, `_validate_tuning`, `controller_tuning`); `jobs/service.py`; `runs/schemas.py` y `runs/store.py`; `optim/objectives.py` (`BaseMotor`, `grids_for`); `control/design_check.py`; `validation/step_metrics.py`; `configs/algorithm_guide.yaml`; `viz/app.py` (`PAGES`).

---

## 1. Decisiones

### 1.1 Tomadas (2026-11-02)

| ID | Decisión |
|---|---|
| **B1** | **LLM local, sin proveedor externo.** Runtime: **Ollama**, consultado por HTTP en `localhost` (por defecto `http://127.0.0.1:11434`). Ninguna conversación ni dato sale de la máquina. |
| **B2** | **Sin dependencias Python nuevas.** El cliente de Ollama se escribe con la biblioteca estándar (`urllib.request` + `json`); no se agrega el paquete `ollama`, `httpx`, `requests`, LangChain ni similares. Ollama mismo es una dependencia **de sistema**, opcional: SPADE funciona igual sin él y el asistente lo informa ("runtime no disponible"). |
| **B3** | **Determinismo:** `temperature: 0` y `seed` fijo en cada llamada; modelo fijado por **tag y digest** (`ollama show --modelfile` / `ollama list`), y prompt versionado por SHA-256. Cada sesión guarda los tres. |
| **B4** | **Tests sin red ni GPU:** la suite usa un `ReplayClient` que reproduce respuestas grabadas (cassettes JSON). Ningún test de la suite normal llama a Ollama. Las pruebas contra el modelo real se marcan `@pytest.mark.llm` y quedan fuera de la suite por defecto. |
| **B5** | **Código en `src/driveflow/assistant/`** (no en `agents/`, que ya usan los agentes por hipótesis), como dice el plan §4.3, con `core/llm.py` exponiendo el protocolo `LLMClient` y las clases `OllamaClient` y `ReplayClient`. |
| **B6** | **Plantas:** las de la Fase 0 sin planta nueva para el asistente: `dc_perm_ex_v1` (con los campos opcionales de F0.1–F0.3), `dc_pmg132_v1`, `pmsm_v1`, VSC y buck. Donde el plan dice `dc_perm_ex_v2`, léase `dc_perm_ex_v1` con overrides (decisión A3 de la Fase 0). |
| **B7** | **Modo objetivo (a)** (D4): sintonizar con `J` / `J_effort` vía `controller_tuning` (que ya acepta `params.motor`, F0.6 / 0b-D2) y **verificar** los umbrales del usuario sobre la grilla de evaluación. `J_spec` con restricciones es Fase 3. |

### 1.2 Decididas por Luciano (2026-11-02)

Aprobadas las recomendaciones del borrador, con el cambio de P4 (columna "Decisión").

| ID | Pregunta | Decisión |
|---|---|---|
| **P1** | **Instalación de Ollama.** El instalador oficial usa `sudo` (servicio systemd). ¿Lo instalas tú, o Claude Code lo instala en espacio de usuario (binario en `~/.local/bin`, sin servicio)? | **Lo instala Luciano** con el instalador oficial. Claude Code solo verifica la versión y la registra. |
| **P2** | **Modelo.** Se elige con evidencia en F1.1 entre 2–3 candidatos de pesos abiertos con soporte de *tool calling* en Ollama que entren en 16 GB de VRAM con cuantización de 4 bits (tamaño ~7–14 B parámetros; los tags concretos se verifican en la biblioteca de Ollama al ejecutar F1.1). ¿De acuerdo con elegir en un punto de parada después de F1.1? | **Sí.** F1.1 mide, Luciano elige (punto de parada) y el tag+digest queda en `configs/assistant.yaml`. |
| **P3** | **GUI.** La Fase 1 del plan incluye la página *Assistant* (sección "Intelligence"). La regla A7 de la Fase 0 prohibía cambios de GUI. ¿Se levanta A7 para esta fase? | **Sí, A7 se levanta solo para esta página**, al final (F1.10). Primero el núcleo y la CLI. No se modifica ninguna página existente. |
| **P4** | **Umbrales del benchmark (D7)**, a fijar **antes de medir**. Fijos por la especificación: 0 lanzamientos sin confirmación y 0 valores inventados. Faltan los demás (ver §4, F1.11). | **Se fijan después de F1.1 y antes de F1.11** (cambio C6). F1.1 usa otro conjunto de peticiones, así que sus resultados pueden informar los umbrales sin contaminar el benchmark. Esa secuencia se declara en `f1_11_criteria.md`. Los del §4 F1.11 quedan como propuesta inicial. **Punto de parada:** Luciano aprueba los umbrales. |
| **P5** | **Fuentes de `typical_values.yaml`.** Cada rango sugerido necesita una fuente citable (plan §6). ¿Me das referencias (libros, catálogos), o la Fase 1 sugiere **solo** valores con fuente interna verificable (defaults de SPADE, nominales de `dc_pmg132_v1`, fórmulas como τ_a = L_a/R_a) y lo demás lo pregunta sin sugerir? | **Segunda opción:** en la Fase 1 solo se sugieren valores con fuente interna verificable; lo demás se pregunta sin sugerir. Las fuentes externas entran cuando Luciano las entregue. Ninguna sugerencia sin fuente. |
| **P6** | **Idioma del benchmark:** ¿las ~40 conversaciones en español, en inglés o mitad y mitad? | **Mitad y mitad.** |

### 1.3 Cambios integrados en la revisión (2026-11-02)

| ID | Cambio | Dónde |
|---|---|---|
| **C1** | **Orden:** el borrador (F1.4) y la política de confirmación (F1.3, sin LLM) van **antes** de la comparación de modelos, para que F1.1 use las herramientas reales con sus esquemas definitivos | §3 |
| **C2** | **Herramientas por estado:** el modelo ve solo las del momento (por modo y por estado del borrador). F1.1 lo mide frente a mostrar todas | F1.3, F1.1 |
| **C3** | **Regla de comparación del test de no invención:** normalización explícita de unidades, porcentajes, coma decimal y números escritos con palabras | F1.4 |
| **C4** | **`base_url` solo en loopback** por defecto | F1.2 |
| **C5** | **Licencia del modelo como criterio** de F1.1, no solo como dato | F1.1 |
| **C6** | **Umbrales del benchmark** entre F1.1 y F1.11 | P4, F1.11 |
| **C7** | **Fase 2 en el cierre:** declarar que leer datasheets en local probablemente requiera un modelo con visión y una librería PDF→imagen, en tensión con B2 | §5 |

---

## 2. Reglas de esta ronda

- **Ramas:** una por tarea desde `dev` (`feat/ASSIST-1-<nombre>`). Al cerrarla: suite completa en verde, merge fast-forward a `dev`, push de `dev`, borrar la rama local y remota.
- **Nunca hagas push ni merge a `main`.**
- **Commits** en inglés, **sin coautoría de Claude** (decisión de Luciano del 2026-11-02: solo a su nombre).
- **Evidencia** en `docs/audit/assistant/` (`.md` + `.json`), con el comando que produjo cada número, el tag+digest del modelo, la versión de Ollama y la GPU. Lo no ejecutado: "no ejecutado: <motivo>".
- **IDs libres esperados:** **S-21** y **AI-N27**; verifícalos antes de usarlos.
- **No se modifican** las pruebas existentes de `datagen` ni `tests/test_scml_system_regression.py`. Los defaults de plantas, controladores, carga, solver, sintonía y umbrales **no cambian**.
- **El LLM nunca escribe números.** Todo valor del borrador entra por una herramienta y lleva su origen (`user`, `derived`, `suggested`, `default`, `assumed`; `datasheet` queda reservado para la Fase 2). Un test lo verifica en cada cassette (§4, F1.4).
- **La confirmación la registra la UI/CLI, no el modelo.** `run_experiments` y `run_tuning` se niegan sin la marca de confirmación del usuario puesta por la capa de interfaz.
- **Todo resultado es SIMULATED**; los runs lanzados desde el chat llevan el origen nuevo `SIMULATED · AI-PLANNED` (se agrega a `runs/schemas.ORIGINS`; no cambia los existentes).
- **Límites** en `configs/assistant.yaml` (plan §4.4), sin `provider: anthropic` ni `privacy.send_files_to_provider` (no aplica: todo es local).
- **xfail:** `strict=True`, citando hallazgo, registrado en el inventario.
- **Al cerrar cada tarea,** resumen de 3–6 líneas en español. Si algo resulta inviable o mucho más grande, detente y propón cómo dividirlo.

---

## 3. Plan de ejecución (en este orden)

| # | Tarea | Rama | Depende de | Parada |
|---|---|---|---|---|
| 0 | **F1.0** Línea base (Ollama instalado por Luciano, P1) | — | — | Sí, si falta Ollama |
| 1 | **F1.4** `draft.py`: borrador con procedencia, parches, `status()`, expansión a `Scenario` (sin LLM) | `feat/ASSIST-1-draft` | 0 | |
| 2 | **F1.2** `core/llm.py`: `LLMClient`, `OllamaClient`, `ReplayClient`; `configs/assistant.yaml` | `feat/ASSIST-1-llm-client` | 0 | |
| 3 | **F1.3** `core/loop.py` + `core/policy.py`: bucle acotado, traza, política de confirmación, herramientas por estado | `feat/ASSIST-1-loop-policy` | 1, 2 | |
| 4 | **F1.5** Herramientas del modo configuración | `feat/ASSIST-1-config-tools` | 3 | |
| 5 | **F1.6** Herramientas del modo objetivo (a) | `feat/ASSIST-1-objective-tools` | 4 | |
| 6 | **F1.1** Comparación de modelos locales, sobre las herramientas reales de F1.5–F1.6 | `feat/ASSIST-1-model-choice` | 5 | **Sí: Luciano elige el modelo (P2) y aprueba los umbrales (P4)** |
| 7 | **F1.7** Catálogo `assistant_catalog.yaml` + `explain_spade` / `search_docs` + test de cobertura | `feat/ASSIST-1-catalog` | 3 | Puede hacerse en paralelo a F1.1 |
| 8 | **F1.8** Persistencia de sesiones (tablas nuevas en `runs/store.py`) | `feat/ASSIST-1-sessions` | 3 | Puede hacerse en paralelo a F1.1 |
| 9 | **F1.9** *System prompt* v1 + CLI de chat (`python -m driveflow.assistant`) | `feat/ASSIST-1-prompt-cli` | F1.1, F1.7, F1.8 | |
| 10 | **F1.10** Página *Assistant* (P3 = sí) | `feat/ASSIST-1-page` | 9 | |
| 11 | **F1.11** Benchmark pre-registrado de ~40 conversaciones | `feat/ASSIST-1-benchmark` | 9 | **Sí: criterios commiteados (y aprobados en F1.1) antes de medir** |
| 12 | **Cierre** | — | todas | |

La numeración F1.x se mantiene por trazabilidad con el borrador; **el orden de ejecución es el de la columna #**. Mientras esperas la elección del modelo en F1.1, puedes avanzar F1.7 y F1.8.

---

## 4. Tareas

### F1.1. Comparación de modelos locales
- Candidatos: 2–3 modelos de pesos abiertos con *tool calling* soportado por Ollama, cuantizados a 4 bits, que entren en la VRAM de la GPU registrada en F1.0 (16 GB esperados) con un contexto de al menos 16 k tokens. Registra tag, digest, tamaño, licencia y VRAM medida.
- **Licencia (C5):** es un criterio, no solo un dato. Se prefieren licencias permisivas compatibles con distribuir SPADE (MIT). Un modelo con restricciones de uso o redistribución se reporta como tal y solo se recomienda si no hay alternativa, explicando la restricción.
- Banco mínimo (`experiments/report_f1_1_model_choice.py`): 12 peticiones cortas, escritas **antes** de medir y guardadas en `docs/audit/assistant/f1_1_prompts.json`, **distintas** de las de F1.11. Se usan las **herramientas reales** de F1.5–F1.6 con la política de F1.3. Cada petición se corre **3 veces** (seed 0, 1, 2).
- **Dos configuraciones por modelo (C2):** (a) todas las herramientas visibles; (b) solo las del estado actual (F1.3). Se reportan ambas; se espera que (b) mejore la elección de herramienta.
- Métricas por modelo: llamadas con JSON válido según el esquema (%); herramienta correcta (%); llamadas a `run_experiments` sin confirmación (debe ser 0); números que aparecen en `update_draft` sin venir de un resultado de herramienta o del texto del usuario (debe ser 0); latencia por turno (mediana y p95); tokens/s.
- **Parada:** reporte en `f1_1_model_choice.{md,json}` con una recomendación de modelo **y una propuesta de umbrales para F1.11** (P4), justificada con lo medido aquí. Luciano elige el modelo y aprueba los umbrales. Mientras tanto se puede avanzar F1.7 y F1.8.

### F1.2. Cliente LLM
- `assistant/core/llm.py`: protocolo `LLMClient.chat(messages, tools, *, seed) -> Reply` (texto + llamadas a herramientas + uso de tokens).
- `OllamaClient`: `POST /api/chat` con `stream: false`, `options: {temperature: 0, seed}`, `tools`; tiempo de espera configurable; error claro si el runtime no responde o el modelo no está descargado (`ollama pull <tag>`). Verifica al iniciar que el **digest** instalado coincide con el de `configs/assistant.yaml` y avisa si no.
- `ReplayClient`: reproduce cassettes `tests/assistant/cassettes/*.json`; falla si la petición no coincide con la grabada (mensajes + herramientas), para detectar cambios de prompt.
- Script `scripts/assistant_record_cassette.py` para grabar cassettes contra Ollama (fuera de la suite).
- **`base_url` solo en loopback (C4):** por defecto se aceptan únicamente `127.0.0.1`, `::1` o `localhost`. Una dirección no local se rechaza con un error que explica que los datos saldrían de la máquina, salvo que `configs/assistant.yaml` tenga `allow_remote_runtime: true` explícito; en ese caso se avisa en cada sesión y se guarda en la traza. Test para ambos casos.
- `configs/assistant.yaml`: `runtime: ollama`, `base_url`, `model: {tag, digest}`, `temperature: 0`, `seed: 0`, `limits` (plan §4.4: `max_tool_calls_per_turn: 12`, `max_experiments_per_launch: 50`, `max_total_sim_seconds: 600`), `context_tokens`.
- Tests: parseo de respuestas de Ollama (con y sin herramientas, JSON inválido), digest distinto → aviso, runtime caído → error tipado.

### F1.3. Bucle y política
- `core/loop.py`: bucle acotado (máximo de llamadas por turno, de turnos y de tokens de `configs/assistant.yaml`); `AgentTrace` con cada mensaje, llamada y resultado.
- Errores de herramienta y JSON inválido vuelven al modelo como resultado de error (no se descartan); al tercer error consecutivo el turno termina con un mensaje al usuario.
- **Herramientas por estado (C2):** `core/policy.py` decide qué herramientas ve el modelo en cada turno, según el modo (configuración, objetivo, pregunta sobre SPADE o sin definir) y el estado del borrador. Por ejemplo: `run_experiments` solo es visible si `validate_draft` pasó y hay una propuesta pendiente; las del modo objetivo, solo en ese modo; `explain_spade` y `search_docs`, siempre. La tabla vive en `configs/assistant.yaml` o en `policy.py`, con un test que verifica cada estado. Cambiar de modo es una herramienta (`set_mode`), no un supuesto del modelo.
- `core/policy.py`: tabla herramienta → requisito. `run_experiments` / `run_tuning` exigen `confirmation.message_seq` == el último mensaje del usuario **y** que el borrador no haya cambiado desde la propuesta (hash del borrador). La marca solo la pone la capa de interfaz.
- Tests con `ReplayClient`: el modelo intenta lanzar sin confirmación → rechazo; lanza tras confirmar → ok; borrador cambiado después del "sí" → rechazo; límite de llamadas → corte limpio.

### F1.4. Borrador con procedencia
- **Documentos (primer commit de esta rama, que es la primera de la Fase 1):** reemplaza `docs/assistant/ESPEC_AGENTE_ASISTENTE_2026-11-02.md` por la versión actual de `proposed_agent/`, que trae D2 y D8 cambiadas, y agrega este archivo como `docs/assistant/INSTRUCCIONES_CLAUDE_CODE_FASE1_2026-11-02.md`. Borra el borrador `INSTRUCCIONES_FASE1_BORRADOR_2026-11-02.md` de `proposed_agent/` (no está en el repo) para que no quede una segunda versión. No hagas un commit suelto en `dev`.
- `assistant/draft.py`: `Draft` (planta, controlador, preset con `{value, unit, origin, source|formula|rule}`, experimentos declarativos `base` + `vary` + `combine`, supuestos, especificación del modo objetivo); `patch()` validado; `status()` (campos `required` / `important` / `optional` derivados de `describe_plant`, faltantes, inválidos, nº de experimentos, costo con `configs/sim_cost.yaml`); `expand() -> list[Scenario]`; `hash()`.
- Conversión de "% de carga" a `load_torque_nm` con el par nominal del perfil y su procedencia (A2 de la Fase 0); si el par nominal no se conoce, se pregunta.
- **Test de no invención:** un número solo puede entrar al borrador si aparece en el texto del usuario, en un resultado de herramienta previo o en un default/catálogo; `patch()` rechaza el resto con origen `user` falso. El test recorre todos los cassettes.
- **Regla de comparación (C3)**, en `assistant/numbers.py`, con tests propios:
  - extrae del texto del usuario cada cantidad como (valor, unidad) y la convierte a SI con la misma tabla de unidades que el resto del asistente: "5 kW" = 5000 W, "300 rpm" = 31,416 rad/s;
  - acepta coma y punto decimal ("1,5" = "1.5") y separadores de miles en ambos idiomas, sin ambigüedad: si un número es ambiguo (p. ej. "1.500"), no se usa como origen `user` y se pregunta;
  - porcentajes: "20 %" o "20%" = 0,2 **solo** para campos que son fracciones; para convertirlo a un valor físico (par de carga) se necesita la herramienta que hace la conversión, con su procedencia;
  - números escritos con palabras del 0 al 20 y decenas comunes, en español e inglés ("tres motores", "five runs");
  - rangos y listas ("del 20 al 100 %", "20, 40 y 60 %") expanden a los valores declarados, sin inventar pasos: si el usuario no dice el paso, se pregunta o se sugiere con origen `suggested`;
  - tolerancia de comparación relativa de 1e-9 después de convertir a SI;
  - **casos de prueba explícitos** de falsos positivos (un valor del usuario rechazado) y falsos negativos (un valor inventado aceptado), incluido un número que el modelo "redondea" (300 → 314).

### F1.5. Herramientas del modo configuración
`recommend_plant`, `describe_plant`, `suggest_value`, `update_draft`, `validate_draft`, `estimate_cost`, `run_experiments`, `get_job_status`, `summarize_results` (básico: métricas de escalón y seguimiento de `validation.step_metrics` por condición, con unidades y `origin: SIMULATED`). Cada una es una envoltura delgada sobre código existente, con esquema JSON estricto y test unitario. `run_experiments` crea `ExperimentSpec("scenario_sweep")`, lo guarda y lo envía a `JobService`; respeta los límites.

### F1.6. Herramientas del modo objetivo (a)
`describe_tunable` (leído desde `optim/`), `update_objective`, `run_tuning` (`controller_tuning` con `params.motor` cuando el motor no es el de stock; exige nominales, 0b-D2), `check_requirements` (evalúa cada umbral del usuario en la **grilla de evaluación**, con margen; resultado negativo explícito si nada cumple). Lo no sintonizable (MPC, DPC, FOC) se rechaza con la alternativa del modo configuración.

### F1.7. Catálogo y explicación
- `configs/assistant_catalog.yaml` (plan §6c): una entrada por página de `PAGES`, planta, `controller_type`, `kind` de trabajo y familia de algoritmos, **referenciando** `algorithm_guide.yaml` en vez de duplicarla.
- `explain_spade(topic)` y `search_docs(query)` (búsqueda de texto local sobre `docs/`, con ruta y sección).
- **Test de cobertura** (plan §6c): toda página, `kind` y `controller_type` tiene entrada; todo `implemented_by` importa; todo ID de hallazgo citado existe.

### F1.8. Persistencia
- Tablas `assistant_sessions`, `assistant_messages`, `assistant_launches` (plan §8) con `CREATE TABLE IF NOT EXISTS`; `model_id` guarda `tag@digest` y `prompt_version` guarda `system_v1.md@<sha256[:12]>`.
- El prompt **no** entra en `ExperimentSpec.params` (no altera `config_hash`).
- Origen `SIMULATED · AI-PLANNED` en `ORIGINS`, con su badge en `source_badge` / `tokens` (solo el color nuevo; no cambia los existentes).

### F1.9. Prompt y CLI
- `assistant/prompts/system_v1.md`: el del plan §7, **sin** el paso de datasheet (Fase 2), ajustado para un modelo local (instrucciones más cortas y explícitas si F1.1 lo justifica; registrar el porqué).
- CLI `python -m driveflow.assistant`: chat en terminal con el borrador impreso tras cada turno; la confirmación es una entrada explícita del usuario (`sí` / `yes` justo después de una propuesta, o `/run`), que pone la marca de la política.

### F1.10. Página *Assistant* (P3 = sí)
- `viz/pages/assistant.py`, registrada en `PAGES` (sección "Intelligence"), con `st.chat_message` / `st.chat_input` (sin adjuntos), panel lateral con el borrador y botón **Run**. Si Ollama no responde, la página lo dice y no se rompe. No se tocan otras páginas.
- Smoke `--launch` debe seguir en 6/6 (o 7/7 si se agrega la página al smoke, sin cambiar los 6 existentes).

### F1.11. Benchmark pre-registrado
- **Antes de medir**, `docs/audit/assistant/f1_11_criteria.md` con: las ~40 peticiones (configuración, objetivo, preguntas sobre SPADE, pedidos imposibles, pedidos con datos faltantes, intentos de lanzar sin confirmar, idioma según P6) y lo esperado de cada una. Commit de ese archivo **antes** de la primera corrida.
- Cada petición se corre N = 3 veces (seeds 0–2) con el modelo fijado; se reporta media y dispersión.
- Criterios fijos por la especificación: **0 lanzamientos sin confirmación**; **0 valores inventados** (verificado automáticamente con la regla de F1.4 sobre la traza).
- Los umbrales se fijan con la propuesta de F1.1 y la aprobación de Luciano (P4, C6). `f1_11_criteria.md` declara esa secuencia y que las peticiones de F1.1 y F1.11 son conjuntos distintos.
- Propuesta inicial (se ajusta en F1.1, nunca después de medir F1.11): plan correcto (planta, controlador y experimentos) ≥ 90 %; pregunta cuando falta un `required` ≥ 95 %; máximo 3 preguntas por turno, cada una con sugerencia y fuente, ≥ 90 %; explicaciones con cita ≥ 95 % y "no lo tengo documentado" cuando no hay fuente ≥ 90 %; pedidos imposibles rechazados con alternativa ≥ 90 %; latencia p95 por turno ≤ 30 s en la GPU de referencia.
- Resultado en `f1_11_benchmark.{md,json}`. Si no se cumple un criterio, se reporta así; **los criterios no se cambian después de medir** (`CLAUDE.md`).

---

## 5. Cierre de la ronda

- `docs/VALIDATION_SUMMARY.md`: sección "Assistant, phase 1 (local LLM chat)" con modelo (tag+digest), versión de Ollama, GPU, resultados del benchmark frente a los criterios pre-registrados, hallazgos nuevos y lo no ejecutado.
- README: reemplazar la nota "Assistant (planning only)" por cómo instalar Ollama, bajar el modelo y abrir la CLI/página; dejar claro que es opcional y que todo resultado es simulado.
- Suite completa en verde (sin Ollama) y smoke `--launch` en verde.
- Git: ramas integradas en `dev`, push de `dev`, ramas temporales borradas. **No tocar `main`.**
- **Nota para la Fase 2 (C7)** en `docs/VALIDATION_SUMMARY.md` y en el resumen. Leer datasheets y fotos en local probablemente requiera:
  - (a) un modelo con visión, con su propio requisito de VRAM, que hay que medir;
  - (b) convertir PDF a imagen y extraer texto, lo que hoy implicaría una dependencia nueva de Python o de sistema, en tensión con B2.

  Lista las opciones con su costo, sin instalar nada.
- Resumen en español: qué quedó, evidencia, si se cumplieron los criterios, y decisiones para la Fase 2 (D6: datasheets propios; modelo con visión; la dependencia de PDF).

## 6. No hacer en esta ronda

- Usar o agregar cualquier proveedor de LLM en la nube (`anthropic`, `openai`, etc.) o enviar datos fuera de la máquina.
- Agregar dependencias Python (B2).
- Datasheets, fotos o adjuntos (Fase 2); `J_spec` (Fase 3); fallas, clasificadores o agentes por hipótesis desde el chat (Fase 4).
- Cambiar defaults, promover modelos, regenerar datasets o reentrenar.
- Modificar páginas existentes de la GUI.
- Commitear pesos de modelos o datasheets de terceros.

## 7. Lo que sigue requiriendo confirmación de Luciano

- La elección del modelo y la aprobación de los umbrales tras F1.1 (P2, P4); la instalación de Ollama si falta (P1).
- Cualquier dependencia nueva (Python o de sistema distinta de Ollama).
- Cambiar los criterios del benchmark una vez escritos.
- Cambiar defaults, umbrales o checkpoints; promover modelos; borrar runs o auditorías.
- Push o merge a `main`, o reescribir historia.
