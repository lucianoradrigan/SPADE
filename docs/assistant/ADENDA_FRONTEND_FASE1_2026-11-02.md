# SPADE · Adenda a la Fase 1: frontend del asistente con el avatar "PhD. Radrigan"

2026-11-02 · Para Claude Code. Complementa `INSTRUCCIONES_CLAUDE_CODE_FASE1_2026-11-02.md` sin cambiar ninguna de sus decisiones. Las reglas del §2 de ese documento siguen valiendo: ramas, sin push a `main`, commits solo a nombre de Luciano y evidencia en `docs/audit/assistant/`.

**Diseño de referencia:** `ESPEC_FRONTEND_PHD_RADRIGAN_2026-11-02.md`. Cópialo a `docs/assistant/` en FE-1.
**Prototipo funcional:** `phd_radrigan_assistant.html`. Es HTML, CSS y JS sin dependencias, con respuestas guionadas. Contiene los estados, las animaciones y la disposición ya probadas en escritorio y en teléfono. **Porta su CSS y su JS; no lo reinventes.**
**Recursos provisionales:** `base.webp` (cuerpo, 560 × 444 px, alfa real, ≈ 35 KB) y `hand.webp` (mano como capa propia, ≈ 11 KB). Se generaron desde la caricatura de Luciano: se le quitó el fondo negro y el texto horneado. Luciano te pasa los tres archivos.

---

## 1. Decisiones (Luciano, 2026-11-02)

| ID | Decisión |
|---|---|
| **FE-D1** | Visión para datasheets (Fase 2): **local**. Se mantiene B1 |
| **FE-D2** | Recursos del avatar: **provisionales** los de este paquete. Los finales (poses, variante clara, capas fuente) llegarán después y reemplazan a estos sin tocar código |
| **FE-D3** | El avatar **queda fuera de la licencia MIT**: es la imagen de Luciano, con licencia propia en `licenses/avatar/`. Se puede **reemplazar por configuración** (`configs/assistant.yaml: persona.avatar_dir`) |
| **FE-D4** | Sin Rive ni Lottie en esta fase: poses + CSS, como el prototipo |
| **FE-D5** | Nombre visible "PhD. Radrigan", siempre junto a la etiqueta **"AI assistant · local model"** |
| **FE-D6** | El avatar **nunca** aprueba resultados: no hay guiño, salto ni sonrisa en `results_ready`. El salto (`confirmed`) ocurre solo cuando **el usuario** confirma un lanzamiento |
| **FE-D7** | `listening` (mientras el usuario escribe) **se omite** con el `st.chat_input` nativo, que no emite eventos de tecleo. No se reemplaza el input nativo en esta fase |

---

## 2. Cuándo va cada parte

| # | Tarea | Cuándo | Rama |
|---|---|---|---|
| **FE-1** | Contrato de eventos del asistente | **Después de S21-1 y antes o durante F1.1**: no depende del modelo y conviene tenerlo antes de grabar cassettes | `feat/ASSIST-1-events` |
| **FE-2** | Recursos y licencia del avatar | Junto con FE-1 o justo después | `feat/ASSIST-1-avatar-assets` |
| **FE-3** | Componente `radrigan_avatar` (porta el prototipo), con página de prueba | **Mientras Luciano elige el modelo** (parada de F1.1), en paralelo a F1.7 y F1.8 | `feat/ASSIST-1-avatar-component` |
| **FE-4** | Integración en la página *Assistant* | **Dentro de F1.10**, que ya estaba planificada | `feat/ASSIST-1-page` (la misma de F1.10) |
| — | Tarjeta de revisión de datasheets (`extraction_review`) | **Fase 2**, no ahora | — |

---

## 3. Tareas

### FE-1. Contrato de eventos (`src/driveflow/assistant/events.py`)
1. **Tipos de evento** de `ESPEC_FRONTEND` §2: `turn_started`, `llm_call` (start/end), `tool_call` (start/end, con `tool` y `ok`), `proposal_ready` (con `draft_hash`, `n_experiments`, `est_wall_s`), `user_confirmed`, `job` (estado, done/total), `results_ready`, `error`, `runtime_unavailable` y `turn_ended`. Cada uno es una dataclass congelada con `to_dict()`.
2. **`core/loop.py` acepta `on_event: Callable[[AgentEvent], None] | None = None`.** Con `None`, el comportamiento y la traza son **idénticos** a los actuales: los tests existentes del bucle y los de política pasan sin cambios. Los eventos también quedan en la traza.
3. **Correspondencia evento → estado del avatar** como función pura, `viz/adapters/assistant_adapter.avatar_state(events) -> (state, label, progress)`, con la prioridad de `ESPEC_FRONTEND` §2 y la duración mínima de 400 ms de `thinking` resuelta en la UI, no aquí.
4. **Etiquetas legibles por herramienta** (en inglés): `describe_plant` → "Checking the plant…", `estimate_cost` → "Estimating cost…", `validate_draft` → "Validating the experiments…", y así con todas las de `CONFIG_TOOLS` y `OBJECTIVE_TOOLS`. **Test de cobertura:** toda herramienta registrada tiene etiqueta.
5. **Tests:**
   - secuencias de eventos de una conversación guionada (configuración con propuesta, confirmación y job; objetivo; pregunta sobre SPADE; error de herramienta; Ollama caído);
   - el estado resultante en cada paso;
   - **ningún `results_ready` lleva a `confirmed`** (FE-D6).

### FE-2. Recursos y licencia
1. `src/driveflow/viz/assets/assistant/base.webp` y `hand.webp`. **Verifica que entren en el wheel:** `pyproject.toml` empaqueta `src/driveflow` entero. Revisa también el instalador de Windows (`packaging/build_installer.py`, `spade.iss`). Si no incluyen archivos `.webp`, repórtalo y ajusta solo la lista de archivos.
2. `licenses/avatar/LICENSE`:
   - "PhD. Radrigan avatar © Luciano Radrigan. All rights reserved. Not covered by the MIT license of SPADE. Redistribution only unmodified and as part of SPADE.";
   - más un `README.md` que explique cómo reemplazarlo (`persona.avatar_dir`).

   Agrega una línea en el README del repo, en la sección de licencia, que apunte ahí. **No modifiques el archivo `LICENSE` (MIT).**
3. `configs/assistant.yaml`: bloque `persona` con `name: "PhD. Radrigan"`, `tagline: "AI assistant · local model"` y `avatar_dir: src/driveflow/viz/assets/assistant`. Si el directorio no tiene los archivos esperados, la UI usa una silueta neutra y lo registra; nunca falla.
4. **No commitees la caricatura original en JPG.**

### FE-3. Componente `radrigan_avatar`
1. **Implementación:** `src/driveflow/viz/components/radrigan_avatar.py` + `radrigan_avatar/{avatar.html, avatar.css, avatar.js}`, con `st.components.v2.component`, **portando el CSS y el JS del prototipo**:
   - capas `base` + `hand` (pivote de la mano en 60 % / 100 %);
   - `chip-glow` en (60 %, 85,8 %);
   - puntos de pensamiento, anillo de barrido, latido por palabra, saludo, barras de voz y barra de progreso;
   - nombre en neón como texto CSS.
2. **Props:** `state` (los 11 de `ESPEC_FRONTEND` §1.3), `label`, `progress`, `compact` (para pantallas angostas) y `beat` (contador que dispara un latido; lo usa el streaming de la respuesta).
3. **Spike documentado** (`docs/audit/assistant/fe3_spike.md`). Verifica en Streamlit 1.66:
   - (a) cómo se pasan datos y recursos a un componente v2 (en `ESPEC_FRONTEND` §3 está la intención, no la firma verificada);
   - (b) si `isolate_styles=True` permite que la figura invada el borde de la columna del chat (si no, deja la figura dentro de su columna);
   - (c) que `prefers-reduced-motion` apaga las animaciones;
   - (d) que las animaciones se pausan con la pestaña oculta;
   - (e) peso total de los recursos < 1 MB.
4. **Página de prueba**, sin registrarla en la navegación: `scripts/assistant_avatar_demo.py`, que se abre con `streamlit run`. Muestra los 11 estados con botones y reproduce una secuencia de eventos grabada. **No agregues páginas a `PAGES` en esta tarea.**
5. **Tests `AppTest`:**
   - cada estado se renderiza sin excepciones;
   - un estado desconocido produce un error claro;
   - sin recursos, se usa la silueta y la página no se rompe.
6. **Captura** de los 11 estados en tema oscuro en `docs/screenshots/assistant/`. En tema claro el avatar provisional pierde el halo: documéntalo como limitación conocida de los recursos provisionales, no lo arregles con CSS.

### FE-4. Integración en F1.10 (página *Assistant*)
Esto es lo que F1.10 ya pedía, más el avatar:
1. **Disposición del prototipo:** columna del avatar (fija, anclada abajo), columna del chat y panel del borrador con **Run**. En pantallas angostas, el avatar compacto va en la cabecera.
2. **El bucle se llama con `on_event`,** que actualiza el componente en un `st.empty()`:
   - `thinking` se pinta **antes** de la llamada al LLM;
   - el progreso del job se refresca con `st.fragment(run_every="2s")`;
   - la respuesta se muestra en streaming y avanza `beat` cada pocas palabras.
3. **Mensajes fijos:**
   - primer mensaje: "I'm SPADE's AI assistant. I set up and run **simulations**; my suggestions require your verification.";
   - en el pie: "All results are SIMULATED.";
   - el *system prompt* (F1.9) incluye que el asistente **no habla en primera persona como Luciano**, y F1.11 tiene un caso de prueba para eso.
4. **Ollama caído o sin modelo:** estado `unavailable`, mensaje con cómo arrancarlo, y la página sigue funcionando.
5. **Criterios de aceptación** (`ESPEC_FRONTEND` §7):
   - todos los estados se alcanzan desde eventos grabados;
   - sin Ollama, la página no se rompe;
   - `reduced-motion` funciona;
   - contraste del chip ≥ 4,5:1;
   - recursos < 1 MB;
   - no se modifica ninguna página existente;
   - smoke `--launch` en verde;
   - test de FE-D6.

---

## 4. No hacer

- Reemplazar `st.chat_input` por un input propio (FE-D7), agregar Rive, Lottie o cualquier biblioteca JS (FE-D4), o cargar algo desde una CDN: SPADE funciona sin internet.
- Usar un proveedor en la nube para nada (FE-D1).
- Tocar otras páginas de la GUI, o el archivo `LICENSE`.
- Commitear la caricatura original en JPG.
