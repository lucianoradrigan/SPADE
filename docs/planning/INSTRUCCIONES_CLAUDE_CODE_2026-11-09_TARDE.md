# SPADE: instrucciones consolidadas para Claude Code (2026-11-09, tarde)

Este es el único documento de trabajo vigente para Claude Code. Reemplaza a `INSTRUCCIONES_CLAUDE_CODE_2026-11-09.md` y a `INSTRUCCIONES_CLAUDE_CODE_TAREAS_LUCIANO_2026-11-09.md`. Lo pendiente de esos dos documentos está incluido aquí.

Las reglas permanentes siguen en `INSTRUCCIONES_SPADE.md` (2026-11-08), ya fusionadas en `CLAUDE.md`. El estado general está en `PLAN_CONSOLIDADO_2026-11-08.md`.

Ejecuta en el orden del §3 sin pedir confirmación entre tareas, salvo en los puntos de parada del §4. Las decisiones del §1 las tomé yo, Luciano Radrigan, el 2026-11-09. No me las vuelvas a preguntar.

## 0. Primer paso obligatorio

```bash
cd ~/post_doc/plataform_proposed/SPADE-dev
git fetch origin && git status -sb && git branch -a
# esperado: dev = origin/dev = 30b649c; release/v0.2.0 = fb31e52 (local); feat/T7-gem-cont1qc-pr (local); main = df1950f
df -h / /mnt/datos
```

- Si el estado no coincide, detente y repórtamelo.
- Guarda este documento en `proposed_agent/` y en `docs/planning/` (en la rama del release, T2).
- Abre el documento de estado `proposed_agent/ESTADO_RONDA_C_2026-11-09.md` y actualízalo al cerrar cada tarea.

## 1. Decisiones (Luciano, 2026-11-09)

| ID | Decisión |
|---|---|
| C1 | CI: `tests.yml` debe correr con cada push a `dev`, y el CI manual se corre sobre `dev` y sobre `release/v0.2.0` |
| C2 | Release v0.2.0 aprobado con condiciones: limpieza de la rama (T2) y CI en verde. Antes de hacer el merge a `main` y crear el tag, me muestras el comando exacto y el resultado del CI (parada P2) |
| C3 | Licencia del avatar: cubre toda imagen del repo que lo muestre, incluidas las 13 capturas |
| C4 | Documentos obsoletos (los que piden coautoría de Claude, entre otros) van a `docs/archive/` con una nota de "superseded". Los internos en español van a `docs/planning/`. No se borra ninguno |
| C5 | S-22: la tolerancia de 1,5× en esa prueba queda aceptada |
| C6 | S-24, para v0.2.1: el límite de tensión se valida contra la alimentación efectiva, que es 65 V si no se da otra |
| C7 | MCP: se permiten clientes en la nube. Por MCP solo salen configuraciones, resultados simulados y documentación; ninguna herramienta lee archivos del usuario |
| C8 | GEM: integra a `dev` la rama del borrador. Te autorizo a publicar el issue desde mi cuenta (`gh`). Solo el issue: el PR espera a que respondan los mantenedores |
| C9 | Ollama: preparas un script único, yo lo corro con sudo y tú verificas el resultado |
| C10 | Conjunto dorado de la Fase 2: descargas, guardas y usas los datasheets del anexo A y las fotos de placas con licencia abierta. Usar significa leerlos tú y escribir los valores esperados, no correr modelos. La Fase 2 no empieza |
| C11 | B4: después del release, solo el pre-registro, sin ejecutar nada |

## 2. Reglas de esta ronda (resumen)

- Ramas: `feat/<ID>-<nombre>` desde `dev`. Suite completa en verde, merge fast-forward, push de `dev` y borrar la rama.
- Commits en inglés, solo a mi nombre, sin `Co-Authored-By`.
- Nunca hagas push ni merge a `main` sin pasar por P2.
- Evidencia en `docs/audit/<ronda>/` (`.md` + `.json`), con el comando de cada número. Lo que no se ejecutó lleva "no ejecutado: <motivo>".
- E10: como máximo 2 líneas de trabajo abiertas.
- Hallazgos nuevos: S-25 y AI-N32 (verifica antes).
- No toques las plantas v1, las pruebas byte a byte ni ningún default.
- Archivos de terceros con derechos de autor (datasheets, fotos no libres) se descargan a `data/upstream/`, nunca se commitean.
- Todo resultado es SIMULADO.

## 3. Tareas, en orden

| # | Tarea | Rama | Parada |
|---|---|---|---|
| 1 | T1 CI | `feat/C1-ci-dev-trigger` | — |
| 2 | T2 Limpieza del release y P2 | `release/v0.2.0` | P2 |
| 3 | T3 Issue de GEM | `feat/T7-gem-cont1qc-pr` → `dev` | P1 (solo si gh falla) |
| 4 | T4 MCP: datos que salen y prueba con cliente | `feat/C7-mcp-egress` | — |
| 5 | T5 Script de Ollama | `feat/C9-ollama-move` | P3 (lo corro yo) |
| 6 | T6 Limpieza del worktree | — | — |
| 7 | T7 S-24 para v0.2.1 | `feat/S-24-voltage-limit` | P4 (solo si algo existente queda rechazado) |
| 8 | T8 Conjunto dorado: descargar, guardar y usar | `feat/C10-golden-set` | P5 (solo si hay menos de 10 fotos libres) |
| 9 | T9 Pre-registro de B4 | `feat/B4-preregistration` | P6 (lo apruebo yo) |

T3 y T5 se pueden adelantar mientras esperas P2.

### T1. CI
- Agrega `dev` a los disparadores de push de `.github/workflows/tests.yml`.
- Lanza el CI manual: `gh workflow run tests.yml --ref dev` y `gh workflow run tests.yml --ref release/v0.2.0`. Para lo segundo, la rama tiene que estar en el remoto: haz push solo de esa rama, nunca a `main`.
- Espera los resultados y regístralos en `docs/audit/ci/c1_ci.md`. Si `gh` no está autenticado, detente y pídeme `gh auth login`.

### T2. Limpieza de `release/v0.2.0` y P2
- Extiende `licenses/avatar/LICENSE` a toda imagen del repo que muestre el avatar (incluidas `docs/screenshots/assistant/avatar_*.png`) y menciónalo en el README.
- Mueve a `docs/archive/` los documentos obsoletos (`DECISIONES_EXPANSION_2026-10-21.md`, `PENDIENTES_EXPANSION_CLAUDE_CODE.md` y cualquier otro que pida coautoría de Claude o esté superado), con una nota al inicio: "Superseded — kept for history; see CLAUDE.md". No borres ninguno.
- Mueve los documentos internos de planificación en español a `docs/planning/`, con un `README.md` de una línea que explique qué son.
- Cambia `/home/seven/...` por rutas relativas donde sea simple, y lista las que quedan.
- Suite completa y smoke `--launch` en verde, y CI en verde sobre la rama (T1).
- **P2:** muéstrame en un mensaje corto:
  - el resultado del CI;
  - el comando exacto del merge a `main` y del tag `v0.2.0`;
  - el resumen de las notas de versión.

  Espera mi OK antes de ejecutarlo.

### T3. Issue de GEM
- Integra `feat/T7-gem-cont1qc-pr` a `dev`. Es el borrador del issue y del PR, es documentación.
- Verifica `gh auth status`. Vuelve a revisar que no exista un issue igual: `gh issue list --repo upb-lea/gym-electric-motor --state all --search "OneQuadrant"`, y también con "one quadrant", "Cont1QC" y "negative current".
- Publica el issue con el texto final de `docs/audit/upstream/gem_cont1qc_issue_draft.md`. Debe incluir los tres niveles, la mención de #138 y la nota verificada sobre `FiniteOneQuadrantConverter`. No agregues texto que diga que lo escribió una IA: sale a mi nombre.
- Guarda la URL y la fecha en `docs/audit/upstream/gem_cont1qc_issue_published.md`.
- No abras el PR ni hagas push del fork de GEM. Cuando te lo pida, revisas las respuestas con `gh issue view <n> --repo upb-lea/gym-electric-motor --comments` y me las resumes.

### T4. MCP: datos que salen y prueba con un cliente real
- En `docs/assistant/MCP.md`, documenta qué sale de la máquina con un cliente en la nube: configuraciones, resultados simulados y documentación de SPADE. Deja explícito que ninguna herramienta lee archivos del usuario.
- Agrega un test que recorra las herramientas expuestas y falle si alguna acepta una ruta de archivo del usuario o lee del disco fuera del workspace de SPADE.
- Prueba con Claude Code como cliente, en una sesión aparte:
  - registra el servidor (`claude mcp add spade -- <comando del servidor>`);
  - lista las herramientas;
  - corre `describe_plant`, `validate_draft` y `estimate_cost`;
  - pide `propose_experiments` e intenta `run_experiments`.
- Reporta si el cliente soporta *elicitation*. Si no la soporta, `run_experiments` debe negarse. Evidencia en `docs/audit/b3/v4_mcp_client_test.{md,json}`.
- Agrega a `MCP.md` una sección "Try it in Claude Desktop" con:
  - el bloque exacto de `claude_desktop_config.json`, con la ruta del intérprete del `.venv`;
  - tres pruebas de 5 minutos;
  - qué debería pasar en cada una.

  Yo la sigo.

### T5. Script de Ollama
- Escribe `scripts/ops/move_ollama_models.sh`, idempotente y con `set -euo pipefail`. El script:
  - detecta la ruta real del almacén del servicio;
  - verifica el espacio en `/mnt/datos`;
  - detiene ollama;
  - copia con `rsync -a` a `/mnt/datos/ollama/models`;
  - crea el override de systemd con `Environment="OLLAMA_MODELS=/mnt/datos/ollama/models"`;
  - ejecuta `daemon-reload` y reinicia;
  - verifica que `ollama list` muestra `qwen3.5:9b`;
  - renombra el original a `models.bak`, sin borrarlo.
- **P3:** dime en una línea el comando que debo correr (`sudo bash scripts/ops/move_ollama_models.sh`). No intentes usar sudo.
- Cuando yo lo corra, verifica que el digest coincide con `configs/assistant.yaml`, corre el smoke del asistente y actualiza `docs/audit/ops/t0_disk.md`. Dame el comando para borrar `models.bak`; lo borro yo.

### T6. Limpieza
- Ejecuta `git worktree remove --force /mnt/datos/spade/wt_suite` y `git worktree prune`.

### T7. S-24 (para v0.2.1)
- La validación de `Scenario` compara el límite de tensión con la alimentación efectiva: la dada, o 65 V (stock) si no se da.
- Antes de cambiarla, busca escenarios, datasets, presets, pruebas y cassettes que hoy queden aceptados y pasen a rechazarse. Si hay alguno, detente y repórtamelo (P4).
- Agrega pruebas para los dos casos y registra S-24 como corregido. Queda en `dev` para v0.2.1, no en el release v0.2.0.

### T8. Conjunto dorado: descargar, guardar y usar (preparación de la Fase 2)

Objetivo: que los documentos del anexo A queden descargados, verificados, catalogados y con sus valores esperados escritos a mano, listos para cuando empiece la Fase 2. No se corre ningún modelo sobre ellos.

1. **Descarga:** escribe `experiments/fetch_golden_datasheets.py`.
   - Descarga cada documento del anexo A a `data/upstream/assistant_golden/<id>/` (fuera de git).
   - Calcula el SHA-256.
   - Guarda la fecha de consulta y el Content-Type.
   - Si un enlace ya no existe o devuelve otra cosa (HTML de error, login), lo marca como `unavailable` con el motivo. Buscar una URL alternativa del mismo documento del mismo fabricante está permitido; registra la nueva URL.
   - Si el documento es una página web, la guarda como HTML y también como PDF impreso o captura (con la herramienta disponible en el sistema; si hace falta una dependencia nueva, pregúntame).
2. **Catálogo:** `tests/assistant/golden/manifest.yaml`. Por documento:
   - id, URL, fecha y SHA-256;
   - fabricante, tipo de motor, estilo (A–F del relevamiento §2) y trampas (relevamiento §3);
   - licencia o condición de uso;
   - `simulable_today` (DC de imanes permanentes y PMSM: sí; inducción: no);
   - `committed: false`.

   El manifiesto sí se commitea; los archivos no.
3. **Uso, valores esperados escritos a mano:** lee tú cada documento ★ (al menos 12) y escribe `tests/assistant/golden/expected/<id>.json` con:
   - por cada campo: `value_printed`, `unit_printed`, `label_printed`, `basis` (fase / línea-línea / desconocido), `waveform` (rms / pico / equivalente DC / desconocido), `temperature_c`, `condition` y `variant`;
   - `null` cuando el dato no está impreso;
   - si un valor es ambiguo, el esperado es `"ambiguous: must ask"`, no tu interpretación;
   - la lista de lo que el agente debe preguntar y de lo que debe advertir (inconsistencias, etiquetas erróneas, fabricante no nombrado).

   Para documentos con muchas variantes (D5, P2), escribe los valores de 2 variantes elegidas de antemano (la primera y la última columna) y anótalo.
4. **Fotos de placas con licencia abierta:**
   - Wikimedia Commons por API (`https://commons.wikimedia.org/w/api.php`): busca fotos de placas de motores eléctricos ("rating plate", "nameplate", "Typenschild", "Leistungsschild", "motor"; y las categorías de placas de características) y verifica la licencia de cada archivo en su página de información.
     - CC0 o CC BY: se commitean en `tests/assistant/golden/photos/` con `ATTRIBUTION.md` (autor, licencia, URL).
     - CC BY-SA: van a `data/upstream/`.
     - Otras licencias: no se usan.
     - Meta: 15–25 fotos variadas (DC, servo e inducción; nítidas, inclinadas, con reflejos).
   - Ingenium: fichas de la colección con foto y transcripción de la placa, por ejemplo 1992.2220.001. La licencia es no comercial, así que van solo a `data/upstream/` con su atribución y número de artefacto. Úsalas como casos de "no simulable".
   - Para cada foto, escribe `expected/<id>.json` mirando la imagen. Marca lo ilegible como `"illegible"`. Para Ingenium, compara tu transcripción con la de la ficha y anota las diferencias.
   - **P5:** si consigues menos de 10 fotos con licencia abierta, detente y repórtamelo con alternativas.
5. **Uso, conversiones:** agrega al relevamiento (`docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md`, en una sección nueva fechada) y a `configs/assistant_conversions.yaml` las reglas del white paper de Kollmorgen (R1 del anexo A), cada una con su fuente:
   - Ke pico ↔ Kb rms (√2);
   - fase ↔ línea (√3; ÷3 en triángulo);
   - Kt pico ↔ rms (√1,5);
   - chequeos de razón Kb/Kt ≈ 60–65 y Ke/Kt ≈ 103–113.

   Es la fuente externa citable para las conversiones (P5 de la Fase 1). Agrega un test que aplique esos chequeos a los expected de P1–P5 y reporte (sin fallar) qué documentos cuadran, cuáles no y bajo qué convención.
6. **Evidencia:** `docs/audit/assistant/golden_set.{md,json}`, con:
   - el conteo por tipo, estilo, trampa y licencia;
   - qué se descargó y qué quedó `unavailable`;
   - los resultados de los chequeos de convención;
   - lo que no se pudo hacer.

### T9. Pre-registro de B4 (no ejecutar)
- Escribe `docs/audit/b4/b4_preregistration.md`. Ronda de modelos consolidada:
  - datos con plantas v2;
  - grilla extendida por debajo de 80 rad/s (AI-N14 y AI-N29);
  - el ESP32 (AI-N28), con candidatos que incluyan condiciones sanas fuera de la grilla en el entrenamiento;
  - una sola visita a la C2000 con la ventana a 10 kHz (AI-N26).
- Incluye: conjuntos de evaluación nuevos, criterios por nivel (los de AI-N27: compuerta D1, no inferioridad de −1 punto, ninguna clase −5 puntos, abstención ±5 puntos), qué se hace si un nivel no cumple, y el costo estimado de CPU y de trabajo manual.
- **P6:** tráemelo para aprobarlo. No generes datos ni entrenes nada.

## 4. Puntos de parada

| ID | Cuándo | Qué me traes |
|---|---|---|
| P1 | `gh` no autenticado (T1, T3) | Pídeme `gh auth login` |
| P2 | T2 lista | Resultado del CI, comandos de `main` y del tag, resumen de las notas. Esperas mi OK |
| P3 | T5 listo | La línea con sudo que debo correr |
| P4 | T7: algo existente quedaría rechazado | Lista y opciones |
| P5 | T8: menos de 10 fotos libres | Conteo y alternativas |
| P6 | T9 listo | El pre-registro de B4 |
| Siempre | Un hallazgo que haga que SPADE diga algo falso, o una prueba byte a byte que se rompe | Evidencia y opciones |

## 5. No hacer

- Hacer push o merge a `main`, o crear el tag, sin el OK de P2.
- Abrir el PR de GEM o publicar el fork.
- Usar sudo.
- Commitear datasheets o fotos con licencia restrictiva.
- Correr el extractor o cualquier LLM sobre el conjunto dorado. La Fase 2 no empezó.
- Ejecutar B4.
- Tocar las plantas v1, las pruebas byte a byte o los defaults.
- Agregar dependencias sin preguntar.

## Anexo A. Documentos del conjunto dorado

★ = con valores esperados escritos a mano en T8.3. Todos son públicos, con derechos de autor: se descargan, no se commitean. "(rel.)" = ya descrito en `RELEVAMIENTO_DATASHEETS_2026-11-02.md` §1.

### DC de imanes permanentes (simulable)

| ID | Documento | Qué prueba |
|---|---|---|
| ★ D1 | Heinzmann PMG 132 (rel.) | Verdad conocida (`dc_pmg132_v2`); 5 variantes de tensión en columnas |
| ★ D2 | Parvalux PM11 110 V (rel.) | Etiqueta errónea "Thermal resistance (Ra)" |
| ★ D3 | Motenergy ME1514 (rel.) | Kt y Ke difieren un 12 %: debe preguntar |
| D4 | Bodine 33A 6435 (rel.) | Inercia sin unidad; pares "90/130" |
| ★ D5 | Portescap, catálogo DC sin núcleo | Unas 120 variantes de bobinado; Kt en mNm/A, V/1000 rpm, J ×10⁻⁷ kg·m² |
| ★ D6 | Motor DC 31×51, serie 1.13.021.3XX | Fabricante no nombrado (no inventarlo); Ncm/A y g·cm²; 12 V y 24 V |
| D7 | Portescap Athlonix 17DCT | Por verificar al descargarlo |

### PMSM, servo y BLDC (simulable como PMSM)

| ID | Documento | Qué prueba |
|---|---|---|
| ★ P1 | Siemens 1FK7042 (rel.) | kE a 20 °C; Kt a 60 K y 100 K; τ_el comprueba R y L |
| ★ P2 | Kollmorgen AKM6 | 14 variantes y 5 tensiones de bus; Nm/Arms; "25 °C" sin un parámetro claro |
| ★ P3 | Kollmorgen B23 (rel.) | Kt pico línea-línea; τ_el ≠ L/R (16 %) |
| ★ P4 | Delta ASDA-B3 / ECM-B3 | KE en mV/rpm; sin número de polos: debe preguntarlo |
| ★ P5 | Nanotec DB59C024035-B3 | R y L línea-línea; L a 1 kHz; Vrms/krpm; "polos", no pares |
| P6 | Nanotec DB59C048035-A | La misma familia a 48 V: consistencia entre variantes |
| ★ P7 | EMRAX 228 (rel.) | Ld/Lq; 9 columnas; convención de Ke y Kt ambigua |
| P8 | maxon EC 60 flat (rel.) | 6 variantes; R y L fase-fase; g·cm² |
| P9 | TI BLDC17F (rel.) | Variantes en filas; oz-in/A; oz-in-s² |
| ★ P10 | Bosch Rexroth MSK061C (hoja de reventa) | "No alcanza": solo corriente y velocidad; "7.500 1/min" |
| P11 | Yaskawa SGM7J-02D (rel.) | Sin constantes eléctricas; coma decimal |

### Inducción (solo extracción; no simulable hoy)

| ID | Documento | Qué prueba |
|---|---|---|
| ★ I1 | WEG W22 16616834 (rel.) | 4 ratings; unidades NEMA; rodamientos; debe decir "no simulable" |
| I2 | Baldor GNEM3309T (rel.) | R línea-línea a 25 °C; "//" para 50 y 60 Hz |

### Fuente de las conversiones (no se extrae)

| ID | Documento | Uso |
|---|---|---|
| R1 | Kollmorgen, *Servo Motor Parameters and their Proper Conversions* (WP RevD) | Reglas √2, √3, ÷3, √1,5 y chequeos de razón (T8.5) |

### Fotos de placas

| Fuente | Licencia | Destino |
|---|---|---|
| Wikimedia Commons (API, T8.4) | CC0 / CC BY: se commitean con atribución · CC BY-SA: `data/upstream/` | 15–25 fotos variadas |
| Ingenium y otras fichas de la misma colección | No comercial: no se commitean | Placas antiguas de inducción, con transcripción oficial |
| Fotos propias de Luciano (D6) | Propias | Se suman cuando existan |
