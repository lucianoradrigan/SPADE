# SPADE / driveflow — Qué hace la plataforma hoy

Snapshot funcional completo, generado el 2026-09-14. Para el diseño detallado de la capa de IA ver
[`design_ai_layer_transversal.md`](design_ai_layer_transversal.md); para el roadmap de fases
A/B/C/D/E ver [`INSTRUCTIONS.md`](INSTRUCTIONS.md). Este documento es una foto del estado actual,
no un plan — va a quedar desactualizado a medida que el proyecto avance.

## Qué es

Un workbench de simulación + control + diagnóstico por ML para motores eléctricos y convertidores
de potencia, con un dashboard Streamlit en vivo. Tres sistemas físicos distintos, unificados bajo
una interfaz común, más una capa de IA transversal (clasificadores/regresores, despliegue a
hardware edge, agentes de monitoreo basados en reglas).

Nada en el dashboard es pre-generado ni sacado de una tabla: cada gráfico es la salida directa de
una simulación real, corrida en el momento en que se hace click en "Generate".

## 1. Los tres sistemas físicos

### Sistema 1 — Motor DC (Fase A, dominio `dc_motor`)

- **Planta**: `DcPermanentlyExcitedMotor`, misma física/parámetros que
  [gym-electric-motor](https://github.com/upb-lea/gym-electric-motor).
- **Controlador**: PI cascada nativo o MPC lineal (`controller_type="PI"`/`"MPC"`, intercambiables
  sobre la misma planta), modo velocidad o torque.
- **Fallas**: fallas de rodamiento (`outer_race`, `inner_race`, `ball`, `cage`, o combinaciones
  custom) inyectadas en dos caminos físicamente independientes — ripple de torque
  (eléctrico/MCSA) y vibración sintética 3 ejes.
- **Ruido**: jitter tipo sensor, independiente de las fallas, opcional (0% por defecto).
- **Modo "Advanced Flow"**: secuencias de varios estados de operación corridas una tras otra, no
  solo simulaciones únicas.
- **Salidas**: `i_a`, `rpm`, `torque`, `u_a`, vibración 3 ejes (`acc_x/y/z`).

### Sistema 2 — PMSM FOC/MTPA

- **Planta**: PMSM saliente, **no conectada al pipeline de fallas/diagnóstico** (no tiene
  `plant_config_id`, nunca pasa por `Scenario`/`run_scenario`).
- **Controlador**: FOC nativo en marco dq, comparación MTPA vs. naive a la misma magnitud de
  corriente.
- Es una demo de comparación de leyes de control, no un escenario de diagnóstico.

### Sistema 3 — VSC / DPC (Fase B, dominio `vsc_dpc`)

- **Planta**: Voltage Source Converter — electrónica de potencia pura, sin partes rotantes ni
  rodamientos.
- **Controlador**: red neuronal Direct Power Control (DPC) ya entrenada, portada de
  [DPC4PowerElectronics](https://github.com/aipoweraau/DPC4PowerElectronics).
- **Sondas de robustez off-distribution**: resistencia de carga, magnitud/frecuencia de
  referencia — todas parten del valor exacto con el que se entrenó la red.
- **Evaluador de dataset propio**: subís un CSV/JSON con el esquema de 15 columnas fijas de la red
  y lo evalúa en open-loop.

## 2. Capa de IA transversal

Plan completo (9 pasos) implementado — ver la nota de estado al final de
`design_ai_layer_transversal.md`.

### 2.1 Clasificadores y regresores config-driven

- Un YAML define arquitectura + tier + dominio; un builder genérico
  (`build_classifier`/`build_forecaster`) lo construye. Agregar una capa es editar un archivo, no
  tocar Python.
- Un solo script de entrenamiento: `experiments/train_model.py --config ... --dataset ...`.
- **Estado real de lo entrenado y promovido** (`configs/registry.yaml`):

  | Dominio | Bloque | Tier PC | Tier Raspberry Pi 5 | Tier ESP32 |
  |---|---|---|---|---|
  | `dc_motor` | Clasificador | 89.5% acc | 86.8% acc (distillado) | 86.8% acc (distillado) |
  | `dc_motor` | Regresor | — no existe — | — | — |
  | `vsc_dpc` | Clasificador | — bloqueado (ver abajo) — | — | — |
  | `vsc_dpc` | Regresor | RMSE 0.06 | RMSE 0.20 (distillado) | RMSE 0.56 (distillado) |

  El clasificador `vsc_dpc` está deliberadamente sin construir: bloqueado por un veredicto de
  separabilidad pendiente (Fase D.2, ver `INSTRUCTIONS.md` Sec. 6 Paso 0). El regresor `dc_motor`
  no tiene necesidad identificada todavía.

### 2.2 Registro de modelos

- `src/driveflow/ai/registry.py`: un manifiesto único (`configs/registry.yaml`) resuelve
  `(domain, tier, block)` → carpeta de la corrida promovida (pesos + config + métricas juntos).
- Promoción (`experiments/promote_run.py`) es un paso deliberado y separado del entrenamiento —
  entrenar no implica quedar en producción.

### 2.3 Despliegue a hardware edge (Raspberry Pi 5 / ESP32)

- El modelo del tier PC ya promovido actúa de "maestro"; se distila (`--distill` en
  `train_model.py`, blend de la etiqueta real con la predicción del maestro) hacia una
  arquitectura más chica por tier.
- **ESP32 tiene una restricción estructural**: el esquema de config prohíbe capas recurrentes
  (`recurrent_type: lstm`/`gru`) — el regresor usa una TCN causal chica en su lugar, el
  clasificador usa bloques DS-CNN (depthwise+pointwise) en vez de Conv1D+SE.
- Export a TFLite (`experiments/export_tflite.py`, corre como `python -m experiments.export_tflite`):
  **float16** para Raspberry Pi 5 (sin necesidad de datos de calibración), **int8** para ESP32
  (necesita datos reales de calibración, no ruido sintético).
- El regresor GRU de Raspberry Pi 5 necesita el Flex delegate (Select TF ops) en tiempo de
  ejecución — no es un `.tflite` 100% builtin, pero es viable en un Raspberry Pi 5 corriendo Linux
  completo.
- Los `.tflite` reales ya generados están descargables directo desde la pestaña IA del dashboard.

### 2.4 Agentes de monitoreo basados en reglas

- `src/driveflow/monitoring/`: un esquema de regla validado (`rules/schema.py`) — condiciones
  booleanas sobre nombres de campo de telemetría, verificadas por sintaxis vía `ast` (nunca
  `eval()`, ni siquiera contra código malicioso).
- **Tier ESP32**: watchdog puro de umbrales duros, sin ML — no tiene un script de agente separado,
  la regla + el esquema alcanzan (ej. la regla de divergencia R∈[1,3]Ω del dominio `vsc_dpc`).
- **Tier Raspberry Pi 5** (`GatewayAgent`): evalúa reglas contra un stream de telemetría con
  histéresis (la condición debe sostenerse N segundos) y debounce (no repite la alerta mientras se
  mantiene activa). Puede operar sin conexión al tier PC.
- **Tier PC** (`ServerAgent`): no evalúa reglas — agrega los `Alert` que ya produjo cada
  `GatewayAgent` en una bitácora + estado "ok"/"alert" por dominio, y rastrea la confianza del
  clasificador en el tiempo para detectar drift y sugerir reentrenamiento.
- **Pendiente**: el badge de estado en el frontend (Sección 5.3 del design doc) no está conectado
  todavía — los agentes existen y están testeados, pero el dashboard no los consulta aún.

## 3. Dashboard (Streamlit)

### 3.1 Página de inicio

- 3 tarjetas de macro-fase (A, B, IA) con hover/entrada animada.
- Diagrama del sistema completo (Plotly, con click sobre cada caja para ver el detalle de ese
  componente — sigue en debugging si el click realmente registra en el navegador).
- Toggle de tema claro/oscuro (`☀️ Light` / `🌙 Dark`), persistido en la sesión.
- Franja tipo osciloscopio animada (canvas JS) debajo del header.
- Texto "About this platform" con el detalle de los 3 sistemas.

### 3.2 Pestaña Fase A

- Sidebar: tipo de falla + combinaciones custom, severidad eléctrica/mecánica, modo de control
  (velocidad/torque), duración, seed, características del motor (editable), ruido.
- Modo "Single simulation" o "Advanced Flow" (secuencia de segmentos).
- 3 sub-tabs: corrientes y vibración, envolvente (ω, i_a), plano i_d–i_q PMSM.
- Historial de configuraciones con navegación atrás/adelante.

### 3.3 Pestaña Fase B

- Sidebar: duración, seed, resistencia de carga, magnitud/frecuencia de referencia.
- 3 sub-tabs: tracking en el tiempo, plano de voltaje complejo, evaluar tu propio dataset.

### 3.4 Pestaña IA

- Selector de dominio (`dc_motor`/`vsc_dpc`).
- **Descarga de modelos edge**: expander siempre visible con los `.tflite` reales disponibles por
  tier/bloque, con su accuracy/RMSE y de qué corrida vienen — aclarado explícitamente que son
  artefactos fijos, no dependen de los parámetros de la corrida de muestra.
- **Fuente de datos**: generar una corrida de muestra (parametrizable: falla/severidad para
  `dc_motor`, resistencia/referencia para `vsc_dpc`, con escape hatch de valor custom en todos los
  sliders) o subir un archivo (con expander de formato requerido + ejemplo descargable real,
  generado de una simulación).
- Paneles de clasificador (clase predicha + confianza) y regresor (forecast de las próximas
  muestras) para el tier **PC únicamente** — los tiers edge no se evalúan en vivo acá, solo se
  descargan.

## 4. Tests

- 380 tests, `pytest`. Incluye tests dirigidos por `streamlit.testing.v1.AppTest` que corren el
  dashboard completo sin navegador.
- CI (GitHub Actions, `.github/workflows/tests.yml`) corre en Python 3.11 y 3.12 contra push/PR a
  `main`.

## 5. Cómo correrlo

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,viz]"

streamlit run src/driveflow/viz/dashboard.py   # dashboard
pytest                                         # suite completa
```

## 6. Pendientes / gaps conocidos

- **CI en GitHub Actions está fallando** en el commit más reciente (`05f4352`) por una causa
  todavía no identificada — se descartó que sea la falta del extra `viz` (ya corregido), pero el
  fallo persiste en el runner real y no se pudo reproducir localmente en dos intentos distintos.
  Pendiente de acceso a los logs reales (`gh auth login`) para diagnosticar.
- Clasificador `vsc_dpc` bloqueado por el veredicto de separabilidad pendiente (Fase D.2).
- Regresor `dc_motor` no existe (sin necesidad identificada todavía).
- El badge de estado de los agentes de monitoreo no está conectado al frontend.
- Los paneles de la pestaña IA solo evalúan el tier PC en vivo — los tiers edge se pueden
  descargar pero no se prueban dentro del dashboard.
- `assets/landing.png` (la captura en el README) es de antes de la capa de IA, el diagrama del
  sistema y el toggle de tema — desactualizada.
