# CHANGELOG_IMPLEMENTATION — Auditoría de IA + refactorización UX/UI de SPADE

Rama `feature/ai-audit-gui-refactor`, base `dev` @ `eb5d878` (idéntica al snapshot `reference/SPADE-main_original.zip`, verificado con `diff -r`; sólo difieren pesos/datos no versionados).
Entorno: Python 3.11.15 · TensorFlow 2.21.0 · Keras 3.15.1 · Streamlit 1.62.0 · Plotly 7.0.0 — fijado en `requirements-lock.txt`.
Respuestas Q01–Q30: [`AUDIT_ANSWERS.md`](AUDIT_ANSWERS.md).

## 0. Proceso (H0 → H3)

1. **H0 línea base.** Suite original: **469 passed, 0 failed** (`docs/audit/baseline_pytest_summary.txt`). Script de reproducción ejecutado sobre el código sin modificar: **7/7 hallazgos reproducidos** (AI-01…AI-06, AI-08) — `docs/audit/repro_h0_before_fix.py` y `…_output.txt`. Checksums del paquete verificados (27/27 OK).
2. **H1 integridad científica y gates** (backend, sin Streamlit) + 42 tests de regresión.
3. **H2 frontend** multipágina nuevo, conectado al backend corregido.
4. **H3 calidad:** suite completa, capturas reales del app en ejecución, documentación.

## 1. Archivos por hallazgo

| Hallazgo | Archivos | Cambio |
|---|---|---|
| AI-01 mezcla por filas | `ai/transfer/data_merger.py` (reescrito), `models/common/acquisitions.py` (nuevo), `models/common/windowing.py` | Mezcla por adquisición completa; `mix_ratio` en ventanas; prefijos `sim::`/`ext::`; exclusión (no reordenamiento) de grupos no monótonos o con fs distinta; ventanas que nunca cruzan huecos (idénticas a las históricas en datos limpios). `stratify_by_fault` también por grupos. |
| AI-02 pérdida de metadatos | `ai/transfer/dataset_validation.py` (nuevo), `viz/dpc_upload_validation.py` (`validate_inference_upload`) | Contrato de entrenamiento separado del de inferencia; metadatos preservados; labels/canales/fs validados; cotas por canal; hash del dataset. |
| AI-03 val == test | `models/common/splits.py` (`grouped_train_val_test_split`, `SplitManifest`, `grouped_split(strict=)`), `ai/transfer/workflow.py` (nuevo) | Split ternario por grupo, sin fallback con fuga, manifiesto con IDs hasheados. |
| AI-04 base mutada | `ai/transfer/pipeline.py` (`clone_for_finetuning`, `verify_base_unchanged`), `ai/provenance.py` (`weights_fingerprint`) | Siempre se entrena un clon; huella verificada; la UI no cachea el modelo entrenable. |
| AI-05 validación de regresores | `ai/transfer/validator.py` (reescrito, API legacy conservada) | Checks por tarea con PASS/FAIL/NOT_APPLICABLE/INCOMPLETE; regresión: finitud, salida no constante, vs base, vs persistencia, RMSE por horizonte; clasificación: macro-F1, recall por clase, confusión, **vs predictor de clase mayoritaria**; convergencia incluida. |
| AI-06 promoción sin gate | `ai/promotion.py` (nuevo), `ai/registry.py` | `PromotionGate` fail-closed; candidatos no activos; `registry.promote()` re-evalúa la evidencia persistida y rechaza TL no aprobados; escritura atómica; `registry_history.yaml`; `rollback()`; override sólo con motivo+autor, auditado. |
| AI-07 generalización vibración | `experiments/report_vibration_generalization.py` (nuevo), `viz/adapters/simulation_adapter.py` (`VIBRATION_PROVENANCE`) | Informe reproducible desde el YAML existente (0/6 folds mejor que el nulo); etiqueta `synthetic_approximate` en UI y exportaciones. |
| AI-08 alertas concurrentes | `monitoring/agents/agent_server.py` | Alertas activas por `(domain, rule, source)`; `domain_health()` (máx. severidad, UNKNOWN, OFFLINE); `acknowledge()` ≠ resolver; `domain_status()` legacy ahora deriva de todas las activas. |
| AI-09 aislamiento de dominios / PMSM | `viz/views/pmsm.py`, tests | PMSM etiquetado "CONTROL DEMO — NO DIAGNOSIS"; validadores por dominio. |
| AI-10 edge | `ai/deployment_status.py` (nuevo) | Ciclo `exported → … → production_approved`; cuantización leída del propio `.tflite`; sin evidencia no hay "hardware verified". |
| AI-11 reproducibilidad | `ai/provenance.py`, `experiments/train_model.py`, `requirements-lock.txt` | Semilla Python/NumPy/TF; `run_metadata.json` (commit, dirty, versiones, hash del dataset); lock del entorno verificado. |
| AI-12 cargas externas | `dataset_validation.py`, vistas con upload | Límite 50 MB y 2 M filas; parseo sólo como datos (CSV/JSON, sin pickle); cotas por canal en entrenamiento; HTML escapado en todos los componentes. |
| Motor semántico | `monitoring/motor_status.py` (nuevo), `viz/components/motor_status.py`, `viz/assets/motor_base.webp` | `MotorViewModel` + `derive_motor_view` desde el ServerAgent; imagen base estable con overlay CSS. |

**Frontend** (todo nuevo salvo shims): `viz/app.py`, `viz/dashboard.py` (entrada legacy, 2115 → 13 líneas), `viz/pages/*.py` (10), `viz/views/*.py` (11), `viz/components/*.py` (7), `viz/theme/{tokens,css,plotly}.py`, `viz/adapters/*.py` (6), `analysis/spectrum.py`, `.streamlit/config.toml` (temas nativos claro/oscuro). `ai_dashboard.py`, `transfer_learning_dashboard.py`, `live_monitoring_dashboard.py` quedan como **shims de compatibilidad**; los helpers legacy defectuosos (`_prepare_windows`, `_save_finetuned_artifacts`) se eliminaron en lugar de dejarlos como código muerto.

**Tests:** nuevos `tests/test_ai_audit_regressions.py` (T-AI-01…09 + AI-07 + baseline mayoritario), `tests/test_ui_contract.py` (T-UI-01…05, sincronía de tokens, espectro). Adaptados a la nueva navegación conservando su intención: `test_dashboard_landing.py`, `test_ai_dashboard.py`, `test_live_monitoring_dashboard.py`, `test_transfer_learning_dashboard.py`, `test_transfer_learning/test_data_merger.py`, `test_pipeline.py`.

## 2. Contratos cambiados intencionalmente (con migración)

| Contrato | Antes | Ahora | Compatibilidad |
|---|---|---|---|
| `DataMerger.merge()` | filas muestreadas, tamaño = len(sim) | adquisiciones completas | firma igual + kwargs opcionales; 5 tests antiguos describían el bug y se reescribieron |
| `TransferLearningPipeline.from_loaded_model()` | entrenaba el objeto recibido | entrena un clon | `clone=False` restaura el comportamiento antiguo |
| `ModelValidator.validate()` (legacy) | regresor ⇒ `mode_collapse_ok=True` | `predictions_finite_ok`, `non_constant_output_ok` | misma firma |
| `registry.promote()` | aceptaba cualquier run | rechaza runs TL sin gate aprobado (`PromotionRejected`) | runs de `train_model.py` sin cambio; formato de `registry.yaml` sin cambio |
| `ServerAgent.domain_status()` | estado del último evento | "alert" si hay cualquier activa | nuevo `domain_health()` para UI |
| UI | landing + 5 fases, `selected_phase` | `st.navigation` multipágina | `streamlit run src/driveflow/viz/dashboard.py` sigue funcionando; claves `ia_*`, `lm_*`, `tl_*`, `dpc_*` conservadas |
| Tema | toggle propio + CSS sobre clases internas de BaseWeb | temas nativos `[theme.dark]`/`[theme.light]`; CSS sólo para clases `spade-*` | los 3 tests del toggle se retiraron (la función la cubre el menú nativo) |

## 3. Resultados de pruebas

- Línea base: **469 passed** (7 min 05 s).
- Tras los cambios: **536 passed, 0 failed** (8 min 09 s) — `docs/audit/final_pytest_output.txt`. 42 tests de auditoría y 25 de contrato UI nuevos (67 en total); 3 tests del toggle de tema retirados (sustituido por temas nativos).
- **No automatizado:** T-UI-06 (responsive): verificado manualmente con capturas a 1920×1080 y 1366×768 (`docs/screenshots/1366_*.png`), sin recorte de botones/ejes observado. No hay tests con hardware ni con datos privados (no existen en el repo).

## 4. Matriz antes → después (sólo experimentos ejecutados)

| Medida | Antes | Después | Fuente |
|---|---|---|---|
| Ventanas con timestamps no monótonos tras merge (caso H0) | 398 pasos no crecientes | 0 | repro H0 / `TestAI01` |
| Metadatos tras validar upload de entrenamiento | `label, source_file, timestamp_s` perdidos | preservados | repro H0 / `TestAI02` |
| Regresor que devuelve NaN / constante | PASS | FAIL | repro H0 / `TestAI05` |
| Promoción de run TL con checks FAILED | aceptada | rechazada, registry intacto | repro H0 / `TestAI06` |
| Estado con A,B activas y A resuelta | "ok" | WARNING/CRITICAL | repro H0 / `TestAI08` |
| Pesos base tras fine-tuning | modificados | huella idéntica | repro H0 / `TestAI04` |
| Clasificador promovido sobre corridas nuevas (seeds 100–108) | registrado: acc 0.895 (n=38) | medido: 0.222 (0.15 s) / 0.579 (1 s) / 0.638 (3 s) | `docs/audit/base_classifier_duration_check.txt` — hallazgo, no corrección |
| Métricas de modelos tras TL, latencia/RAM edge, precisión tras cuantizar | — | *not evaluated* | — |

## 5. Comparación con los mockups (`figures/gui/concept_v2/`)

| Mockup | Presente | Diferencias deliberadas / pendientes |
|---|---|---|
| 01 Overview | KPIs, flujo de trabajo, quick launch, runs recientes, miniatura de motor, diagrama del sistema | Sin "Engine ready", "98.4 % data integrity" ni gráficos de ejemplo: se sustituyen por checks reales; runs recientes = log de la sesión; motor UNKNOWN hasta monitorear. Sin selector de idioma ni avatar (no hay usuarios/i18n). |
| 02 DC motor | Controles 25 % / resultados 75 %, KPIs, pestañas Time signals / Spectrum / AI diagnosis / Envelope / Parameters, run metadata + export | El "Diagnosis: Review 78 %" pasa a "AI suggestion" con p sin calibrar y "requires verification"; la condición inyectada se muestra como entrada de simulación. Controlador PI/MPC añadido (existía en backend). Sin `st.form`: el patrón "Custom value" necesita reruns; la simulación sólo corre con el botón. |
| 03 TL workbench | Wizard de 4 pasos, linaje, barra de split, checks de integridad, curvas, gate con PASS/PENDING/BLOCKED | Todos los estados se calculan; sin curva de "test loss" por época (el test se evalúa una sola vez, al final, por diseño). |
| 04 Live monitoring | KPIs, consenso por agente, historial, motor | Título y badges "SIMULATED MONITORING" en vez de "Live"; sin gráfico continuo de calidad de señal (no hay stream). Acknowledge implementado; "Resolve" manual no se ofrece (la resolución viene de la fuente). |
| 05 Model registry | Tabla filtrable, detalle con pestañas, artefactos con tamaño y hash, despliegue | Sólo runs y formatos existentes (`.weights.h5`, `.tflite`, json/yaml); sin precisión/recall inventados; "Hardware verified" 0. |
| Motor | Imagen estable `motor_nominal_cyan` + anillo de estado, halo, rotor proporcional a rpm, reduced-motion, aria-label | No se alternan los renders healthy/warning/critical (geometría distinta y color incrustado); no se usan los WebP animados. Sin hotspots por región (no hay sensores por región). |

## 6. Elementos todavía no operativos

- Diagnóstico de **PMSM** (sin modelo de falla, dataset, clasificador ni registro).
- **Ingesta de hardware / telemetría conectada** (API `record_heartbeat` lista; sin fuente).
- **Despliegue físico** en Raspberry Pi 5 / ESP32 y paridad numérica de `.tflite`.
- Calibración de probabilidades, abstención OOD, IC entre seeds/máquinas.
- Resampleo con antialias para datos externos a otra frecuencia.
- `docs/plataforma_estado_actual.md` describe la estructura anterior del dashboard; no se reescribió (el README y este documento describen la actual).

## 7. Requiere confirmación del investigador

Ver la tabla final de `AUDIT_ANSWERS.md` (umbrales de aceptación, métrica primaria, tipo de split, unidad de mezcla, autoridad de promoción/override, cotas físicas, umbrales del detector, tema por defecto, motor 3D).
