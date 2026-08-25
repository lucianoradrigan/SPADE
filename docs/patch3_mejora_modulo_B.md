# Patch 3: desacoplar ruido de fondo y excitación de falla en Módulo B

**Estado:** decidido y aplicado. Continúa la cadena de investigación de
`docs/patch2_retiro_modulo_C.md` (retiro del Módulo C): con el corrector residual descartado, el
problema pasa a ser qué excita el filtro modal del Módulo B — el diseño original
(`docs/addendum_vibracion_v1.md` Sec. 3) usaba rizado de torque + tren de impulsos de falla. Este
patch verifica si el torque tiene sustento real como excitador y, si no, lo saca.

## Paso 0 — ¿hay acoplamiento torque→vibración específicamente en las frecuencias de falla?

El hallazgo de Patch 2 (coherencia torque↔vibración ≈0.022) se midió en banda ancha (20-1800Hz)
sobre rodamientos **sanos**. No descarta acoplamiento puntual en un rodamiento **con falla real**,
en sus frecuencias características (BPFO/BPFI/BSF/FTF) — el mismo mecanismo que ya usa el camino
eléctrico (MCSA).

**Método:** coherencia torque↔vibración (`scipy.signal.coherence`, `nperseg=4096`) en ventanas
±2Hz alrededor de cada frecuencia característica, sobre rodamientos con daño artificial de un
solo tipo: KA01-KA30 (pista externa, confirmado contra ficha técnica: `Component=OR`) y
KI01-KI21 (pista interna, confirmado: `Component=IR`). Los KB\* (daño real por fatiga) se
excluyeron: su ficha técnica (KB23.pdf) muestra daño simultáneo en IR y OR con múltiples puntos
"random" — no dan una frecuencia característica limpia para aislar.

**Resultado:**

| | En su propia frecuencia de falla | En frecuencias que NO son su falla (control) |
|---|---|---|
| KA\* (defecto en OR) | BPFO: media=0.398 | BPFI: 0.340, BSF: **0.416** (más alta que su propio BPFO) |
| KI\* (defecto en IR) | BPFI: media=0.315 | BPFO: 0.390, BSF: **0.453** (más alta que su propio BPFI) |

La coherencia sube muy por encima del baseline de rodamientos sanos (0.022) — pero **no es
específica a la frecuencia del defecto real**: un rodamiento con daño solo en pista externa (KA\*)
muestra tanta o más coherencia en BSF (bolas, sin dañar) que en BPFO (su defecto real). Sube
pareja en toda la banda baja (9.6-123Hz, armónicos bajos de ~25Hz de giro), no selectivamente en
la frecuencia de la falla real.

**Conclusión parcial:** hay señal de "hay falla" en el torque, pero no de "qué tipo de falla" —
un modulador de amplitud condicionado a `fault_type` específico no tiene sustento.

## Paso 3b — ¿esa misma banda ya sale alta en rodamientos sanos?

Dato que faltaba para cerrar la interpretación: si la banda 9.6-123Hz ya sale alta en sanos, es
un artefacto genérico de armónicos de giro. Si sale baja, hay señal genuina de "presencia de
falla".

**Método:** mismo cálculo, banda fija 9.6-123Hz, sobre las mismas 12 corridas sanas (K001-K006)
del test de coherencia original.

**Resultado:**

```
Sanos (9.6-123Hz):        media = 0.168   (consistente entre las 12 corridas, rango 0.144-0.183)
Fallados (misma banda):   media = 0.315-0.453
Baseline banda ancha sanos (original, Patch 2): 0.022
```

Ni "mismo orden que falla" ni "cercano al baseline original" — resultado intermedio pero
**consistente** entre archivos (no es el caso de "inconsistencia entre sanos" del árbol de
decisión original). Interpretación combinada:

1. Hay un **artefacto genérico de armónicos de giro**, presente sin falla (0.168 ≫ 0.022).
2. Hay un **incremento adicional real cuando hay falla** (~+0.15 a +0.28 sobre el piso sano) — sí
   es señal de "presencia de falla" genuina, no solo el artefacto de giro.

## Decisión de diseño

Ninguna de las dos lecturas cambia la conclusión para `force_synthesis.py` (ahora
`background_noise.py` + `fault_impulses.py`):

- El componente genérico de giro estaría presente **incluso en corridas sanas** — no aporta nada
  usar torque para modelarlo; un generador estocástico calibrado por RMS ya lo cubre sin
  necesitar torque como entrada.
- El componente de "presencia de falla" ya se descartó como no-específico por tipo (control
  cruzado BSF > BPFO en KA\*, Paso 0) — no sirve para condicionar qué tren de impulsos excitar.

**Torque sale de la excitación del Módulo B por completo.**

## Arquitectura final

```
sim/vibration/
├── background_noise.py   # BackgroundNoiseGenerator: ruido blanco, ganancia RMS-calibrada
│                          # contra sanos (calibration.py::fit_background_noise_gain).
│                          # NO depende de torque/corriente/velocidad/carga.
├── fault_impulses.py      # ImpulseTrainGenerator (compartido con el camino eléctrico,
│                          # datagen/fault_injection.py) + FaultImpulseGenerator (wrapper con
│                          # geometría + fault_type + severity). Depende solo de omega
│                          # (cinemático) y severity (parámetro de escenario).
├── modal_model.py          # ModalVibrationModel: excitacion = background.step() + fault.step();
│                          # el MISMO filtro modal calibrado (Paso 2) da forma a ambas.
└── vibration_synthesizer.py  # sin Módulo C (ver patch2_retiro_modulo_C.md);
                             # vibration_source siempre "synthetic_b".
```

`VibrationSynthesizer.step(omega)` — ya no recibe `i_d`, `i_q` ni `torque` (interfaz simplificada,
coherente con que ninguno de esos tres tenía sustento como entrada).

## Paso 2 — calibración modal solo contra tramos sanos

`sim/vibration/calibration.py::calibrate_module_b()` y `fit_background_noise_gain()` ahora
**exigen** (no solo documentan) que las corridas de ajuste sean de `HEALTHY_BEARING_CODES`
(K001-K006) — `_assert_healthy_only()` lanza `ValueError` si se cuela cualquier otro código. Esto
aísla la respuesta propia de la estructura mecánica de la energía de falla. El mismo filtro
calibrado da forma tanto al ruido de fondo como al tren de impulsos de falla, porque ambas
excitaciones pasan por la misma estructura física antes de llegar al sensor.

`configs/vibration_module_b.yaml` ahora incluye `background_noise_gain` (calibrado por RMS
contra el pool de sanos, ajustado en `experiments/calibrate_module_b.py`).

## Test de regresión

`tests/test_patch3_regression.py`:
- `severity=0` (con o sin `fault_type` puesto) → sin picos en BPFO/BPFI/BSF/FTF, banda por banda
  comparada contra el piso de ruido general.
- Con falla → energía en la banda de la frecuencia correcta sube >3x contra el caso sano
  (comparación de banda, no pico global crudo — las resonancias estructurales calibradas dominan
  la amplitud cruda, igual que se documentó para el criterio de cierre de A.5).
- Amplitud del pico de banda crece monótonamente con `severity`.

## Backlog (diferido, no implementado)

- Calibrar la curva severidad→amplitud contra las 32 condiciones de rodamiento de Paderborn (6
  sanas + 12 daño artificial + 14 fatiga real), en vez de un número fijado a mano.
- Verificar corrimiento de frecuencia por deslizamiento de jaula (BPFO/BPFI reales vs. teóricos)
  con análisis de envolvente sobre rodamientos con falla real.
- Feature auxiliar para el clasificador de `paper_federative` (Macro-fase C, no Módulo B): el
  **incremento** de coherencia torque-vibración sobre el piso sano (no la coherencia cruda, que ya
  está alta en sanos) como indicador binario de falla vs. sano en la banda de giro.

## Ver también

- `docs/patch2_retiro_modulo_C.md` — retiro del corrector residual (Módulo C).
- `envelope_diag_final.html` (artifact de la sesión, no versionado) — diagnóstico de coherencia
  envolvente↔velocidad/carga que motivó revisar mecanismos alternativos antes de este patch.
- `docs/patch4_modulacion_zona_carga.md` — agrega modulación de amplitud a `fault_impulses.py`
  (los impulsos que este patch dejó sin torque, ahora con amplitud dependiente de la posición del
  defecto respecto a la zona de carga).
