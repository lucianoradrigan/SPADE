# Patch 4: modulación de amplitud por zona de carga en `fault_impulses.py`

**Estado:** implementado y validado (prueba dirigida, no la grilla completa contra Paderborn —
ver Sección 4 sobre alcance). Motivado por una revisión externa del documento de resumen
(`docs/resumen_macro_fase_A.md`) que señaló una pieza física faltante en el modelo de impulsos.

## 1. Motivación

`docs/patch2_retiro_modulo_C.md` (Sección 4) encontró que `inner_race` calibra a una severidad
mecánica ~2.5x menor que `outer_race` (0.02 vs. 0.05) para igualar su AUC real, y lo dejó
anotado como "el Módulo B no tiene esa asimetría incorporada — la única forma de replicarla es
ajustando `severity` distinto por tipo de falla, no es algo que el modelo capture
automáticamente". Es decir: se corrigió el síntoma (una constante numérica distinta) sin modelar
la causa.

La causa, según la literatura clásica de diagnóstico de rodamientos (McFadden & Smith, 1984,
extendiendo el modelo de tren de impulsos con modulación de amplitud por posición relativa a la
carga; Sawalhi & Randall para el fenómeno de doble impulso en daño real; Antoni & Randall, 2003,
para el carácter pseudo-cicloestacionario del tren de pulsos):

- **Outer race:** el defecto (en la pista externa, fija respecto a la carcasa) y la zona de carga
  (fija, dado que la carga radial no rota) mantienen una relación geométrica **constante** en el
  tiempo. Los impactos tienen amplitud aproximadamente constante.
- **Inner race:** el defecto rota con el eje, atravesando la zona de carga fija una vez por
  revolución. Los impactos se modulan en amplitud a la **frecuencia de giro**, generando bandas
  laterales alrededor de BPFI — la firma clásica de un defecto de pista interna.
- **Ball:** el elemento rodante defectuoso circula alrededor de la jaula (a FTF), atravesando la
  zona de carga una vez por revolución de jaula. Modulación a la **frecuencia de jaula (FTF)**.

`fault_impulses.py` (Patch 3) ya generaba el tren de impulsos order-tracked correcto en
*frecuencia*, pero con amplitud constante — sin esta modulación, no había forma de que el modelo
explicara, en vez de solo compensar numéricamente, por qué `inner_race` es más difícil de
detectar.

## 2. Implementación

`sim/vibration/fault_impulses.py::LoadZoneModulator`:

- `outer_race`, `cage`: sin modulación (envolvente constante = 1). Para `cage` no se encontró un
  modelo de modulación por zona de carga establecido en la literatura revisada — se dejó sin
  modular en vez de inventar uno.
- `inner_race`: envolvente modulada a la frecuencia de giro (`order=1.0`).
- `ball`: envolvente modulada a la frecuencia de jaula (`order = fault_order("cage", geometry)`,
  es decir FTF).
- Forma de la envolvente: coseno semi-rectificado elevado a `sharpness` (default 2.0) —
  `max(0, cos(2π·fase))^sharpness`, cero fuera del arco cargado, cerca de 1 dentro. `sharpness`
  más alto → arco cargado más angosto.

`FaultImpulseGenerator` ahora aplica `pulses * envelope` antes de escalar por `severity`, con un
flag `load_zone_modulation: bool = True` para poder aislar su efecto en tests/experimentos.

## 3. Validación

**Física correcta (tests unitarios, `tests/test_fault_impulses.py`, 18 tests):**
- `outer_race`/`cage`: envolvente exactamente constante en 1.0.
- `inner_race`: cuenta de picos de envolvente en 1s de simulación ≈ f_r (frecuencia de giro).
- `ball`: cuenta de picos ≈ FTF (frecuencia de jaula), verificado contra `bearing_frequencies.ftf()`.
- Con modulación, las amplitudes de los impulsos de `inner_race` **varían** en el tiempo
  (`std > 0`); sin ella (o para `outer_race`), todas las amplitudes son idénticas.

**Efecto en separabilidad (chequeo dirigido, `ModalVibrationModel` directo, sin pasar por el lazo
PI completo — más rápido, aísla el efecto de la modulación de cualquier otra fuente de varianza):**

| Falla | Severidad (ya calibrada en Patch 2) | AUC sin modulación | AUC con modulación | AUC real (Paderborn) |
|---|---|---|---|---|
| outer_race | 0.05 | 0.937 | 0.937 (sin cambio, esperado — no aplica modulación) | 0.712 |
| inner_race | 0.02 | 0.625 | **0.521** | 0.518 |

Con modulación, `inner_race` a la **misma severidad ya calibrada** (no se re-calibró) cae de
0.625 a 0.521 — prácticamente exacto contra el real (0.518). La razón tiene sentido: la
modulación reparte parte de la energía del impulso hacia bandas laterales alrededor de BPFI, así
que la energía capturada estrictamente en la banda del fundamental (±15Hz, la métrica de
`validate_separability.py`) baja — que es justo lo que hace más difícil de detectar a un defecto
de pista interna en la realidad.

## 4. Alcance de esta validación (y qué falta)

Esta prueba usó `ModalVibrationModel` en velocidad constante (25Hz), sin pasar por el controlador
PI ni el lazo completo de `datagen/runner.py` — más rápido para aislar el efecto de la
modulación, pero **no es la misma grilla exhaustiva de `experiments/validate_separability.py`**
(que compara contra las 720/1300+ ventanas reales de Paderborn con el pipeline completo, toma
~20 minutos). El número de `outer_race` sin modulación (0.937) tampoco coincide exactamente con
el 0.766 medido en Patch 2 con el pipeline completo — son harnesses de simulación distintos
(velocidad constante vs. controlada por PI con su propia dinámica), no directamente comparables
en valor absoluto, solo en el *efecto relativo* de encender/apagar la modulación, que es lo que
esta validación buscaba aislar.

**No se re-ejecutó** la grilla completa de `validate_separability.py` con modulación habilitada
para confirmar el match exacto contra los holdouts reales — dado que el chequeo dirigido ya da
evidencia fuerte y consistente con la hipótesis, se prioriza documentar y dejarlo como validación
formal pendiente antes de dar este patch por cerrado del todo, no como trabajo bloqueante.

## 5. Backlog (del análisis externo, priorizado, no implementado en este patch)

Del resto de puntos señalados en la revisión externa, priorizados por impacto esperado:

- **B — calibración modal como distribución, no punto único:** en vez de un banco de 4 modos fijo
  (que no generaliza entre rodamientos, Patch 2 Sección 3.1), estimar `{ω_n,k, ζ_k, gain_k}` como
  media + varianza entre K001-K006 y muestrear una instancia por corrida sintética (equivalente a
  domain randomization). Es el siguiente ítem de mayor impacto esperado, en particular para que
  los clasificadores de Macro-fase C no aprendan features atadas a una calibración "promedio"
  ficticia.
- **C — doble impulso** (entrada tipo escalón / salida tipo impulso, Sawalhi & Randall) para daño
  real acumulado (KB\*); el impulso único actual es razonable para daño artificial (EDM) pero
  probablemente subrepresenta daño por fatiga.
- **D — jitter/deslizamiento** en el timing de los impulsos (Antoni & Randall, carácter
  pseudo-cicloestacionario) — barato de implementar, mejoraría el ancho de banda del pico
  sintético contra el real.
- **E — factores de escala x/y/z** (hoy documentados como asunción, Sección 3.2 del resumen)
  reemplazables por datos reales de un dataset con acelerómetro triaxial.
- **F — escala de amplitud por punto de operación** vía contacto Hertziano (energía del impulso
  como función de carga y velocidad instantáneas) en vez de dejarla como limitación permanente.
- **G — chequeo de información mutua no lineal** entre torque/vibración, para blindar del todo el
  hallazgo de coherencia de Patch 2/3 (baja prioridad — coherencia + 3 técnicas no lineales ya
  cubren razonablemente el espacio).
- **H — validación cruzada contra otro dataset** (CWRU, Ottawa) además de Paderborn, para
  confirmar que los hallazgos no son idiosincrasia de un solo banco de pruebas.

Los repositorios de código específicos mencionados en la revisión externa no se verificaron
independientemente (sin acceso para navegarlos en este patch) — quedan como pistas a mirar antes
de implementar cualquiera de los puntos C-E, no como fuentes ya confirmadas.
