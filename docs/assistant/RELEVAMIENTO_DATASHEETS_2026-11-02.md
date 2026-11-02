# SPADE · Agente Asistente: relevamiento de datasheets de motores

Versión 1.0 · 2026-11-02 · Documento de planificación para Luciano (en español). Complementa `ESPEC_AGENTE_ASISTENTE_2026-11-02.md` (capacidad C7) y `PLAN_AGENTE_ASISTENTE_2026-11-02.md` (§2.2).

**Objetivo:** conocer la variedad real de datasheets y placas, para que el agente:
- extraiga información de muchos estilos distintos;
- sepa **qué datos sacar según lo que quiere simular el usuario**.

**Método:** se leyeron 13 documentos públicos, 12 hojas técnicas y una guía de placas (lista en §1), con `WebFetch`, que extrae el texto del PDF o de la página. Los chequeos de consistencia de §3 son aritmética propia sobre los valores leídos (comando en §8). **No son mediciones.** Si un chequeo no cuadra, eso significa que el documento es ambiguo o inconsistente; no que el motor esté mal.

---

## 1. Catálogo revisado

| # | Documento | Tipo de motor | Estilo | Qué trae | Qué falta | Rasgo útil para el agente |
|---|---|---|---|---|---|---|
| 1 | [Heinzmann PMG 132](https://www.thierry-lequeu.fr/data/karting/PMG-132-e.pdf) | DC imanes permanentes (disco) | Tabla con 5 variantes de tensión (24–72 V) en columnas + curvas | U, I (S1 y S2 10 min), P, n por variante; R_a = 16 mΩ, L = 0,019 mH, J = 0,025 kg·m² | Kt, Ke, eficiencia numérica (solo curva) | **Es el motor de stock de SPADE** (ver §6). Caso dorado con verdad conocida |
| 2 | [Motenergy ME1513/1514](https://electricmotorsport.com/wp/wp-content/uploads/2026/06/me1514-data-sheet-motenergy.pdf) | DC imanes permanentes con escobillas | Lista plana, sin tablas | P a 72 V, I continua y pico 30 s, Ke = 0,0225 V/rpm, Kt = 0,192 N·m/A | R_a, L_a, J | **Ke y Kt difieren en 12 %** en SI: inconsistencia interna |
| 3 | [Parvalux PM11 110 V](https://www.parvalux.com/product/pm11-110v-3000-rpm-2/datasheet.pdf) | DC imanes permanentes | Tabla de una variante | U, I, n, T, stall, Kt, constante de velocidad, J = 1260 g·cm², 2 polos | L_a | **Etiqueta errónea:** "Thermal resistance (Ra) 17.2 Ohms" es la resistencia de armadura (110 V / 6,4 A = 17,19 Ω) |
| 4 | [Bodine 33A 6435](https://bodine-electric.com/products/dc-motors/33a-series-permanent-magnet-dc-motor/6435) | DC imanes permanentes | Página web, valores pareados "90/130" | KT en oz-in/A, KE en V/krpm, R, L, inercia | **Unidad de la inercia** (no está impresa), duty | Unidades de EE. UU.; pares de valores sin asignar a una tensión |
| 5 | [Kollmorgen/IDC B23](https://www.kollmorgen.com/sites/default/files/public_downloads/B23-Motor_Data-Sheet_en-US_Rev1.pdf) | Servo sin escobillas, 4 polos | Tabla motor + tabla sistema con 2 buses | R a temperatura ambiente **y** a Tmax, L, Kt "phase to phase peak", Kv en Vp-p/kRPM, J, fricción estática y dinámica | Conexión estrella/triángulo, base de R y L | Convención de **pico línea-línea**; oz-in entre corchetes con el SI |
| 6 | [maxon EC 60 flat](https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/171/EC-flat-60-bldc-datasheet.pdf) | BLDC, 7 pares de polos | 6 variantes en columnas (12/24/48 V × 2) | R y L **fase-fase**, Kt, constante de velocidad, J en g·cm², datos térmicos | Ld/Lq, límites del inversor | Estilo de catálogo maxon (muy común) |
| 7 | [TI BLDC17F](https://e2e.ti.com/cfs-file/__key/telligent-evolution-components-attachments/00-171-00-00-00-34-62-40/BLDC17FDatasheetR.pdf) | BLDC, 8 polos | **Variantes en filas** | R y L línea-línea, Kt en oz-in/A, back-EMF en V/kRPM, J en oz-in-s² | Corriente continua, base de Kt y Ke | Tabla transpuesta respecto a maxon |
| 8 | [EMRAX 228 v1.7](https://emrax.com/wp-content/uploads/2025/03/EMRAX-228_datasheet_v1.7.pdf) | PMSM axial, 10 pares de polos | **9 columnas** (3 tensiones × 3 refrigeraciones) en 2 páginas | Ld/Lq por fase, R de fase (= R_LL/2, en nota al pie), Ke en "VRMS/RPM", **flujo PM en V·s**, Kt en N·m/Arms, J, rodamientos 6206/3206 | Convenciones de Ke y del flujo (ver §3) | Uno de los pocos con Ld/Lq; útil para MTPA |
| 9 | [Siemens 1FK7042](https://my.goagilix.com/assets/73/c4/73c408ce-327f-4137-967c-3dad152a8a6a/Siemens_1FK70425AF711UA5_Specification_Sheet.pdf) | PMSM servo, 8 polos | Hoja de producto de un solo artículo | Kt, kE "at 20 °C" en V/1000 min⁻¹, R "at 20 °C", inductancia de campo giratorio, τ eléctrica y mecánica, J en kg·cm², límites | Ld/Lq por separado | Valores a **60 K y 100 K** de calentamiento; τ_el permite verificar R y L |
| 10 | [Yaskawa SGM7J-02D](https://www.yaskawa.eu.com/products/motion-control/productdetail/product/sgm7j-02d7f61_6840.php) | Servo PMSM | Página web de producto | Potencia, par nominal y máximo, velocidad, J (×10⁻⁴ kg·m²) | **Todas las constantes eléctricas**, I, polos | Caso "no alcanza para simular": hay que pedir el manual |
| 11 | [WEG W22 (16616834)](https://www.wolfautomation.com/content/weg-16616834-three-phase-motor-datasheet.pdf) | Inducción, jaula, 6 polos | Paquete de 22 páginas: 4 ratings en columnas + curvas en imagen | U, I, n, deslizamiento, par (ft·lb), I de rotor bloqueado, I en vacío, η y FP a 25/50/75/100 %, J en "sq.ft.lb", **rodamientos 6308 ZZ / 6207 ZZ** | Circuito equivalente (R1, X1, R2, X2, Xm) | Unidades NEMA; las curvas son imágenes |
| 12 | [Baldor GNEM3309T](https://www.baldor.com/api/products/GNEM3309T/infopacket) | Inducción, inverter duty | Placa + tabla de carga 25–150 % + datos de rendimiento | Placa 50 y 60 Hz separadas con "//", **R línea-línea a 25 °C**, inercia en lb-ft², rodamientos DE 6307 / ODE 6206 | Circuito equivalente | "Typical performance, not guaranteed" |
| 13 | [Placa NEMA/IEC (EC&M)](https://www.ecmweb.com/maintenance-repair-operations/motors/article/20896372/understanding-induction-motor-nameplate-information) | Inducción (guía) | Campos de placa | Tensión doble o triple, FLA por tensión, código de rotor bloqueado, letra de diseño, SF, clase de aislación, duty, rodamientos | R, L, J | Lista canónica de campos de placa |

**Fotos de placas.** No encontré un conjunto público de **fotos de placas de motores con transcripción y licencia clara**:
- [nameplate_sample2](https://huggingface.co/datasets/darjja/nameplate_sample2) (MIT) tiene 30 fotos de equipos variados, no de motores;
- [nameplate1](https://huggingface.co/datasets/kahua-ml/nameplate1) (CC-BY-4.0) tiene 1000 imágenes solo con cajas de detección, sin texto.

Las fotos tendrán que salir del laboratorio (decisión D6) o de un generador sintético (§7).

## 2. Estilos de documento

| Estilo | Ejemplos | Dificultad para el agente | Cómo lo maneja |
|---|---|---|---|
| **A. Variantes en columnas** | PMG 132, maxon, EMRAX (9 columnas), WEG (4 ratings) | Asignar cada valor a su columna; al extraer el texto las columnas se desordenan (pasó en PMG 132 y EMRAX) | Leer la página **como imagen** además de como texto. Extraer **todas** las variantes y que el usuario elija la suya |
| **B. Variantes en filas** | BLDC17F | Tabla transpuesta | La misma lógica de variantes, con otra orientación |
| **C. Hoja de un solo artículo** | Siemens 1FK7, Parvalux | Bajo; el riesgo está en las condiciones (20 °C, 60 K / 100 K) | Guardar la condición junto al valor |
| **D. Página web sin constantes** | Yaskawa, Motenergy, Bodine | Faltan R, L o Kt; valores pareados sin asignar | Decir qué falta para simular y pedir el manual o el valor |
| **E. Paquete de catálogo con curvas** | WEG (22 páginas), Baldor | Datos solo en curvas (imágenes); muchas páginas | Usar las tablas numéricas; las curvas solo como "informativo", nunca leer números de una curva sin marcarlo |
| **F. Placa de características (foto)** | Guía NEMA/IEC | Foto borrosa, reflejos, abreviaturas, valores dobles "230/460", "//" | Esquema de placa con calidad de imagen y confianza por campo |

## 3. Convenciones y trampas encontradas

Esto es lo que más afecta la exactitud: **el mismo número significa cosas distintas según el fabricante**.

| Trampa | Ejemplos reales | Qué debe hacer el agente |
|---|---|---|
| **Fase o línea-línea** | maxon y BLDC17F: R y L "phase to phase"; EMRAX: R de fase = R_LL/2 (nota al pie); Baldor: R línea-línea | Guardar la base declarada. Para el modelo dq: R_s = R_LL/2 y L_s ≈ L_LL/2 (estrella) |
| **rms, pico o equivalente DC** | B23: Kt "phase to phase peak", Kv en Vp-p; EMRAX: Ke "VRMS/RPM" | Guardar la forma de onda declarada; si no se declara, queda **desconocida** y se resuelve con un chequeo de consistencia |
| **Unidad de Ke** | V/rpm (Motenergy), V/krpm (Bodine, BLDC17F), V/1000 min⁻¹ (Siemens), Vp-p/kRPM (B23), VRMS/RPM (EMRAX), o la inversa como constante de velocidad en rpm/V (maxon, Parvalux) | Una tabla de conversión en Python, no en el LLM |
| **Temperatura de R y Kt** | Siemens: R y kE a 20 °C, pares a 60 K y 100 K; B23: R a ambiente y a Tmax (+43 %); EMRAX y Baldor: 25 °C | Guardar la temperatura. Usar el valor frío por defecto y decirlo |
| **Unidades de EE. UU.** | oz-in, oz-in/A, oz-in-s², lb-ft, **lb-ft² ("sq.ft.lb", WK²)**, HP | Conversión con factores fijos (§8) |
| **Unidades de inercia** | g·cm² (maxon, Parvalux), kg·cm² (Siemens), ×10⁻⁴ kg·m² (Yaskawa), lb-ft² (WEG, Baldor), oz-in-s² (B23, BLDC17F), **sin unidad** (Bodine) | Inercia sin unidad = **campo inválido**; se pregunta |
| **Ratings múltiples** | WEG: 230/460 V 60 Hz + 380/400/415 V 50 Hz; Baldor: "60 // 50 Hz"; Bodine: "90/130 V" | Extraer cada rating por separado; el usuario elige |
| **Continuo, pico o corto plazo** | PMG 132: 110 A S1 y 200 A S2 10 min; Motenergy: pico 30 s; EMRAX: S2 30 s | Separar `continuous` y `peak` con su duración; el pico es el límite de la simulación |
| **Etiqueta errónea** | Parvalux: "Thermal resistance (Ra) 17.2 Ohms" | El chequeo U/I_stall ≈ R lo detecta; el agente transcribe la etiqueta tal cual y Python marca la incoherencia |
| **Inconsistencia interna** | Motenergy: Ke = 0,0225 V/rpm = 0,215 V·s/rad frente a Kt = 0,192 N·m/A (+12 %) | Se informan ambos y se pregunta cuál usar; nunca se promedia en silencio |
| **Coma decimal** | Yaskawa: "0,64 Nm" en un título | Parser de números con la configuración regional |
| **Datos solo en curvas** | PMG 132 (eficiencia), WEG (par-velocidad) | Marcar `only_in_curve`; no se usan como parámetro |

**Chequeos de consistencia** (aritmética sobre valores impresos; comando en §8):

| Documento | Chequeo | Resultado | Lectura |
|---|---|---|---|
| maxon EC 60 flat (24 V) | Kt frente a 60/(2π·k_n) | 0,0305 frente a 0,0305 | Consistente (convención equivalente DC) |
| Parvalux PM11 | Kt frente a 60/(2π·k_n); R frente a U/I_stall | 0,30 frente a 0,289; 17,2 frente a 17,19 Ω | Consistente; confirma que "Thermal resistance" es R_a |
| Bodine 6435 | Kt frente a Ke en SI | 0,424 frente a 0,420 | Consistente |
| BLDC17F-19 | Kt frente a Ke en SI | 0,0417 frente a 0,0420 | Consistente: Ke es "equivalente DC", aunque no lo declara |
| Motenergy ME1513 | Ke frente a Kt en SI | 0,215 frente a 0,192 | **12 % de diferencia**: preguntar |
| Siemens 1FK7042 | √3·kE (línea-línea rms) frente a Kt; L/R frente a τ_el | 1,47 frente a 1,40 N·m/A; 5,63 frente a 5,60 ms | Consistente (Kt a 100 K, imanes más calientes). R y L tienen la misma base |
| B23 | L/R frente a τ_el impresa | 1,52 frente a 1,769 ms | 16 % de diferencia: R o L con otra base o temperatura. Preguntar |
| EMRAX 228 HV | Kt frente a 1,5·p·ψ (ψ impreso) | 0,98 frente a 0,962 (sin √2) o 1,36 (con √2) | Coincide **sin** √2: Kt parece referido a corriente de pico, aunque dice "Arms" |
| EMRAX 228 HV | ψ impreso frente a ψ desde Ke (como línea-línea rms) | 0,0641 frente a 0,0537 V·s | No coincide con la convención declarada; Ke ≈ ψ·p·ω (pico de fase, ±2,5 %). **Convención a confirmar** |

**Conclusión.** Las constantes de un datasheet no pueden convertirse con una sola regla fija. Python debe:
1. probar las convenciones posibles;
2. quedarse con la que hace consistentes Kt, Ke, R, L y τ;
3. **preguntar al usuario** si ninguna o más de una cuadra.

## 4. Qué extraer según lo que quiere hacer el usuario

**Leyenda:**
- **R** = requerido: sin él no se simula;
- **I** = importante: tiene sugerencia o default, pero cambia el resultado;
- **·** = informativo: se guarda, no se usa hoy.

| Dato del datasheet | DC: control de velocidad o de par | DC: sintonía (modo objetivo) | DC: fallas de rodamiento | PMSM: FOC / MTPA | Inducción (Fase 0b, futuro) | Destino en SPADE |
|---|---|---|---|---|---|---|
| Tipo de motor | R | R | R | R | R | `plant_config_id` |
| Tensión nominal (por rating) | R | R | R | R (bus DC) | R | `dc_supply_v` (Fase 0) / `U_SUPPLY` |
| Corriente nominal continua | R | R | I | R | R | Límite / nominal |
| Corriente pico (y su duración) | I | **R** (techo de la sintonía) | · | R | I | `dc_limit_overrides.i` |
| Velocidad nominal / máxima | R | R | **R** (frecuencias de falla) | R | R | `omega_ref`, límites |
| Potencia y par nominal | I (chequeo) | I | I (100 % de carga) | I | R | `load_level` = fracción de T_n |
| R_a / R_s (base y temperatura) | R | R | I | R | R | `r_a` / `r_s` |
| L_a / L_s, Ld, Lq | **I** (sugerencia por τ_el) | R | · | **R** (Ld ≠ Lq para MTPA) | R | `l_a` / `l_d`, `l_q` |
| Kt / Ke / constante de velocidad / flujo | R (ψ_e) | R | I | R (ψ_p) | — | `psi_e` / `psi_p` |
| Pares de polos | — | — | — | R | R | `p` |
| Inercia del rotor | **I** (sugerencia por clase de potencia) | **R** (sin ella la sintonía no vale) | I | R | R | `j_rotor` |
| Fricción (estática, viscosa) | · | I | · | · | · | Hoy no configurable (carga fija) |
| **Rodamientos DE/NDE** (p. ej. 6308 ZZ) | · | · | **R** | · | · | → tabla de geometría → `fault_order_override` (ver nota) |
| η y FP en puntos de carga | · (chequeo P = T·ω) | · | · | · | R | Chequeo / estimación del circuito |
| I en vacío, I de rotor bloqueado, par máximo | · | · | · | · | R | Estimación del circuito equivalente |
| Clase de aislación, IP, duty, peso | · | · | · | · | · | Solo informativo |

**Nota sobre rodamientos.** Es un vínculo valioso con el diagnóstico. Las placas industriales (WEG, Baldor, EMRAX) traen la **designación** del rodamiento, y SPADE calcula BPFO/BPFI/BSF/FTF a partir de `BearingGeometry(n_elements, element_diameter_m, pitch_diameter_m, contact_angle_rad)` (`sim/vibration/bearing_frequencies.py`). El flujo sería:
1. designación (6308) → tabla de geometría;
2. geometría → orden de falla;
3. orden de falla → `Scenario.fault_order_override`, que ya existe.

La tabla de geometrías por designación **no existe en SPADE** y necesita una fuente con licencia clara: catálogos de fabricantes de rodamientos o mediciones publicadas. Si el rodamiento no está en la tabla, se pregunta. **Nunca se infiere la geometría desde el nombre.**

**Cómo usa el agente esta matriz:**
1. Pregunta (o deduce de la conversación) **qué quiere hacer** el usuario.
2. Toma la columna correspondiente.
3. Solo exige los **R** de esa columna y sugiere los **I**.

Ejemplos:
- Para "simula este motor DC con control de velocidad", **no** hace falta preguntar por los rodamientos.
- Para "simula una falla en la pista externa", los rodamientos son **requeridos**.
- Si el usuario no dice qué quiere, el agente propone la columna más común para ese tipo de motor y lo dice.

## 5. Cambios al diseño del extractor

Estos cambios se derivan del relevamiento y actualizan `PLAN_AGENTE_ASISTENTE` §2.2.

1. **Cada valor numérico lleva su convención**, además del valor y la unidad:

```json
"torque_constant": {
  "value_printed": "0.41", "unit_printed": "N-m/A", "label_printed": "Kt, Phase to Phase Peak",
  "basis": "line_to_line",          // phase | line_to_line | unknown
  "waveform": "peak",               // rms | peak | dc_equivalent | unknown
  "temperature_c": null, "condition": null,     // e.g. "100 K rise", "S2 10 min"
  "variant": "B23 / 220 VAC",       // which column or row
  "source": "page 1, electrical data table", "confidence": "high",
  "only_in_curve": false
}
```

   El LLM **transcribe** la etiqueta y la convención solo si están impresas; si no, pone `unknown`. Python convierte.
2. **Variantes:** el esquema pasa de `ratings[]` a `variants[]`. Cada variante tiene su propio conjunto de valores, y el agente pregunta cuál corresponde al motor del usuario. Si el usuario dice el código de artículo, se elige sola.
3. **Lectura doble de los PDF:** texto **e** imagen de cada página. El texto extraído desordenó columnas en PMG 132, EMRAX y Bodine; la imagen conserva la tabla.
4. **Chequeos de consistencia** (`physics/consistency.py`), con tolerancias pre-registradas:
   - Kt ↔ Ke, probando las convenciones posibles;
   - τ_el = L/R;
   - P = T·ω;
   - R ≈ U/I_stall;
   - T_n ≈ Kt·I_n;
   - deslizamiento ↔ velocidad síncrona (inducción);
   - P_out = √3·U·I·FP·η (inducción).

   El resultado decide la convención o produce **una pregunta concreta** al usuario.
5. **Etiquetas sospechosas:** si una etiqueta no coincide con su unidad o magnitud (como "Thermal resistance" en Ω), se marca y se pregunta. No se corrige en silencio.
6. **"No alcanza para simular":** si después de extraer faltan **R** de la columna elegida (caso Yaskawa web), el agente lo dice y pide el documento técnico o los valores. No completa con defaults sin avisar.

## 6. Hallazgo lateral: el datasheet del motor de stock de SPADE

El motor DC de SPADE declara sus parámetros a partir del Heinzmann PMG 132 (`sim/motors/dc_permanently_excited_motor.py`, comentario en la línea 47).

**Coinciden con el datasheet:** `r_a = 16 mΩ`, `l_a = 19 µH` y `j_rotor = 0,025 kg·m²`.

**No coinciden los nominales:**

| Magnitud | Datasheet PMG 132 (60 V, S1) | Default GEM en SPADE |
|---|---|---|
| Corriente nominal | 110 A | 97 A |
| Par continuo | 20 N·m (impreso); 19,9 N·m desde P/ω | 16 N·m |
| Velocidad | 2870 min⁻¹ = 300,5 rad/s | 300 rad/s |
| ψ_e | 0,181 V·s (desde T/I) a 0,194 V·s (desde (U − R·I)/ω) | 0,165 V·s |

SPADE valida la fidelidad del port de GEM, no la representación del motor real, así que esto **no es un defecto de SPADE**. Aun así, conviene documentarlo como observación en `docs/SIMULATION_STABILITY_REPORT.md`, por ejemplo como S-20 informativo, porque el agente va a usar este datasheet como **caso dorado número 1**. Que el agente reproduzca los valores del datasheet y no los de GEM es justamente el comportamiento correcto.

## 7. Conjunto de evaluación (Fase 2): propuesta

Los datasheets tienen derechos de autor, así que **no se commitean**. Se trata igual que `Data4train.mat`:
- **`tests/assistant/golden/manifest.yaml`:** URL, SHA-256, fabricante y estilo (A–F), con un `expected.json` escrito a mano **antes** de correr el agente.
- **`experiments/fetch_golden_datasheets.py`:** descarga a `data/assistant_golden/` (ignorado por git). Las pruebas que lo necesitan se saltan si falta el archivo, como las de FMU.
- **Generador sintético de datasheets (nuevo, sin problemas de licencia).** Toma parámetros conocidos (incluidos los de los motores de stock de SPADE) y genera PDFs e imágenes en los 6 estilos de §2, combinando las trampas de §3: línea-línea frente a fase, rms frente a pico, unidades de EE. UU., variantes en columnas, etiquetas ambiguas, fotos con ruido o rotación. Tiene dos ventajas:
  - la verdad es exacta;
  - sí se puede commitear y correr en CI con un `ReplayClient`.
- **Fotos reales de placas:** las del laboratorio (D6), con su transcripción escrita a mano.

**Métricas** (los umbrales los fijas tú antes de medir, D7):
- exactitud por campo y por convención;
- tasa de invención (objetivo 0);
- detección de inconsistencias (Motenergy, B23, Parvalux, EMRAX);
- variante correcta;
- "pide lo que falta" en el estilo D.

## 8. Reproducibilidad

- **Lectura de los documentos:** `WebFetch` sobre las URLs de §1, el 2026-11-02. El texto extraído puede desordenar columnas; los valores marcados como "unclear" por la extracción no se usaron en los chequeos.
- **Chequeos de §3 y §6:**

```python
import math
rpm = 2 * math.pi / 60
# PMG 132 @ 60 V
w = 2870 * rpm; T = 5970 / w; psi_T = T / 110; psi_E = (60 - 110 * 0.016) / w      # 300.5 rad/s, 19.86 Nm, 0.181, 0.194
# Ke from speed constant / V-per-rpm, SI
0.0225 / rpm            # Motenergy Ke = 0.215 V*s/rad  vs Kt 0.192
1 / (313 * rpm)         # maxon k_n 313 rpm/V -> 0.0305
1 / (33 * rpm)          # Parvalux -> 0.289 ; 110/6.4 = 17.19 ohm
5.9 * 0.0070616, 4.4 / (1000 * rpm)    # BLDC17F 0.0417 Nm/A, 0.0420 V*s/rad
60 * 0.0070616, 44 / (1000 * rpm)      # Bodine 0.424, 0.420
ke = 89 / (1000 * rpm); math.sqrt(3) * ke, 29e-3 / 5.15     # 1FK7: 1.472 Nm/A, 5.63 ms
16.1e-3 / 10.6                                               # B23: 1.52 ms vs 1.769 printed
1.5 * 10 * 0.06411, 0.06881 * math.sqrt(2 / 3) / (10 * rpm)  # EMRAX: 0.962 ; 0.0537 V*s
```

- **Factores de conversión** (para la tabla de Python): 1 oz-in = 0,0070616 N·m; 1 lb-ft = 1,35582 N·m; 1 lb-ft² (WK²) = 0,0421401 kg·m²; 1 oz-in-s² = 0,0070616 kg·m²; 1 g·cm² = 10⁻⁷ kg·m²; 1 kg·cm² = 10⁻⁴ kg·m²; 1 hp = 745,7 W.

## 9. Impacto en las fases

| Fase | Cambio por este relevamiento |
|---|---|
| 0 | `describe_plant()` declara, para cada campo, su unidad SI **y** su convención (p. ej. `r_s`: fase, Ω, 20 °C). El extractor convierte hacia eso |
| 1 | La matriz de §4 se convierte en `configs/assistant_requirements.yaml` (columna × campo → R / I / ·); `draft.status()` la usa para saber qué preguntar según el objetivo |
| 2 | Esquema con convenciones y variantes (§5), lectura doble del PDF, `consistency.py`, generador sintético, conjunto dorado de §7, tabla de geometrías de rodamientos (con fuente) |
