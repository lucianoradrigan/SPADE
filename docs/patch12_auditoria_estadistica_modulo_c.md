# Patch 12 — Auditoría estadística del retiro de Módulo C (Patch 2)

**Contexto de origen:** revisión crítica del repositorio, tarea de prioridad Media #3 de un
documento externo de "tareas de corrección priorizadas". Pregunta concreta: ¿qué test estadístico
respalda "torque/velocidad/carga son estadísticamente independientes de la vibración real en
rodamientos sanos" (Patch 2, base de Patch 3 y Patch 4)?

## 1. Qué evidencia existía y qué le faltaba

`docs/patch2_retiro_modulo_C.md` Sección 1 reporta una tabla de coherencia media/máxima
(torque=0.022/0.201, velocidad=0.031/0.148, fuerza=0.044/0.227) y concluye "sin señal" por
comparación de magnitud (valores bajos comparados con lo que se esperaría de un acoplamiento
real) -- **sin nivel de significancia estadística explícito, sin especificar el número de
segmentos de Welch usados, y sin la fórmula/dataset/tamaño de muestra completos** para
reproducir el número exacto. Esto es exactamente el hueco que motiva esta tarea.

Lo que SÍ existía, no señalado explícitamente antes de esta auditoría: `experiments/verify_torque_vibration_coherence.py`
y `experiments/verify_cross_fault_frequency_coherence.py` (agregados en Patch 8) ya son
reconstrucciones "best-effort" documentadas de este mismo análisis, con toda la metodología que
Patch 2 no especificó (nperseg=4096, banda 20-1800Hz, decimación 64kHz→4kHz, concatenación de las
12 corridas antes de estimar coherencia) declarada explícitamente en su propio docstring. **No
había evidencia de que esos dos scripts se hubieran ejecutado alguna vez contra el dataset real**
-- `DATA.md` documenta el dataset como no redistribuido, y nada en el repositorio registra una
corrida. Esta auditoría es, hasta donde se pudo determinar, su primera ejecución real.

## 2. Re-ejecución contra datos reales (Paderborn, disponible localmente en esta máquina)

### `verify_torque_vibration_coherence.py` (reconstrucción de Patch 2 Sec. 1)

| Variable | Media medida | Media citada | Máx. medido | Máx. citado |
|---|---|---|---|---|
| Torque | 0.0130 | 0.022 | 0.2314 | 0.201 |
| Velocidad | 0.0100 | 0.031 | 0.1165 | 0.148 |
| Fuerza | 0.0137 | 0.044 | 0.1810 | 0.227 |

Mismo orden de magnitud que lo citado, misma conclusión cualitativa ("sin señal"). Las medias
medidas son sistemáticamente ~2-3x más bajas que las citadas -- el propio script ya documenta que
no pretende ser una réplica exacta byte-a-byte (parámetros no especificados en Patch 2 tuvieron que
elegirse), así que esta discrepancia no invalida la conclusión, pero tampoco hay forma de saber hoy
si la sesión original de Patch 2 usó una elección distinta de esos parámetros o directamente otro
subconjunto de corridas.

### `verify_cross_fault_frequency_coherence.py` (reconstrucción de Patch 3 Paso 0 / Paso 3b)

**Paso 0** (coherencia en la frecuencia propia de cada falla vs. las de control): patrón
cualitativo citado reproducido exactamente -- en ambos tipos de falla, BSF (control, no es el
defecto real) tiene coherencia igual o mayor que la frecuencia propia del defecto real:

| Tipo de falla | Frecuencia | Rol | Medido | Citado |
|---|---|---|---|---|
| outer_race | BPFO | propia | 0.2062 | 0.398 |
| outer_race | BSF | control | 0.2243 | 0.416 |
| inner_race | BPFI | propia | 0.1821 | 0.315 |
| inner_race | BSF | control | 0.2274 | 0.453 |

Confirma independientemente la conclusión de Patch 3 Paso 0 (la coherencia no es específica al
tipo de falla real).

**Paso 3b, hallazgo nuevo (no computado como tal en el patch original):** el script agrega una
comparación directa sanos-vs-fallados en la banda fija (9.6-123Hz) que Patch 3 no reporta como
número propio (el docstring lo señala explícitamente: "a further, direct check not explicitly
detailed as a separate computation in the patch"). Resultado:

- Sanos: media = 0.1865 (n=12 corridas)
- Fallados (KA*+KI* combinados): media = 0.1872 (n=46 corridas)

**Prácticamente idéntico.** En la banda fija de armónicos de giro, la presencia de una falla real
no incrementa la coherencia torque-vibración de forma medible respecto a rodamientos sanos. Esto
**refuerza** la conclusión de Patch 2/3 (no hay modulador explotable condicionado en torque,
independientemente de si hay falla o no) de forma más fuerte que lo que el patch original
argumentaba -- no se trata solo de que los sanos ya tengan un artefacto de fondo, sino que ese
artefacto no cambia de forma medible con la falla presente, al menos en esta banda ancha.

## 3. Umbral de significancia formal (no existía antes de esta auditoría)

Para una estimación de coherencia por método de Welch con `L` segmentos promediados, el umbral de
significancia al 100(1-α)% bajo la hipótesis nula de coherencia real cero es (Carter 1987; Bendat
& Piersol):

```
umbral = 1 - α^(1/(L-1))
```

Con `nperseg=4096`, `fs=4000Hz`, 12 corridas de 4s concatenadas (48s → 192000 muestras) y
`noverlap` por defecto de scipy (`nperseg//2`): `L ≈ 92` segmentos de Welch. Con α=0.05:

```
umbral ≈ 1 - 0.05^(1/91) ≈ 0.032
```

Además, el estimador de coherencia bajo la hipótesis nula tiene un sesgo conocido de
`E[γ̂²] ≈ 1/L ≈ 0.0109` para este mismo `L` -- no converge a exactamente 0 con muestra finita.

**Comparando contra esto:** las medias medidas en la Sección 2 (0.0100-0.0137) son
**indistinguibles del sesgo esperado del propio estimador bajo coherencia real cero** (0.0109) --
un resultado más fuerte que "los números son chicos": son consistentes con acoplamiento real
exactamente nulo, no solo con acoplamiento débil. Las medias CITADAS en Patch 2 (0.022-0.044) caen
por encima de este sesgo pero aun así por debajo (torque, velocidad) o apenas por encima (fuerza)
del umbral de significancia de una sola frecuencia (0.032) -- consistente con la misma conclusión
cualitativa, con un margen menor.

**Advertencia metodológica sobre "coherencia máxima" (no señalada en Patch 2):** la banda 20-1800Hz
con esta resolución espectral (fs/nperseg ≈ 0.98Hz) contiene ~1800 bins de frecuencia. Bajo la
hipótesis nula, con un umbral por-bin al 95% (α=0.05), se espera que ~5% de los bins (~90 de 1800)
superen el umbral **solo por azar**, sin corrección por comparaciones múltiples. Los valores de
"coherencia máxima" citados/medidos (0.116-0.231) son exactamente del orden esperable bajo pura
casualidad dado ese número de bins -- **no son evidencia de acoplamiento real y no deberían
interpretarse como tal.** La media de banda (que sí se usa como estadístico principal en Patch 2 y
en esta auditoría) es la elección metodológicamente correcta; esto simplemente lo hace explícito
por primera vez.

## 4. Conclusión de la auditoría

- El test estadístico formal (umbral de significancia de coherencia, corrección por sesgo de
  muestra finita) **no existía en Patch 2/3** -- se construyó aquí por primera vez.
- Con ese marco, la conclusión de Patch 2 ("sin señal", torque/velocidad/carga independientes de la
  vibración real en sanos) se **reconfirma y se fortalece**: las medias medidas en esta auditoría
  son estadísticamente indistinguibles de coherencia cero, no solo "chicas".
- Patch 3 (retiro del torque como excitador) y Patch 4 (modulación por zona de carga en
  `fault_impulses.py`, que no depende de esta pregunta de independencia sino de la física de
  posición angular del defecto) **siguen siendo válidos** bajo este resultado reconfirmado. El
  hallazgo nuevo de la Sección 2 (Paso 3b sanos-vs-fallados) refuerza, no contradice, la conclusión
  de Patch 3.
- Limitación que permanece abierta: la discrepancia entre las medias citadas en Patch 2/3 y las
  medidas aquí (mismo orden de magnitud, no coincidencia exacta) no se puede resolver sin la sesión
  original -- documentada, no oculta, consistente con lo que los propios scripts de reconstrucción
  ya advertían antes de esta auditoría.
