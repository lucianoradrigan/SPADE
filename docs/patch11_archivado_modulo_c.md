# Patch 11 — Archivado del código muerto de "Módulo C"

**Contexto de origen:** revisión crítica del repositorio, tarea de prioridad Baja-Media #6 de un
documento externo de "tareas de corrección priorizadas".

## Qué se hizo

`experiments/train_module_c.py` y `experiments/train_module_c_envelope.py` -- marcados
`HISTORICAL / NOT RUNNABLE` desde su creación (Módulo C, el corrector residual de vibración, fue
retirado por `docs/patch2_retiro_modulo_C.md`) -- se movieron a `experiments/_archive/` (`git mv`,
historial preservado).

**Por qué:** `INSTRUCTIONS.md` ya documentaba explícitamente ("trampa de nomenclatura") que
"Módulo C" (corrector de vibración retirado) se confunde fácilmente con "Fase C" (clasificadores de
diagnóstico, la fase activa del roadmap) por el nombre parecido. Mantener los dos scripts junto a
los scripts activos de `experiments/` (que sí se ejecutan como parte del flujo normal) hacía más
probable que una sesión futura de Claude Code -- o una persona -- los tomara como referencia
ejecutable sin leer el encabezado `HISTORICAL / NOT RUNNABLE` completo.

## Qué se actualizó

- `src/driveflow/sim/vibration/calibration.py`: única referencia de código a la ruta anterior,
  actualizada a `experiments/_archive/...` con una nota de por qué se movieron.
- Cabecera de ambos scripts: nota agregada señalando el archivado y la razón.
- No se encontraron otras referencias (docs, tests) a la ruta anterior.

## Qué NO se hizo

No se borraron los scripts (la opción que el documento de correcciones dejaba como alternativa).
Se conservan como registro de la investigación que fundamentó el retiro de Módulo C -- el propio
`docs/patch2_retiro_modulo_C.md` los cita como evidencia primaria (el hallazgo "0.0% de mejora de
RMSE" reportado ahí proviene de estos dos scripts), así que borrarlos rompería esa trazabilidad sin
necesidad real.
