# Patch 5: corrección de alcance de la Macro-fase B — DPC no es comparable con PI/MPC

**Estado:** decidido y documentado. Extraído de `docs/macro_fase_B2_dpc_deployment.md` (sección
"Limitación honesta sobre el alcance de B") como documento propio porque `INSTRUCTIONS.md` (v2)
lo referencia como una corrección de principio, no solo un detalle de implementación.

## Formulación original (INSTRUCTIONS.md v1)

> **DPC es un controlador más**, intercambiable con PI/MPC clásicos ya presentes en GEM.
>
> Integrar como controlador adicional (`controller_type="DPC"`) en `datagen/runner.py`, para
> ampliar el dataset de la Macro-fase A con corridas bajo control DPC y poder comparar DPC vs. PI
> vs. MPC bajo las mismas condiciones.

Esta formulación asume que DPC, PI y MPC son tres controladores intercambiables sobre la misma
planta (el motor DC de la Macro-fase A), difiriendo solo en la ley de control.

## Hallazgo

`DPC4PowerElectronics` (el toolbox original portado en B.1/B.2) no controla un motor: controla un
**Voltage Source Converter (VSC)** — un filtro LCL alimentando una carga resistiva, sin parte
mecánica ni rotor. Su loss basada en modelo (`Adf`/`Bdf`, ver `control/dpc/loss.py`) y su
generador de referencia (un vector rotando a 50Hz/50V, `control/dpc/reference.py`) son específicos
de ese dominio eléctrico de potencia, no de control de motores.

PI (Macro-fase A) y cualquier MPC futuro contra el motor DC operan sobre variables físicas
distintas (velocidad angular, torque, corriente de armadura de un motor con carga mecánica) sin
punto de comparación común con las variables de DPC (tensión/corriente de un convertidor sin masa
rotante). No hay una "misma condición" bajo la cual comparar ambos: no comparten planta, ni
espacio de estados, ni objetivo de control.

## Decisión

- Se retira el objetivo de "comparar DPC vs. PI vs. MPC bajo las mismas condiciones" tal como
  estaba formulado — es una comparación que no tiene sentido físico con el toolbox tal como existe.
- DPC se integra en `datagen/runner.py` como un **segundo camino de escenario independiente**
  (`plant_config_id="vsc_dpc_v1"`, dispatch separado del motor DC en `_run_vsc_scenario` —
  ver `docs/macro_fase_B2_dpc_deployment.md`), no como una tercera rama dentro del loop del motor.
- El esquema de datos (Parquet) sí es común entre dominios (columnas del VSC en NaN para registros
  del motor DC y viceversa), permitiendo exportar ambos tipos de corrida en el mismo dataset — pero
  eso es *compartir formato de almacenamiento*, no *comparabilidad de métricas de desempeño*.
- La Macro-fase B no aporta datos de entrada a la Macro-fase C (los clasificadores de
  `paper_federative` consumen vibración + MCSA de rodamientos, ninguno de los cuales existe en el
  dominio VSC). Ver `docs/patch7_fase_D_dpc.md` para el uso real de los datos generados por B.

## Ver también

- `docs/macro_fase_B1_dpc.md` — port a Keras de la red y la loss.
- `docs/macro_fase_B2_dpc_deployment.md` — despliegue en lazo cerrado, checkpoints v1→v2→v3,
  integración en `runner.py`.
- `docs/patch7_fase_D_dpc.md` — qué se hace con los datos de la Macro-fase B en vez de compararlos
  contra PI/MPC.
