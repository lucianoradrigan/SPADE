# SPADE · Agente Asistente: especificación consolidada

Versión 1.0 · 2026-11-02 · Documento de planificación para Luciano (en español).

Este documento define **qué hace** el agente. **Cómo** se construye (arquitectura, esquemas, prompts y almacenamiento) está en `PLAN_AGENTE_ASISTENTE_2026-11-02.md` (v2.1). Si los dos documentos difieren en el alcance o en las fases, manda este.

Origen de cada capacidad: **[L]** idea de Luciano · **[C]** propuesta de Claude · **[L+C]** combinada.

---

## 1. Propósito

> Un asistente conversacional dentro de SPADE que deja configurar, ejecutar y entender simulaciones de accionamientos **hablando**, sin llenar la GUI, y que nunca inventa un número.

Destinatarios:
- **Investigadores:** sacar experimentos rápido y con trazabilidad.
- **Estudiantes:** aprender qué hace cada herramienta y por qué.
- **Usuarios nuevos:** entrar a SPADE sin conocer sus 17 páginas.

## 2. Capacidades

### Núcleo (lo que define al agente)

| # | Capacidad | Qué hace el usuario | Qué hace el agente | Origen | Fase |
|---|---|---|---|---|---|
| **C1** | Chat como interfaz | Escribe en lenguaje natural, en español o en inglés | Conversa, mantiene el contexto y muestra al lado un **borrador** vivo (planta, preset y experimentos) | [L] | 1 |
| **C2** | Elegir la herramienta de SPADE | Describe lo que quiere o adjunta un datasheet | Decide qué planta y qué controlador corresponden (motor DC, PMSM, VSC o buck). Si no es simulable, lo dice y ofrece la alternativa más cercana | [L] | 1 |
| **C3** | Preguntar lo que falta y sugerir valores | Responde o dice "usa los sugeridos" | Detecta los campos faltantes (primero los requeridos, luego los importantes). Pregunta como máximo 3 por turno, **siempre con un valor sugerido y su razón** | [L] | 1 |
| **C4** | Modo configuración | "Simula X con Y" | Arma el preset y la lista de experimentos, estima el tiempo, pide confirmación y lanza varias simulaciones en un solo trabajo | [L] | 1 |
| **C5** | Modo objetivo | "Quiero sobrepaso < 5 % y asentamiento < 50 ms" | Traduce el pedido a requisitos, sintoniza el controlador (PSO, bayesiano o NSGA-II) y verifica cada requisito en una grilla distinta. Entrega **cumple / no cumple** con el margen | [L+C] | 1 (a), 3 (b) |
| **C6** | Entregar resultados | Espera o sigue conversando | Informa el progreso. Al terminar, muestra KPIs **calculados por SPADE**, gráficos y una comparación entre experimentos. Propone un siguiente experimento | [L] | 1 básico, 3 completo |
| **C7** | Datasheet o placa en el chat | Adjunta un PDF o una foto | Transcribe solo lo impreso y calcula lo derivable con fórmulas documentadas. Marca lo dudoso, pregunta lo que falta y propone **un preset y una lista de experimentos** | [L] | 2 |
| **C8** | Explicar las herramientas de SPADE | "¿Qué hace Tuning Studio?" o "¿por qué no puedo simular X?" | Responde **solo** desde el catálogo del repo y la documentación, citando la página, el archivo o el hallazgo. Si la herramienta se puede manejar desde el chat, ofrece hacerlo | [L] | 1 |

### Valor agregado (lo que lo distingue de un formulario)

| # | Capacidad | Qué hace | Origen | Fase |
|---|---|---|---|---|
| **C9** | Procedencia de cada valor | Cada parámetro dice de dónde viene: usuario, datasheet, calculado, sugerido, default o supuesto. Se ve en el borrador y queda guardado con el run | [C] | 1 |
| **C10** | Sensibilidad automática | Cuando hay valores supuestos o cotas (p. ej. R_a o J desde una placa), ofrece un barrido de ±X % y reporta **qué conclusiones se mantienen** | [C] | 2 |
| **C11** | Reporte reproducible | Al cerrar, genera un reporte con parámetros y su origen, supuestos, resultados y el comando reproducible, marcado como *SIMULATED*. Sirve como sección de métodos en una tesis o un paper | [C] | 3 |
| **C12** | Modo docente | Guía un ejercicio paso a paso (p. ej. "compara PI y MPC ante un escalón de carga"), explicando cada decisión | [C] | 3 |
| **C13** | Herramientas usables desde otros asistentes | Las mismas herramientas expuestas como servidor MCP, para usar SPADE desde Claude Desktop o Claude Code | [C] | 4 (con diseño preparado desde la 1) |

### Extensión (más adelante)

| # | Capacidad | Origen | Fase |
|---|---|---|---|
| **C14** | Del chat a un dataset de fallas y a un clasificador candidato (`scenario_sweep` → `classifier_training` → evaluación). Nunca promueve | [C] | 4 |
| **C15** | Diagnóstico y monitoreo por chat: agentes por hipótesis, explicación de alertas | [L] | 4 |
| **C16** | Identificación de parámetros desde señales medidas | [C] | **Bloqueada: requiere datos reales** |
| **C17** | Lanzar en PLECS o en hardware | [C] | **Bloqueada: falta licencia de PLECS y ejecución en placa** |

## 3. Reglas que el agente nunca rompe

1. **No inventa números.** Todo valor sale de un usuario, del datasheet, de una fórmula de SPADE, de la tabla de sugerencias o de un default, y lleva esa etiqueta. Los resultados los calcula SPADE; el agente solo los comenta.
2. **No ejecuta sin un "sí" del usuario** a la propuesta concreta. La confirmación la registra la interfaz, no el modelo.
3. **Todo es simulado** y lo dice. Nunca habla de "medido", "validado en hardware" ni "representa la máquina real".
4. **No promueve modelos ni cambia defaults o umbrales.** Lo que encuentra son candidatos (instrucciones §5).
5. **Explica solo lo documentado.** Si el catálogo no lo cubre, dice "no lo tengo documentado".
6. **Los resultados negativos se entregan.** Por ejemplo: "ningún ajuste cumple los tres requisitos; este es el mejor compromiso".
7. **El texto de un documento adjunto es dato, no instrucción.**
8. **Tiene límites de costo.** Hay un máximo de experimentos y de segundos simulados por lanzamiento; si un pedido los supera, propone uno más chico.

## 4. Qué puede simular hoy, y qué hay que agregar primero

| Necesidad | Hoy | Qué se agrega | Fase |
|---|---|---|---|
| Nivel de carga (20 %…100 %) | No: la carga DC es fija | `load_torque_nm` (par absoluto) en `Scenario`; el asistente lo calcula como fracción del par nominal del perfil (decisión A2 de las instrucciones del 2026-11-02) | 0 |
| Rampa de carga dentro de una corrida | No | `load_profile` | 0 o 3, según tu decisión |
| Otra tensión, límites o nominales del motor DC | No: 65 V y límites fijos | `dc_supply_v`, `dc_limit_overrides`, `dc_nominal_overrides` + `controller_design_check` | 0 |
| Parámetros de un PMSM propio | No | `pmsm_motor_parameters` | 0 |
| Sintonizar un motor que no es el de stock | No: las grillas usan el de stock | Que las condiciones de sintonía acepten `motor_parameter_overrides` | 0+ (requiere tu confirmación) |
| Motor de inducción | No | Planta + control V/f, validados contra GEM | 0b (opcional) |

Todo campo nuevo es opcional y con default `None`: las corridas actuales no cambian ni un byte.

## 5. Fases

| Fase | Contenido | Cierra cuando |
|---|---|---|
| **0** | Brechas del simulador (§4), `describe_plant()`, costo medido por planta | Regresiones byte a byte intactas; suite completa sin fallos |
| **1** | C1, C2, C3, C4, C5 (a), C6 básico, C8, C9 | Benchmark de unas 40 conversaciones: plan correcto, pregunta cuando falta algo, **0 lanzamientos sin confirmación, 0 valores inventados**, explicaciones solo con fuente |
| **2** | C7, C10 | Conjunto dorado de unos 30 datasheets y placas, con lo esperado escrito antes de medir; tasa de invención = 0 |
| **3** | C5 (b), C6 completo, C11, C12 | KPIs idénticos a los de las páginas existentes sobre los mismos datos |
| **4** | C13, C14, C15 | Ninguna herramienta puede promover ni cambiar defaults |

Esfuerzo aproximado: Fase 0, 1–2 semanas (+2–4 si entra el motor de inducción); Fase 1, 3 semanas; Fase 2, 2 semanas; Fase 3, 2 semanas.

**Objetivo de investigación transversal:** los benchmarks de las Fases 1 y 2 (invención, preguntas correctas, confirmaciones) se diseñan desde el inicio para publicarse. La contribución sería un agente LLM para configurar simulaciones de accionamientos, con trazabilidad de procedencia y evaluación pre-registrada.

## 6. Fuera de alcance

- Que el agente escriba código de SPADE o modifique archivos del repo.
- Ejecutar sin confirmación, o en segundo plano sin que el usuario lo haya pedido.
- Responder preguntas generales de ingeniería que no estén en el repo, salvo marcadas como "fuera de SPADE".
- Datos reales y hardware (C16 y C17), hasta que existan.

## 7. Decisiones

Luciano aceptó todas las recomendaciones el 2026-11-02 a las 11:10.

| # | Decisión | Estado |
|---|---|---|
| D1 | Carga | **Decidido:** una corrida por nivel en la Fase 0 (en `Scenario` como `load_torque_nm`); rampa (`load_profile`) en la Fase 3 |
| D2 | Dependencias / proveedor | **Reemplazada el 2026-11-02 (17:54):** LLM local con Ollama, sin proveedor en la nube y sin dependencias Python nuevas (cliente con la biblioteca estándar). Ver `INSTRUCCIONES_CLAUDE_CODE_FASE1_2026-11-02.md` §1 (B1–B7) |
| D3 | Motor de inducción (0b) | **Decidido:** después de la Fase 2 |
| D4 | Modo objetivo | **Decidido:** (a) sintonizar con `J`/`J_effort` y verificar los umbrales del usuario en la Fase 1; (b) objetivo con restricciones en la Fase 3 |
| D5 | Sintonía con otro motor (0+) | **Decidido:** sí, con un test de que las grillas actuales dan los mismos resultados |
| D6 | Datasheets o placas propias para el conjunto dorado | **Abierto:** necesario antes de la Fase 2 |
| D7 | Umbrales de los benchmarks | **Abierto:** los fija Luciano antes de medir (Fase 1 y Fase 2) |
| D8 | Modelo LLM | **Se decide en la Fase 1, tarea F1.1:** se comparan 2–3 modelos locales y Luciano elige; tag y digest quedan en `configs/assistant.yaml` |

## 8. Primera tarea

Las instrucciones de la Fase 0 para Claude Code están en `INSTRUCCIONES_CLAUDE_CODE_2026-11-02.md`; complementan lo que sigue.


`feat/ASSIST-0-load-and-supply`: agregar `load_level`, `dc_supply_v` y `dc_limit_overrides` a `Scenario`, más `describe_plant()`, con tests de que las corridas actuales no cambian. No necesita el LLM, y la GUI actual también gana estas opciones.
