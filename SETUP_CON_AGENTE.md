# SPADE: poner la plataforma en marcha (Windows, Linux o macOS)

Este archivo está pensado para entregárselo a un asistente de código (Claude Code, Codex, Cursor, Copilot u otro) con un mensaje como:

> «Sigue las instrucciones de este archivo para instalar y abrir SPADE en mi computador.»

También lo puede seguir una persona a mano: los comandos son los mismos.

**Resultado esperado:**
- la plataforma SPADE (paquete Python `driveflow`) queda instalada;
- el smoke test pasa 6/6;
- la interfaz web (Streamlit) queda abierta en http://localhost:8501.

No se necesita GPU, CUDA ni datos externos. Todo corre en CPU y los modelos ya vienen en el repositorio.

---

## Instrucciones para el agente

Lee esta sección antes de ejecutar nada.

1. **Sigue los pasos en orden** y verifica la salida de cada uno antes de seguir. No declares éxito sin haber visto `6. no exceptions: PASS` en el paso 5.
2. **No modifiques archivos del repositorio** para hacer que algo funcione. Tampoco hagas commit ni push, ni reentrenes modelos o regeneres datasets. Si algo falla, usa la tabla de problemas (sección 8). Si el problema no aparece ahí, detente y reporta el error exacto.
3. **Pide confirmación al usuario antes de:**
   - instalar software del sistema (Python, Git, Homebrew, paquetes `apt`, `winget`);
   - usar `sudo` o una consola de Administrador;
   - cambiar el registro de Windows o la política de ejecución de PowerShell.
   
   Todo lo demás ocurre dentro de la carpeta del proyecto y de su entorno virtual `.venv`.
4. **Versión de Python:** solo **3.11 o 3.12**, de 64 bits. Con 3.10 o con 3.13 o superior la instalación falla, porque así lo exige `pyproject.toml`. Si la versión del sistema no sirve, instala 3.12 aparte con `uv` (recomendado) sin tocar el Python del sistema.
5. **Mensajes que no son errores:**
   - TensorFlow imprime avisos de CUDA, cuDNN, oneDNN o «GPU will not be used»: son informativos;
   - `RuntimeWarning` y `DeprecationWarning`: son normales.
6. **Al terminar, entrega al usuario el reporte de la sección 9.**

---

## 1. Detectar el sistema

| Sistema | Comando | Qué mirar |
|---|---|---|
| Linux / macOS | `uname -sm` | `Linux x86_64`, `Darwin arm64` (Apple Silicon) o `Darwin x86_64` (Mac Intel) |
| Windows (PowerShell) | `[Environment]::Is64BitOperatingSystem` | Debe decir `True` |

Además, revisa qué Python hay disponible:
- Linux y macOS: `python3 --version`, `python3.12 --version` y `python3.11 --version`;
- Windows: `py -0p`.

En un Mac con Apple Silicon, si `uname -m` dice `x86_64`, la terminal está corriendo bajo Rosetta. En ese caso usa un Python nativo arm64, por ejemplo `/opt/homebrew/bin/python3.12` o el que instala `uv`.

Según el resultado, sigue **solo** la sección de tu sistema: 2A, 2B o 2C.

---

## 2A. Linux (Ubuntu / Debian; probado en Ubuntu 26.04 y en el CI con 24.04)

```bash
# Herramientas del sistema (pedir confirmación: usa sudo)
sudo apt update
sudo apt install -y git curl build-essential

# Código
git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE

# uv trae su propio Python 3.12, sin importar cuál tenga el sistema
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,viz]"
```

Si el sistema ya tiene Python 3.11 o 3.12 y prefieres no usar `uv`, cambia las últimas tres líneas de `uv` por estas:

```bash
sudo apt install -y python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

Continúa en el **paso 5**.

---

## 2B. Windows 10 / 11 (PowerShell; verificado por el CI en `windows-latest`)

Ejecuta todo en **PowerShell**, no en WSL.

```powershell
# Herramientas (pedir confirmación)
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
```

Después de instalar, **cierra y vuelve a abrir PowerShell** y verifica con `py -3.12 --version` y `git --version`.

**Rutas largas (recomendado, una sola vez).** Requiere PowerShell como Administrador y reiniciar después, así que pide confirmación antes. TensorFlow trae rutas muy profundas, y sin este cambio `pip` puede fallar con `WinError 206`.

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Clona el código en una ruta corta:

```powershell
mkdir C:\dev -Force
cd C:\dev
git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

Si `Activate.ps1` falla con *«running scripts is disabled on this system»*, hay dos salidas:
- pide confirmación y ejecuta `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`;
- o no actives el entorno y usa `.\.venv\Scripts\python.exe` en lugar de `python` en todos los comandos siguientes.

En Windows, los comandos de los pasos 5 y 6 usan `\` en las rutas (`scripts\smoke_test.py`). Continúa en el **paso 5**.

---

## 2C. macOS (verificado por el CI en `macos-latest`, Apple Silicon)

```bash
# Herramientas (pedir confirmación)
xcode-select --install          # Git y compiladores; omitir si ya están
# Si no hay Homebrew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 git

git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Ahora instala el paquete. El comando depende de la arquitectura:

| `uname -m` | Comando |
|---|---|
| `arm64` (M1–M4) | `pip install -e ".[dev,viz]"` |
| `x86_64` (Mac Intel) | `pip install -e ".[dev,viz]" -c constraints/macos-intel.txt` |

En un Mac Intel, las restricciones son **obligatorias** y solo sirve Python 3.12, porque Google no publica TensorFlow más nuevo que 2.16.2 para Mac Intel. Sin las restricciones, la instalación termina pero `import` falla.

Continúa en el **paso 5**.

---

## 3. Si no quieres usar Git

Descarga el ZIP desde https://github.com/lucianoradrigan/SPADE/archive/refs/heads/main.zip, descomprímelo y entra a la carpeta `SPADE-main`. Desde ahí, sigue la parte de tu sistema a partir de la creación del entorno (`venv`).

## 4. Tiempo y espacio

- **Disco:** unos 3 GB, casi todo TensorFlow.
- **Descarga e instalación:** entre 3 y 10 minutos, según la conexión.
- **Smoke test:** alrededor de 1 minuto.

---

## 5. Verificar la instalación (obligatorio)

Ejecuta esto con el entorno activado, desde la carpeta `SPADE`:

```bash
python -m pip check
python scripts/smoke_test.py --launch
```

En Windows, el segundo comando es `python scripts\smoke_test.py --launch`.

**Criterio de éxito:**
- `pip check` dice `No broken requirements found.`;
- el smoke test imprime `[PASS]` en cada chequeo;
- la última línea es `6. no exceptions: PASS -- N/N checks passed`;
- el código de salida es `0` (`echo $?` en Linux/macOS, `$LASTEXITCODE` en Windows).

Qué revisa cada chequeo:

| # | Chequeo |
|---|---|
| 1 | Las dependencias se importan (imprime sus versiones). |
| 2 | La app Streamlit arranca. Con `--launch`, el servidor real responde en `/_stcore/health`. |
| 3 | Cargan los módulos de simulación, control, IA y monitoreo. |
| 4 | Corre una simulación corta de cada planta (motor DC con PI y con MPC, VSC con DPC, PMSM con FOC). |
| 5 | Un clasificador registrado predice sobre una ventana simulada. |
| 6 | Ninguna excepción se escapó. |

Si algún chequeo da `[FAIL]`, ve a la sección 8 y **no sigas** hasta resolverlo.

Esta prueba más amplia es opcional y tarda unos 10 minutos:

```bash
python -m pytest -m "not slow" -q
```

---

## 6. Abrir la plataforma

```bash
python -m streamlit run src/driveflow/viz/app.py
```

En Windows: `python -m streamlit run src\driveflow\viz\app.py`.

El navegador se abre en **http://localhost:8501**. El comando queda corriendo; para detenerlo, presiona `Ctrl+C`.

Si el sistema pregunta por el firewall (Windows) o por conexiones entrantes (macOS), basta con permitir redes privadas. En macOS también se puede rechazar sin problema para uso local.

Si es un servidor sin pantalla, agrega `--server.headless true --server.address 0.0.0.0` y abre el puerto 8501.

**Prueba manual rápida:**
1. Página **DC Motor**: presiona **Generate**. Deben aparecer gráficos de corriente, velocidad y vibración.
2. Página **AI Diagnostics**: debe mostrar una predicción marcada como *AI SUGGESTION — REQUIRES VERIFICATION*.
3. Las demás páginas (PMSM, VSC DPC, Monitoring, Model Registry…) deben abrir sin errores.

Para volver a abrirla otro día:
- Linux / macOS: `cd SPADE`, `source .venv/bin/activate` y `python -m streamlit run src/driveflow/viz/app.py`.
- Windows: `cd C:\dev\SPADE`, `.\.venv\Scripts\Activate.ps1` y `python -m streamlit run src\driveflow\viz\app.py`.

---

## 7. Actualizar a la última versión

```bash
git pull
pip install -e ".[dev,viz]"       # en Mac Intel, agrega -c constraints/macos-intel.txt
python scripts/smoke_test.py
```

---

## 8. Problemas conocidos

| Síntoma | Causa | Solución |
|---|---|---|
| `requires a different Python: 3.10…` o `3.13…` / `3.14…` | Versión de Python no soportada | Crea el entorno con 3.12: `uv venv --python 3.12 .venv` (Linux y macOS) o `py -3.12 -m venv .venv` (Windows). |
| `ensurepip is not available` (Linux) | Falta el paquete `venv` | `sudo apt install python3.12-venv`, o usa `uv`. |
| `Could not find a version that satisfies the requirement tensorflow` | Python de 32 bits, o 3.13 o superior | Instala Python 3.12 de 64 bits. |
| `WinError 206` / `No such file or directory: …tensorflow\include…` | Límite de 260 caracteres en las rutas de Windows | Activa `LongPathsEnabled` (sección 2B), reinicia y clona en `C:\dev\SPADE`. |
| `DLL load failed while importing _pywrap_tensorflow_internal` | Falta Visual C++ Redistributable | `winget install Microsoft.VCRedist.2015+.x64` y reabre PowerShell. |
| `running scripts is disabled on this system` | Política de ejecución de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (con confirmación), o usa `.\.venv\Scripts\python.exe`. |
| Se abre la Microsoft Store al escribir `python` | Alias de ejecución de Windows | Usa `py -3.12`, o desactiva los alias de Python en *Configuración → Alias de ejecución de aplicaciones*. |
| `pyarrow requires NumPy 2.0 or newer` (Mac Intel) | Instalado sin restricciones | `pip install -e ".[dev,viz]" -c constraints/macos-intel.txt` |
| `illegal hardware instruction` (Mac) | TensorFlow x86_64 bajo Rosetta | Recrea `.venv` con un Python arm64 nativo (`/opt/homebrew/bin/python3.12`). |
| `Illegal instruction (core dumped)` (Linux) | La CPU no tiene AVX (máquina virtual o equipo muy antiguo) | Usa una máquina o VM con AVX habilitado. |
| `ModuleNotFoundError: streamlit` | Se instaló sin el extra `viz` | `pip install -e ".[dev,viz]"` |
| `streamlit: command not found` | El entorno no está activado | Activa `.venv`, o usa `python -m streamlit …`. |
| Puerto 8501 ocupado | Otra instancia de Streamlit | Agrega `--server.port 8502`. |
| Avisos de CUDA, cuDNN u oneDNN | Mensajes informativos de TensorFlow | Ignóralos. Para ocultarlos: `TF_CPP_MIN_LOG_LEVEL=2` (`$env:TF_CPP_MIN_LOG_LEVEL="2"` en PowerShell). |
| Mac Intel: la página *Transfer Learning* falla al saltar al paso 4 | Streamlit ≤ 1.61, forzado por TensorFlow 2.16 | Recorre los pasos en orden. |

Las guías detalladas de cada sistema están en el repositorio: `docs/INSTALL_UBUNTU.md`, `docs/INSTALL_WINDOWS.md` y `docs/INSTALL_MACOS.md`.

---

## 9. Reporte final (el agente se lo entrega al usuario)

```
Sistema: <SO, versión, arquitectura>
Python del entorno: <python --version>
Ruta del proyecto: <ruta>
pip check: <resultado>
Smoke test: <N/N PASS o lista de FAIL>
Interfaz: <abierta en http://localhost:8501 / no se abrió + motivo>
Cambios al sistema hechos (con confirmación): <p. ej. instalé Python 3.12 con winget, activé LongPathsEnabled>
Problemas pendientes: <ninguno / descripción + error exacto>
Para volver a abrirla: <comando exacto para este sistema>
```

---

## Qué es esto y qué no es

SPADE es una plataforma de **simulación** de accionamientos eléctricos con varios componentes:
- plantas: motor DC, PMSM y convertidor VSC;
- controladores: PI, MPC, FOC y DPC neuronal;
- diagnóstico de fallas con IA;
- monitoreo con alertas.

Los modelos de IA están entrenados **solo con datos simulados**, y sus predicciones se muestran como sugerencias que requieren verificación. Los modelos para Raspberry Pi 5 y ESP32 no se han probado en hardware. El detalle de lo validado y lo pendiente está en `docs/VALIDATION_SUMMARY.md`.
