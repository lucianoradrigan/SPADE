**SPADE for Windows 10/11 (64-bit): double-click installer.** No git, Python or administrator rights needed.

1. Download `SPADE-Setup-<version>-<commit>.exe` below and double-click it.
2. Windows may show "Windows protected your PC" (the installer is not code-signed): click **More info → Run anyway**.
3. Follow the wizard. SPADE installs for your user in `%LOCALAPPDATA%\SPADE` (about 2 GB) and adds a **SPADE** shortcut to the Start menu (and the desktop, if ticked).
4. Open **SPADE**. A console window appears and the browser opens at http://localhost:8501 after a few seconds. Keep that window open while you use SPADE; close it to stop SPADE. If Windows Defender Firewall asks about `python.exe`, allowing private networks is enough.

Included: SPADE (the commit this release was built from), Python, every dependency (TensorFlow, Streamlit, fmpy, …) and the OpenModelica FMUs for Windows (release `fmu-v2`). Your work (workspace, datasets, runs) is saved in `%LOCALAPPDATA%\SPADE\SPADE`. Uninstall from Settings → Apps → SPADE; it removes the program and keeps your workspace.

Built and checked by `.github/workflows/windows-installer.yml`: the installer is installed silently on a GitHub Windows runner, then the smoke test, the `SPADE/OmBuck-v0` FMU environment and the app (port 8501) are checked. **Not yet tried on a colleague's clean Windows machine.**

---

**SPADE para Windows 10/11 (64 bits): instalador con doble clic.** No hace falta git, Python ni permisos de administrador.

1. Descarga `SPADE-Setup-<versión>-<commit>.exe` (abajo) y haz doble clic.
2. Si Windows muestra "Windows protegió su PC" (el instalador no está firmado): **Más información → Ejecutar de todas formas**.
3. Sigue el asistente. SPADE se instala para tu usuario en `%LOCALAPPDATA%\SPADE` (unos 2 GB) y crea el acceso directo **SPADE** en el menú Inicio (y en el escritorio, si lo marcas).
4. Abre **SPADE**. Aparece una ventana de consola y el navegador se abre en http://localhost:8501 en unos segundos. Deja esa ventana abierta mientras usas SPADE; ciérrala para detenerlo.
