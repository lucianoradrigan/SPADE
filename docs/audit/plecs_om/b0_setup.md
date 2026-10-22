# B0 — OpenModelica and fmpy set-up (round PLECS/OM)

| Item | Version | Command |
|---|---|---|
| Docker | 29.1.3 (Ubuntu package `docker.io`) | `docker --version`. Installed by the user with sudo; user added to group `docker` plus an ACL on `/var/run/docker.sock` for the running session |
| OpenModelica | 1.27.1, image `openmodelica/openmodelica:v1.27.1-minimal`, digest `sha256:ac9a0223e69b8055d444b42f58ccb9680342b56848c68b81385bf83d7c0ba824` | `docker pull …`; `scripts/fmu/om.sh <script.mos>`; `getVersion()` → "OpenModelica 1.27.1" |
| Modelica Standard Library | 4.1.0+maint.om | `installPackage(Modelica, "4.1.0+maint.om", exactMatch=true)` into `data/fmu/openmodelica_home` (77 MB, git-ignored); `getVersion(Modelica)` → "4.1.0" |
| fmpy | 0.3.32 (BSD-2) | `uv pip install -e ".[dev,viz,fmu]"` in `SPADE-dev/.venv`. It pulled attrs, cmake, jinja2, lark, lxml, msgpack and nbformat |

- **Pinning:** `scripts/fmu/OPENMODELICA_VERSION` (image, digest, OM version, MSL version). `scripts/fmu/om.sh` runs `omc` from that exact digest, with the repository mounted at `/work`, and installs the pinned MSL the first time.
- **`pyproject.toml`:** new extra `fmu = ["fmpy"]` (D3) and pytest marker `fmu`. The extra `[plecs]` mentioned in the decision records of 2026-10-21 (`DECISIONES_EXPANSION_2026-10-21.md` D7, `TODO_EXPANSION_CLAUDE_CODE.md`, `PENDIENTES_EXPANSION_CLAUDE_CODE.md`) was never added to `pyproject.toml`, and no E2 code comment refers to it. Those records are historical and were left as they are; D3 of this round supersedes them.
- **Licences:** see `DATA.md`, section "OpenModelica toolchain…".
  - The OpenModelica runtime header offers BSD New / AGPL v3 / OSMC-PL 1.8 at the recipient's choice.
  - Redistribution of binary FMUs is possible under BSD New with notices.
  - **Still to check before B8:** the third-party components inside the actual built FMUs.
