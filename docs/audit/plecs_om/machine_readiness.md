# Machine readiness (round PLECS/OM, §0b) — 2026-10-22

| Requirement | Command | Result | Action |
|---|---|---|---|
| `CLAUDE.md` at the repo root | `ls CLAUDE.md` | **missing** (neither in SPADE-dev nor in SPADE) | Reported; work continues without it |
| GUI spec | `ls docs/design/SPADE_GUI_Complete_Specification.md` | present (§3 at line 150, §3.4 at 204, §3.8 at 527, §3.15 at 717) | — |
| Docker usable without sudo | `docker --version` | **not installed** (`docker: command not found`) | `sudo -n true` → "interactive authentication is required": Claude cannot install it. The user must run once: `sudo apt-get update && sudo apt-get install -y docker.io && sudo usermod -aG docker $USER`. Claude then uses `sg docker -c …`, with no re-login needed. Until then **route b (B0–B8) is blocked**; A1, A2 and X1–X3 continue |
| Alternatives without sudo | `which omc podman apptainer conda` | none installed | The official OpenModelica installer for Ubuntu is an apt repository and also needs sudo |
| Disk space | `df -h ~` | 74 GB free of 468 GB (84 % used) | Above the 15 GB threshold |
| GitHub CLI | `gh auth status` | logged in as lucianoradrigan (keyring) | B7/B8 can use it |
| Python of `.venv` | `.venv/bin/python --version` | 3.12.13 | OK |
| OS / arch | `/etc/os-release`, `uname -m` | Ubuntu 26.04.1 LTS, x86_64 | — |
