# T0 — disk and Ollama (round B, 2026-11-09)

Decision R4 (Luciano, 2026-11-09). Raw values: `t0_disk.json`.

## Disk

| | Before | After |
|---|---|---|
| `/` (system, 468 GB) | 8.6 GB free | **41 GB free** |
| `/mnt/datos` (916 GB) | 568 GB free | 544 GB free |

## What moved (nothing was deleted)

| From | To | Files / bytes (verified identical after the move) |
|---|---|---|
| `SPADE/data`: all SPADE datasets and evaluation caches. `SPADE-dev/data` links to it | `/mnt/datos/spade/data` | 4737 / 6,895,866,272 |
| `plataform_proposed/data/BearingDataCenter`: the KAt/Paderborn raw data, outside the repos | `/mnt/datos/spade/upstream/BearingDataCenter` | 2656 / 27,725,501,987 |

- A symbolic link is left at each old path, so no path in the repo changes.
- `SPADE-dev/data → ../SPADE/data → /mnt/datos/spade/data`.
- `workspace/` is 204 KB and was not moved.
- **Not touched:** the general caches (`~/.cache/uv` 23 GB, `~/.cache/pip` 7.1 GB). They are not SPADE's, and moving the uv cache can affect the virtual environments. Moving them is an option for Luciano.

## Ollama

- **`qwen3:14b` deleted** (9.3 GB), with `ollama rm` through the user-level server on `127.0.0.1:11435` that served it. That server (pid 503168) was then stopped. It was a process started for AI-N30 3(a), never a configured service, so nothing is left to remove.
- **Pinned model:** `qwen3.5:9b`, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, the same as `configs/assistant.yaml` (checked through the client). Ollama 0.40.13.
- **Smoke:**
  - **Client, no tools, think on:** empty reply. The default `num_predict` is spent on thinking.
  - **Client, think off:** answers in 0.4 s, but describes "SPADE" as an itemset-mining algorithm. The model alone does not know this SPADE, which is why the assistant answers from catalog cards.
  - **Assistant** (`cli.build_agent`), "What is the dc_perm_ex_v2 plant?": answers in 6.8 s with the catalog cards and a correct description.
- **Not executed: moving the system service's model store** (11 GB in `/usr/share/ollama/.ollama/models`) to `/mnt/datos/ollama/models`. The session has no `sudo`.

### Commands for Luciano

Run them in this order; they keep `qwen3.5:9b` and its digest.

```bash
sudo systemctl stop ollama
sudo mkdir -p /mnt/datos/ollama/models
sudo rsync -a /usr/share/ollama/.ollama/models/ /mnt/datos/ollama/models/
sudo chown -R ollama:ollama /mnt/datos/ollama
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_MODELS=/mnt/datos/ollama/models"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl start ollama
ollama list        # qwen3.5:9b must show ID 56671c2ab938
# only after checking the line above:
sudo rm -rf /usr/share/ollama/.ollama/models
```

These commands free another ~11 GB on `/`.
