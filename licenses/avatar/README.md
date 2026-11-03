# Avatar of the SPADE assistant ("PhD. Radrigan")

The images in `src/driveflow/viz/assets/assistant/` are the likeness of Luciano Radrigan. They are
**not** covered by SPADE's MIT license (`LICENSE`); their terms are in [`LICENSE`](LICENSE) here:
all rights reserved, redistribution only unmodified and as part of SPADE.

The current files are **provisional** (decision FE-D2, 2026-11-02): generated from the caricature
with the black background and the baked-in text removed. The final poses replace them without code
changes. The original caricature (JPG) is not in the repository.

## Replacing the avatar

Point `persona.avatar_dir` in `configs/assistant.yaml` to a directory with your own layers:

| File | Content |
|---|---|
| `base.webp` | the body, with a real alpha channel |
| `hand.webp` | the hand as its own layer (it pivots at 60 % / 100 % of its box) |

If the directory does not have both files, the assistant page shows a neutral silhouette and logs
it; it never fails (`driveflow.assistant.config.avatar_assets`). The visible name stays next to the
tag "AI assistant · local model" (FE-D5).
