"""Record a cassette against the local Ollama (task F1.2). Not part of the test suite.

Input: a JSON file {"seed": 0, "requests": [{"messages": [...], "tools": [...]}, ...]} with Ollama
chat messages and tools. Each request is sent once to the model pinned in configs/assistant.yaml
(or --model TAG) and the exchange is written to the cassette, which ReplayClient replays in the
tests without network or GPU. Full conversations are recorded through the tool loop (task F1.3).

    python scripts/assistant_record_cassette.py requests.json tests/assistant/cassettes/<name>.json [--model TAG]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driveflow.assistant.config import load_config  # noqa: E402
from driveflow.assistant.core.llm import OllamaClient, RecordingClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("requests")
    ap.add_argument("cassette")
    ap.add_argument("--model", help="model tag to use instead of the pinned one (digest read from Ollama)")
    args = ap.parse_args()
    cfg = load_config()
    if args.model:
        cfg["model"] = {"tag": args.model, "digest": None}
    client = OllamaClient(cfg)
    if args.model:  # pin the installed digest of the requested tag for this recording
        found = next((m for m in client.installed_models() if m.get("name") == args.model), None)
        client.digest = found.get("digest") if found else None
    info = client.check()
    spec = json.loads(Path(args.requests).read_text())
    rec = RecordingClient(client, model_id=client.model_id(), runtime_version=info["runtime_version"])
    for r in spec["requests"]:
        rec.chat(r["messages"], r.get("tools", []), seed=spec.get("seed", 0))
    Path(args.cassette).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cassette).write_text(json.dumps(rec.cassette(), indent=1, ensure_ascii=False) + "\n")
    print(f"recorded {len(rec.exchanges)} exchanges with {client.model_id()} (Ollama {info['runtime_version']}) -> {args.cassette}")
    for w in info["warnings"]:
        print("warning:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
