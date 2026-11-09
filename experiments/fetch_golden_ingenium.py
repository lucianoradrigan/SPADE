"""T8.4 (round C, decision C10): Ingenium collection records of old motors (photo + official nameplate transcription).

    python experiments/fetch_golden_ingenium.py 1992.2220.001 1992.2292.001 ...

Licence: non-commercial (CC BY-NC as stated on the records) -> data/upstream/assistant_golden/ingenium/<artifact>/ only,
never committed. Saves the record page and its images and prints the 'Markings' text found in the page.
"""

import html
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "upstream" / "assistant_golden" / "ingenium"
UA = "SPADE-golden-set-fetch/1.0 (research use; https://github.com/lucianoradrigan/SPADE)"


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=90) as r:
        return r.read()


def main(ids):
    log = {}
    for a in ids:
        d = OUT / a
        d.mkdir(parents=True, exist_ok=True)
        rec = {"artifact": a}
        for base in ("https://collection.ingenium.ca/en/id/", "https://collection.ingeniumcanada.org/en/id/"):
            try:
                page = get(base + a)
                rec["url"] = base + a
                break
            except Exception as exc:  # noqa: BLE001
                rec.setdefault("errors", []).append(f"{base}: {exc}")
        else:
            log[a] = {**rec, "status": "unavailable"}
            continue
        (d / "page.html").write_bytes(page)
        text = html.unescape(re.sub(r"<[^>]+>", "\n", page.decode("utf-8", "replace")))
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        rec["status"] = "downloaded"
        rec["fields"] = {}
        for i, l in enumerate(lines):
            if l.rstrip(":").lower() in ("markings", "object type", "manufacturer", "date", "artifact no.", "artifact number", "model", "power", "description", "licence", "license", "copyright"):
                rec["fields"][l.rstrip(":")] = " | ".join(lines[i + 1:i + 3])
        own = a.replace(".", "-")  # the artifact's own images only (the page also shows related items)
        imgs = sorted(set(re.findall(r"https?://[^\"'\s>]+?\.(?:jpe?g|png)", page.decode("utf-8", "replace"), flags=re.I)))
        imgs = [u for u in imgs if own in u]
        rec["images"] = sorted(set(imgs + [u.replace("_thumb", "") for u in imgs]))
        for k, u in enumerate(rec["images"][:3]):
            try:
                (d / f"image_{k}{Path(u.split('?')[0]).suffix}").write_bytes(get(u))
            except Exception as exc:  # noqa: BLE001
                rec.setdefault("image_errors", []).append(f"{u}: {exc}")
        log[a] = rec
        print(a, rec["status"], json.dumps(rec["fields"], ensure_ascii=False)[:700])
    prev = json.loads((OUT / "fetch_log.json").read_text()) if (OUT / "fetch_log.json").exists() else {}
    (OUT / "fetch_log.json").write_text(json.dumps({**prev, **log}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
