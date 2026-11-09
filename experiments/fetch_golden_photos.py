"""T8.4 (round C, decision C10): candidate photos of electric-motor rating plates from Wikimedia Commons, by API.

    python experiments/fetch_golden_photos.py search          # list candidates with their licence -> data/upstream/assistant_golden/photos/candidates.json
    python experiments/fetch_golden_photos.py download T1 ...  # download chosen titles (keys of candidates.json)

Licence rule (annex A): CC0 or CC BY -> may be committed to tests/assistant/golden/photos/ (resized to <= 1600 px,
with ATTRIBUTION.md); CC BY-SA -> data/upstream/ only; anything else (public domain marks, GFDL, unknown, non-free) is
not used. The licence is read from each file's own information page (imageinfo extmetadata), not from the search.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = "SPADE-golden-set-fetch/1.0 (research use; https://github.com/lucianoradrigan/SPADE)"
UPSTREAM = REPO / "data" / "upstream" / "assistant_golden" / "photos"
QUERIES = ["electric motor rating plate", "motor nameplate", "Typenschild Elektromotor", "Leistungsschild Motor",
           "Typenschild Drehstrommotor", "nameplate induction motor", "servo motor nameplate", "DC motor nameplate", "Motortypenschild",
           "intitle:Typenschild", "intitle:Leistungsschild", "intitle:nameplate motor", "intitle:\"rating plate\"", "intitle:Motorschild",
           "Elektromotor Typenschild kW", "motor data plate kW rpm"]
#: Commons calls rating plates "manufacturing tablets"; these two hold the motor plates found (2026-11-09).
CATEGORIES = ["Category:Manufacturing tablets of industrial devices for producing, converting or storing electrical energy",
              "Category:Manufacturing tablets of machinery"]


def api(**params) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def licence_class(short: str) -> str:
    s = (short or "").lower().replace("-", " ")
    if s.startswith("cc0") or "cc zero" in s:
        return "cc0"
    if s.startswith("cc by sa") or s.startswith("cc by sa"):
        return "cc_by_sa"
    if s.startswith("cc by") and "nc" not in s and "nd" not in s:
        return "cc_by"
    return "other"


def info(titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 40):
        d = api(action="query", titles="|".join(titles[i:i + 40]), prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth="1600")
        for p in d["query"]["pages"]:
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            g = lambda k: (em.get(k) or {}).get("value")
            out.append({"title": p["title"], "mime": ii.get("mime"), "width": ii.get("width"), "height": ii.get("height"),
                        "url": ii.get("url"), "thumb_url": ii.get("thumburl"), "page": ii.get("descriptionurl"),
                        "licence": g("LicenseShortName"), "licence_url": g("LicenseUrl"), "artist": g("Artist"),
                        "credit": g("Credit"), "description": g("ImageDescription"), "class": licence_class(g("LicenseShortName"))})
    return out


def search() -> None:
    titles = set()
    for q in QUERIES:
        d = api(action="query", list="search", srsearch=q, srnamespace="6", srlimit="50")
        titles |= {r["title"] for r in d["query"]["search"]}
    for c in CATEGORIES:
        try:
            d = api(action="query", list="categorymembers", cmtitle=c, cmtype="file", cmlimit="200")
            titles |= {m["title"] for m in d["query"]["categorymembers"]}
        except Exception:  # noqa: BLE001 -- a category that does not exist
            pass
    cands = [c for c in info(sorted(titles)) if (c["mime"] or "").startswith("image/")]
    UPSTREAM.mkdir(parents=True, exist_ok=True)
    (UPSTREAM / "candidates.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False) + "\n")
    by = {}
    for c in cands:
        by[c["class"]] = by.get(c["class"], 0) + 1
    print(len(cands), "image candidates; by licence class:", by)


def download(titles: list[str], committed_dir: Path) -> None:
    cands = {c["title"]: c for c in json.loads((UPSTREAM / "candidates.json").read_text())}
    for t in titles:
        c = cands[t]
        dest_dir = committed_dir if c["class"] in ("cc0", "cc_by") else UPSTREAM if c["class"] == "cc_by_sa" else None
        if dest_dir is None:
            print("skip (licence not usable):", t, c["licence"])
            continue
        url = c["thumb_url"] if (c["width"] or 0) > 1600 else c["url"]
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        name = t.split(":", 1)[1].replace(" ", "_")
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / name).write_bytes(body)
        print(c["class"], len(body), dest_dir / name)


if __name__ == "__main__":
    if sys.argv[1] == "search":
        search()
    else:
        download(sys.argv[2:], REPO / "tests" / "assistant" / "golden" / "photos")
