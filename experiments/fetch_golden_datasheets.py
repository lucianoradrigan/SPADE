"""T8 (round C, decision C10): download the golden-set documents of annex A to data/upstream/assistant_golden/<id>/.

    python experiments/fetch_golden_datasheets.py [--only D1 P3 ...]

The files are third-party, copyrighted and NEVER committed (data/ is outside git). For each document the script
tries its URLs in order and keeps the first that returns the expected kind (a PDF for "pdf", an HTML page for "web").
It records the URL used, the date, the Content-Type, the size and the SHA-256 in
data/upstream/assistant_golden/fetch_log.json. A URL that is gone or returns something else (an error page, a login,
HTML instead of a PDF) is recorded with the reason; if none works the document is "unavailable".
A web page is saved as page.html and also printed to page.pdf with the system's headless Chrome (no new dependency).
No model reads these files: they are read by hand to write tests/assistant/golden/expected/<id>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "upstream" / "assistant_golden"
UA = "Mozilla/5.0 (X11; Linux x86_64) SPADE-golden-set-fetch/1.0"

#: id -> (kind, [urls in order]); the first is the one in RELEVAMIENTO_DATASHEETS_2026-11-02.md §1 when it exists.
DOCS = {
    "D1": ("pdf", ["https://www.thierry-lequeu.fr/data/karting/PMG-132-e.pdf"]),
    "D2": ("pdf", ["https://www.parvalux.com/product/pm11-110v-3000-rpm-2/datasheet.pdf"]),
    "D3": ("pdf", ["https://electricmotorsport.com/wp/wp-content/uploads/2026/06/me1514-data-sheet-motenergy.pdf"]),
    "D4": ("web", ["https://bodine-electric.com/products/dc-motors/33a-series-permanent-magnet-dc-motor/6435"]),
    "D5": ("pdf", ["https://cdn.zener.ru/media24/PDF/825046.pdf"]),
    "D6": ("pdf", ["https://www.mikrocontroller.net/attachment/687733/DC-Motor_31x51__1.13.021.3XX.pdf"]),
    "D7": ("pdf", ["https://www.radiolocman.com/datasheet/pdf.html?di=178953", "https://www.radiolocman.com/datasheet/pdf.html?di=178949"]),
    "P1": ("pdf", ["https://my.goagilix.com/assets/73/c4/73c408ce-327f-4137-967c-3dad152a8a6a/Siemens_1FK70425AF711UA5_Specification_Sheet.pdf"]),
    "P2": ("pdf", ["https://hallaweb.jlab.org/tech/Detectors/public_html/manuals/data_sheets-manuals/J-L/K/kollmorgen/AKM6%20Series%20Data%20Sheet.pdf"]),
    "P3": ("pdf", ["https://www.kollmorgen.com/sites/default/files/public_downloads/B23-Motor_Data-Sheet_en-US_Rev1.pdf"]),
    "P4": ("pdf", ["https://www.mouser.com/datasheet/2/632/DELTA_IA_ASDA_ASDA_B3_C_EN_20230523_web-3318784.pdf",
                   "https://inverterdrive.com/file/Delta-Section-A-Specifications-ASDA-B3-Manual-V9",
                   "https://backend.inverterdrive.com/file/Delta-ASDA-B3-Servo-Drive-Catalogue",
                   "https://backend.inverterdrive.com/file/Delta-Section-A-Specifications-ASDA-B3-Manual-V9"]),
    "P5": ("pdf", ["https://www.nanotec.com/fileadmin/files/Datenblaetter/BLDC/DB59/DB59C/DB59C024035-B3.pdf",
                   "https://www.nanotec-cn.cn/fileadmin/files/Datenblaetter/BLDC/DB59/DB59C/DB59C024035-B3.pdf",
                   "https://www.nanotec.com/fileadmin/files/Datenblaetter/BLDC/DB59/DB59C/DB59C024035-A.pdf"]),
    "P6": ("pdf", ["https://www.nanotec.com/fileadmin/files/Datenblaetter/BLDC/DB59/DB59C/DB59C048035-A.pdf"]),
    "P7": ("pdf", ["https://emrax.com/wp-content/uploads/2025/03/EMRAX-228_datasheet_v1.7.pdf"]),
    "P8": ("pdf", ["https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/171/EC-flat-60-bldc-datasheet.pdf"]),
    "P9": ("pdf", ["https://e2e.ti.com/cfs-file/__key/telligent-evolution-components-attachments/00-171-00-00-00-34-62-40/BLDC17FDatasheetR.pdf"]),
    "P10": ("pdf", ["https://www.resale.de/images/168677061_b_iba-1024-bosch_rexroth_motor_msk061c_datasheet.pdf"]),
    "P11": ("web", ["https://www.yaskawa.eu.com/products/motion-control/productdetail/product/sgm7j-02d7f61_6840.php"]),
    "I1": ("pdf", ["https://www.wolfautomation.com/content/weg-16616834-three-phase-motor-datasheet.pdf"]),
    "I2": ("pdf", ["https://www.baldor.com/api/products/GNEM3309T/infopacket"]),
    "R1": ("pdf", ["https://kollmorgen.com/sites/default/files/2024-11/Kollmorgen_Servo%20Motor%20Parameters%20and%20their%20Proper%20Conversions%20WP_RevD_en.pdf",
                   "https://www.kollmorgen.com/sites/default/files/public_downloads/Servomotor_Parameters_and_Conversions_10_15A_FINALDOC.pdf"]),
}


def _get(url: str) -> tuple[bytes, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return r.read(), r.headers.get("Content-Type", ""), r.geturl()


def _kind_ok(kind: str, body: bytes, ctype: str) -> str | None:
    """None if the body is the expected kind, else the reason."""
    if kind == "pdf":
        return None if body[:5] == b"%PDF-" else f"not a PDF (Content-Type {ctype!r}, starts {body[:40]!r})"
    text = body[:4000].decode("utf-8", "replace").lower()
    if "html" not in ctype.lower() and "<html" not in text:
        return f"not an HTML page (Content-Type {ctype!r})"
    if any(w in text for w in ("access denied", "403 forbidden", "404 not found", "page not found", "please log in", "sign in to")):
        return "error or login page"
    return None


def _print_pdf(html_url: str, out: Path) -> str | None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        return "no headless browser on this system"
    r = subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", f"--print-to-pdf={out}", html_url],
                       capture_output=True, text=True, timeout=180)
    return None if out.exists() and out.stat().st_size > 1000 else f"print failed: {r.stderr[-300:]}"


def fetch(doc_id: str) -> dict:
    kind, urls = DOCS[doc_id]
    folder = OUT / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    attempts = []
    for url in urls:
        try:
            body, ctype, final = _get(url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            attempts.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        reason = _kind_ok(kind, body, ctype)
        if reason:
            attempts.append({"url": url, "error": reason})
            continue
        name = "document.pdf" if kind == "pdf" else "page.html"
        (folder / name).write_bytes(body)
        rec = {"id": doc_id, "status": "downloaded", "kind": kind, "url": url, "final_url": final, "content_type": ctype,
               "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "file": f"{doc_id}/{name}",
               "consulted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "failed_attempts": attempts}
        if kind == "web":
            err = _print_pdf(url, folder / "page.pdf")
            rec["printed_pdf"] = None if err else {"file": f"{doc_id}/page.pdf", "sha256": hashlib.sha256((folder / "page.pdf").read_bytes()).hexdigest()}
            if err:
                rec["printed_pdf_error"] = err
        return rec
    return {"id": doc_id, "status": "unavailable", "kind": kind, "reason": "; ".join(a["error"] for a in attempts) or "no URL",
            "failed_attempts": attempts, "consulted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for doc_id in args.only or list(DOCS):
        rec = fetch(doc_id)
        log[doc_id] = rec
        print(f"{doc_id:4s} {rec['status']:12s} {rec.get('bytes', '')!s:>9} {rec.get('url') or rec.get('reason', '')[:120]}")
    log_path.write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
