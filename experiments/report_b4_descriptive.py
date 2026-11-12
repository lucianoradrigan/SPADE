"""B4 §5.4 (descriptive, pre-registered; changes no verdict): the bootstrap 95 % CI of delta macro-F1 (1000 resamples of
acquisitions, seed 0) and the confusion matrices of candidate and promoted run on E1, E3 and E4. Both were computed by
report_b4.py (report_ai_n27_regen._compare) and stored in b4_results.json; this script only lays them out, adds the
teacher of every distilled B4 run (AI-N37: train_model did not record it then) and writes the section into b4_results.md.

    python experiments/report_b4_descriptive.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "audit" / "b4" / "b4_results"
BEGIN, END = "<!-- B4-5.4-BEGIN -->", "<!-- B4-5.4-END -->"
# AI-N37: the teachers passed on the command line (docs/audit/b4/b4_pipeline.sh); run_metadata.json of these runs is not rewritten.
TEACHERS = {"configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-10_run01": "configs/classifiers/pc_server_5class_w2048/2026-11-10_run01",
            "configs/classifiers/esp32_dscnn/2026-11-10_run01": "configs/classifiers/pc_server/2026-11-10_run01"}


def main() -> None:
    d = json.loads(OUT.with_suffix(".json").read_text())
    desc = {}
    L = [BEGIN, "", "## §5.4 descriptive (pre-registered; no verdict changes)", "",
         "Computed by `report_b4.py` at evaluation time (stored in `b4_results.json`) and laid out by "
         "`python experiments/report_b4_descriptive.py`. CI: 95 % percentile bootstrap of Δ macro-F1 (candidate − promoted, points), "
         "1000 resamples of whole acquisitions, seed 0.", "",
         "| Tier | Set | Acquisitions | Δ macro-F1 (pt) | 95 % CI (pt) | CI below −1 pt? |", "|---|---|---|---|---|---|"]
    for tier, t in d["tiers"].items():
        desc[tier] = {}
        for s, v in t["sets"].items():
            ci = v.get("delta_macro_f1_ci95_points")
            desc[tier][s] = {"delta_macro_f1_points": v["delta_points"]["macro_f1"], "ci95_points": ci, "n_acquisitions": v["n_acquisitions"],
                             "confusion_candidate": v["candidate"]["confusion"], "confusion_promoted": v["promoted"]["confusion"]}
            L.append(f"| {tier} | {s} | {v['n_acquisitions']} | {v['delta_points']['macro_f1']:+.1f} | [{ci[0]:+.1f}, {ci[1]:+.1f}] | {'yes' if ci[0] < -1 else 'no'} |")
    L += ["", "**Reading.** No CI changes a verdict (the rule is on the point estimates). Four notes:",
          "- **RPi5 E1:** the point estimate (−0.4 pt) passes, but the CI reaches −2.2 pt: non-inferiority on E1 is not established with confidence.",
          "- **ESP32 E1:** +3.1 pt, CI [−2.6, +10.4]: the gain is not established.",
          "- **ESP32 E3:** the whole CI is below 0 ([−0.4, −0.1] pt): a small but consistent loss, within the −1 pt margin.",
          "- **ESP32 E4:** the CI spans −12.3 to +16.8 pt (36 acquisitions): the low-speed comparison is not informative on its own; the int8-vs-Keras check is.",
          "", "### Confusion matrices (rows: true class, columns: predicted; argmax, abstention not applied)", ""]
    for tier, t in d["tiers"].items():
        cls = t["classes"]
        for s in ("E1", "E3", "E4"):
            v = t["sets"][s]
            L += [f"**{tier} · {s}**", "", "| true \\ pred | " + " | ".join(cls) + " |", "|---|" + "---|" * len(cls)]
            for i, c in enumerate(cls):
                L.append(f"| {c} | " + " | ".join(f"{v['candidate']['confusion'][i][j]} ({v['promoted']['confusion'][i][j]})" for j in range(len(cls))) + " |")
            L += ["", "Candidate (promoted).", ""]
    L += ["- **ESP32 E4:** 306 of 468 inner_race windows go to outer_race in the int8 candidate (the promoted run's error goes the other way). That is where AI-N36 loses inner_race.",
          "", "### Teachers of the distilled B4 runs (AI-N37)", "", "| Run | Teacher (from `b4_pipeline.sh`) |", "|---|---|"]
    L += [f"| `{r}` | `{t}` |" for r, t in TEACHERS.items()]
    L += ["", END]
    d["descriptive_5_4"] = desc
    d["teacher_run"] = TEACHERS
    d["deviations"] = [x if not x.startswith("§5.4") else "§5.4: CI and confusion matrices were computed and stored in the JSON at evaluation time, but the first report omitted them and wrongly called them not executed (corrected 2026-11-12)" for x in d["deviations"]]
    OUT.with_suffix(".json").write_text(json.dumps(d, indent=2, default=float) + "\n")
    md = OUT.with_suffix(".md").read_text()
    sec = "\n".join(L) + "\n"
    if BEGIN in md:
        md = md[: md.index(BEGIN)] + sec + md[md.index(END) + len(END) + 1:]
    else:
        md = md.replace("\n## New findings", "\n" + sec + "\n## New findings")
    OUT.with_suffix(".md").write_text(md)
    print("ok")


if __name__ == "__main__":
    main()
