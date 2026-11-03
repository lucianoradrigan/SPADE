#!/usr/bin/env bash
# AI-N27 option 2 (docs/audit/assistant/ai_n27_regen_preregistration.md): generation (§1.2), P2,
# training and TFLite export (§2), P3. Stops at the first failure. Evaluation (§4-§6) is a separate
# step (experiments/report_ai_n27_regen.py). Run from the repository root:
#   bash scripts/oneoff/ai_n27_regen_pipeline.sh <log dir> [start stage]
set -euo pipefail
LOG=${1:?log dir}
START=${2:-gen3}
mkdir -p "$LOG"
PY=".venv/bin/python"
export PYTHONPATH=src
W=10
TRAIN="--epochs 30 --batch-size 32 --seed 42 --keep-rpm-level --cap-per-class 3000"
started=0
stage() {  # stage <name> <command...>; skips stages before START; times each one
  local name=$1; shift
  [[ $name == "$START" ]] && started=1
  if [[ $started == 0 ]]; then echo "skip $name"; return; fi
  echo "[$(date -Is)] start $name: $*" | tee -a "$LOG/pipeline.log"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date -Is)] done  $name in $((SECONDS - t0)) s" | tee -a "$LOG/pipeline.log"
}
run_dir() { grep -o 'Saved: .*' "$LOG/$1.log" | tail -1 | sed 's/Saved: //; s|.*/configs/|configs/|'; }

stage gen3 $PY experiments/generate_diagnosis_grid_dataset.py --dc-converter-model dcm --dc-plant-substeps 1000 --seeds 3 --duration-s 2.5 --seed 7 --workers $W --out data/diagnosis_grid_dataset_dcm.parquet
stage gen5 $PY experiments/generate_diagnosis_grid_dataset.py --dc-converter-model dcm --dc-plant-substeps 1000 --seeds 3 --duration-s 2.5 --seed 7 --workers $W --faults outer_race,inner_race,ball,cage --out data/diagnosis_grid_dataset_5class_dcm.parquet
stage held3 $PY experiments/generate_ai_n27_heldout.py --grid 3class --workers $W --out data/ai_n27_heldout_3class_dcm.parquet
stage held5 $PY experiments/generate_ai_n27_heldout.py --grid 5class --workers $W --out data/ai_n27_heldout_5class_dcm.parquet
stage geneval $PY experiments/report_classifier_evaluation.py --dc-converter-model dcm --dc-plant-substeps 1000 --extra-faults ball,cage --seeds 4 --first-seed 200 --duration-s 1.5 --workers $W --cache data/eval_cache_5class_dcm.parquet --generate-only
stage p2 $PY experiments/report_ai_n27_regen_preconditions.py p2

stage train_pc3 $PY experiments/train_model.py --config configs/classifiers/pc_server.yaml --dataset data/diagnosis_grid_dataset_dcm.parquet $TRAIN
PC3=$(run_dir train_pc3)
stage train_esp32 $PY experiments/train_model.py --config configs/classifiers/esp32_dscnn.yaml --dataset data/diagnosis_grid_dataset_dcm.parquet $TRAIN --distill --distill-alpha 0.5 --teacher-run "$PC3"
ESP=$(run_dir train_esp32)
stage train_pc5 $PY experiments/train_model.py --config configs/classifiers/pc_server_5class_w2048.yaml --dataset data/diagnosis_grid_dataset_5class_dcm.parquet $TRAIN
PC5=$(run_dir train_pc5)
stage train_rpi5 $PY experiments/train_model.py --config configs/classifiers/rpi5_resnet1d_se_5class_w2048.yaml --dataset data/diagnosis_grid_dataset_5class_dcm.parquet $TRAIN --distill --distill-alpha 0.5 --teacher-run "$PC5"
RPI=$(run_dir train_rpi5)
stage export_rpi5 $PY -m experiments.export_tflite --domain dc_motor --tier rpi5 --block classifier --run-dir "$RPI"
stage export_esp32 $PY -m experiments.export_tflite --domain dc_motor --tier esp32 --block classifier --run-dir "$ESP" --dataset data/diagnosis_grid_dataset_dcm.parquet --n-calibration-samples 1000
stage p3 $PY experiments/report_ai_n27_regen_preconditions.py p3 --pairs \
  "configs/classifiers/pc_server/2026-10-25_run01=$PC3" \
  "configs/classifiers/esp32_dscnn/2026-10-18_run01=$ESP" \
  "configs/classifiers/pc_server_5class_w2048/2026-10-20_run01=$PC5" \
  "configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01=$RPI"
printf 'PC3=%s\nESP32=%s\nPC5=%s\nRPI5=%s\n' "$PC3" "$ESP" "$PC5" "$RPI" | tee "$LOG/candidates.txt"
echo "[$(date -Is)] pipeline finished" | tee -a "$LOG/pipeline.log"
