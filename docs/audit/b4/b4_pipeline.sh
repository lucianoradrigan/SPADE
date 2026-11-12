#!/usr/bin/env bash
# B4 pipeline (night 2026-11-09/10). Resumable: each stage is skipped if its output exists.
# Run from anywhere:  bash docs/audit/b4/b4_pipeline.sh >> "${B4_WORK:-workspace/b4_work}/b4_pipeline.log" 2>&1
# Environment (all optional): SPADE_REPO (default: the repo that holds this script), PY (default: $SPADE_REPO/.venv/bin/python),
# B4_WORK (logs, run pointers, times and ESP32 variants; default: $SPADE_REPO/workspace/b4_work; on 2026-11-10 it was /mnt/datos/spade/b4_work).
set -euo pipefail
SPADE_REPO="${SPADE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$SPADE_REPO"
export PYTHONPATH=src:. TF_CPP_MIN_LOG_LEVEL=2
PY="${PY:-$SPADE_REPO/.venv/bin/python}"
W="${B4_WORK:-$SPADE_REPO/workspace/b4_work}"
mkdir -p "$W"
T="/usr/bin/time -f STAGE_TIME\t%e_wall_s\t%U_user_s\t%S_sys_s -a -o $W/times.tsv"
SP=40,60,80,100,125,150,175,200,220
COMMON="--epochs 30 --batch-size 32 --seed 42 --cap-per-class 3000 --keep-rpm-level"
stage() { echo "=== $(date +%H:%M:%S) $1"; echo -n "$1	" >> $W/times.tsv; }
run_saved() { # runs train_model, records the run dir in $W/<name>.run
  local name=$1; shift
  [ -s $W/$name.run ] && { echo "skip $name ($(cat $W/$name.run))"; return; }
  stage "train_$name"; $T $PY experiments/train_model.py "$@" 2>&1 | tee $W/$name.log | grep -E "^Saved:|Error|Traceback" || true
  grep -oP "^Saved: \K.*" $W/$name.log | tail -1 > $W/$name.run; [ -s $W/$name.run ] || { echo "FAILED $name"; exit 1; }
}

[ -f docs/audit/b4/b4_preconditions.json ] || { stage preconditions; $T $PY experiments/b4_preconditions.py; }
$PY -c "import json,sys;d=json.load(open('docs/audit/b4/b4_preconditions.json'));sys.exit(0 if d['Q1']['pass'] and d['Q3']['pass'] else 'PRECONDITION FAILED: stop')"

[ -f data/b4_grid_3class.parquet ] || { stage gen_G3; $T $PY experiments/generate_diagnosis_grid_dataset.py --speeds $SP --first-run-seed 80001 --plant dc_perm_ex_v2 --out data/b4_grid_3class.parquet; }
[ -f data/b4_grid_5class.parquet ] || { stage gen_G5; $T $PY experiments/generate_diagnosis_grid_dataset.py --speeds $SP --first-run-seed 80001 --plant dc_perm_ex_v2 --faults outer_race,inner_race,ball,cage --out data/b4_grid_5class.parquet; }
[ -f data/b4_offgrid_healthy.parquet ] || { stage gen_offgrid; $T $PY experiments/generate_b4_sets.py offgrid --out data/b4_offgrid_healthy.parquet; }
[ -f data/b4_train_3class.parquet ] || { stage concat; $T $PY -c "
import pandas as pd
a=pd.read_parquet('data/b4_grid_3class.parquet'); b=pd.read_parquet('data/b4_offgrid_healthy.parquet')
assert not set(a.source_file.astype(str)) & set(b.source_file.astype(str))
pd.concat([a,b],ignore_index=True).to_parquet('data/b4_train_3class.parquet'); print(len(a),len(b))"; }
[ -f data/b4_e4_3class.parquet ] || { stage gen_E4_3; $T $PY experiments/generate_b4_sets.py e4 --grid 3class --out data/b4_e4_3class.parquet; }
[ -f data/b4_e4_5class.parquet ] || { stage gen_E4_5; $T $PY experiments/generate_b4_sets.py e4 --grid 5class --out data/b4_e4_5class.parquet; }
[ -f data/b4_eval_cache_5class.parquet ] || { stage gen_E3; $T $PY experiments/report_classifier_evaluation.py --generate-only --plant dc_perm_ex_v2 --extra-faults ball,cage --cache data/b4_eval_cache_5class.parquet; }
echo "=== $(date +%H:%M:%S) DATA_DONE"; du -sh data/b4_*.parquet

run_saved c1_teacher3 --config configs/classifiers/pc_server.yaml --dataset data/b4_train_3class.parquet $COMMON
run_saved c2_pc --config configs/classifiers/pc_server_5class_w2048.yaml --dataset data/b4_grid_5class.parquet $COMMON
run_saved c3_rpi5 --config configs/classifiers/rpi5_resnet1d_se_5class_w2048.yaml --dataset data/b4_grid_5class.parquet $COMMON --distill --distill-alpha 0.5 --teacher-run "$(cat $W/c2_pc.run)"
run_saved c4a_esp32 --config configs/classifiers/esp32_dscnn.yaml --dataset data/b4_train_3class.parquet $COMMON --distill --distill-alpha 0.5 --teacher-run "$(cat $W/c1_teacher3.run)"
run_saved c4b_esp32 --config configs/classifiers/esp32_dscnn.yaml --dataset data/b4_train_3class.parquet $COMMON
echo "=== $(date +%H:%M:%S) TRAIN_DONE"

R3=$(cat $W/c3_rpi5.run)
[ -f $R3/model.tflite ] || { stage export_rpi5; $T $PY -m experiments.export_tflite --domain dc_motor --tier rpi5 --block classifier --run-dir $R3; }
[ -f $W/esp32_selection.json ] || { stage esp32_variants; $T $PY experiments/b4_esp32_variants.py --k4a "$(cat $W/c4a_esp32.run)" --k4b "$(cat $W/c4b_esp32.run)" --work $W; }
echo "=== $(date +%H:%M:%S) EXPORT_DONE"
