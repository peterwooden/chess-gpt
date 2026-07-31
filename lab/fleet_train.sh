#!/bin/bash
# Overnight fleet: one atomic training per experiment in lab/OVERNIGHT.md.
# FLEET_FLAGS carries the speed flags chosen by the recipe suite.
set -u
cd "$(dirname "$0")/.."
FLAGS=${FLEET_FLAGS:-}
OUT=runs/lab/fleet
mkdir -p "$OUT"
LAB=data/downloads/lab
D40=$LAB/enriched-40k.parquet

T() {
  id=$1; shift
  if [ -f "$OUT/$id.json" ]; then echo "skip $id"; return; fi
  echo "== $id =="
  # shellcheck disable=SC2086
  uv run python -m lab.train --output "$OUT/$id.json" $FLAGS "$@" >/dev/null 2>&1 \
    && echo "done $id" || { echo "FAIL $id"; touch "$OUT/$id.SKIP"; }
}

T 00-control       --data $D40 --games 10000
# --- data ---
T 01-winner1600    --data $LAB/winner1600.parquet --games 19000
T 02-winner2000    --data $LAB/winner2000-enriched.parquet --games 18500
T 03-decisive      --data $LAB/decisive-10k.parquet
T 04-nobullet      --data $LAB/nobullet-10k.parquet
T 05-20k           --data $D40 --games 20000
T 06-40k           --data $D40 --games 40000
T 07-dedup         --data $LAB/dedup-10k.parquet
T 08-both1800      --data $LAB/both1800-10k.parquet
T 09-drawsonly     --data $LAB/draws-8k.parquet --games 8000
# --- representation ---
T 10-hist2         --data $D40 --games 10000 --history 2
T 11-hist8         --data $D40 --games 10000 --history 8
T 12-repetition    --data $D40 --games 10000 --repetition-feature
T 13-material      --data $D40 --games 10000 --material-feature
T 14-nostate       --data $D40 --games 10000 --no-state-token
# --- architecture ---
T 15-mlp           --data $D40 --games 10000 --arch mlp --layers 2 --mlp-hidden 192
T 16-persquare     --data $D40 --games 10000 --per-square-readout
T 17-moe           --data $D40 --games 10000 --moe
T 18-geobias       --data $D40 --games 10000 --geo-bias
T 19-pieceinit     --data $D40 --games 10000 --piece-value-init
T 20-d64L2         --data $D40 --games 10000 --d-model 64 --layers 2
T 21-d96L4         --data $D40 --games 10000 --d-model 96 --layers 4
T 22-d160L8        --data $D40 --games 10000 --d-model 160 --layers 8 --heads 8
T 23-heads8        --data $D40 --games 10000 --heads 8
T 24-heads2        --data $D40 --games 10000 --heads 2
T 25-ff2           --data $D40 --games 10000 --ff-mult 2
T 26-dropout       --data $D40 --games 10000 --dropout 0.1
# --- objective ---
T 27-value025      --data $D40 --games 10000 --value-weight 0.25
T 28-value05       --data $D40 --games 10000 --value-weight 0.5
T 29-value10       --data $D40 --games 10000 --value-weight 1.0
T 30-auxmat        --data $D40 --games 10000 --aux-material
T 31-auxplies      --data $D40 --games 10000 --aux-plies
T 32-labelsmooth   --data $D40 --games 10000 --label-smoothing 0.1
T 33-valueonly     --data $D40 --games 10000 --policy-weight 0 --value-weight 1.0
# --- optimization ---
T 34-epochs3       --data $D40 --games 10000 --epochs 3
T 35-cosine        --data $D40 --games 10000 --epochs 3 --schedule cosine
T 36-lr3e3         --data $D40 --games 10000 --learning-rate 3e-3
T 37-b4096         --data $D40 --games 10000 --batch-size 4096 --learning-rate 4.8e-3
T 38-warmup        --data $D40 --games 10000 --warmup-frac 0.05
# --- post-training (need parents above) ---
T 39-ft-elite      --data $LAB/winner2000-enriched.parquet --games 18500 --init-checkpoint $OUT/00-control.pt
T 40-ft-decisive   --data $LAB/decisive-10k.parquet --init-checkpoint $OUT/00-control.pt
if [ ! -f $LAB/selfplay-41.parquet ]; then
  uv run python -m lab.selfplay --checkpoint $OUT/00-control.pt --games 1000 \
    --temperature 0.8 --winner-only --output $LAB/selfplay-41.parquet >/dev/null 2>&1
fi
T 41-selfplay      --data $LAB/selfplay-41.parquet --games 999999 --init-checkpoint $OUT/00-control.pt
if [ ! -f $LAB/selfplay-42.parquet ]; then
  uv run python -m lab.selfplay --checkpoint $OUT/28-value05.pt --games 600 \
    --search value1 --winner-only --output $LAB/selfplay-42.parquet >/dev/null 2>&1
fi
T 42-expertiter    --data $LAB/selfplay-42.parquet --games 999999 --init-checkpoint $OUT/28-value05.pt
T 43-distill       --data $D40 --games 10000 --d-model 64 --layers 2 --teacher runs/lab/runA-d128L6.pt
echo FLEET-TRAIN-DONE
