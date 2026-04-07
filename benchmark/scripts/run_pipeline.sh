#!/bin/bash
#
# End-to-end Memory Affordance pipeline driver.
#
# Runs: keyframes -> QA generation -> uniqueness check -> mask generation -> benchmark build
#
# Stages can be skipped via env vars:
#   SKIP_KEYFRAMES=1 SKIP_QA=1 SKIP_CHECK=1 SKIP_MASKS=1 SKIP_BUILD=1
#
# Usage:
#   bash run_pipeline.sh
#
# Tweak these to point at your data:
#   VIDEO_DIR    - directory with .mp4 files
#   KEYFRAMES    - where to write keyframes
#   QA_RESULTS   - where to write QA + check results
#   MASKS        - where to write masks
#   BENCHMARK    - where to write final benchmark
#
# Required env (for QA generation):
#   QWEN_AUTH_TOKEN - DashScope API key (sk-...)

set -euo pipefail

# ----------------------------------------------------------------------
# Config (override via env vars)
# ----------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/KRA_src}"
VIDEO_DIR="${VIDEO_DIR:-$PROJECT_DIR/data/ego4d/batch_videos}"
KEYFRAMES="${KEYFRAMES:-$PROJECT_DIR/data/keyframes/ego4d_batch}"
QA_RESULTS="${QA_RESULTS:-$PROJECT_DIR/results/episode_qa}"
MASKS="${MASKS:-$PROJECT_DIR/results/episode_masks}"
BENCHMARK="${BENCHMARK:-$PROJECT_DIR/benchmark/final}"

QWEN_MODEL="${QWEN_MODEL:-qwen3.5-397b-a17b}"
QWEN_API_URL="${QWEN_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
QWEN_AUTH_TOKEN="${QWEN_AUTH_TOKEN:-}"

KF_INTERVAL="${KF_INTERVAL:-180}"
KF_MAX="${KF_MAX:-50}"
KF_DEDUPE="${KF_DEDUPE:-12}"

QA_MAX_IMAGES="${QA_MAX_IMAGES:-30}"
QA_TEMPERATURE="${QA_TEMPERATURE:-0.5}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
err() { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

# Activate env if needed
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mem-aff 2>/dev/null || true
  elif [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    source "$PROJECT_DIR/.venv/bin/activate"
  fi
fi

cd "$PROJECT_DIR"

# ----------------------------------------------------------------------
# Stage 1 — Keyframe extraction
# ----------------------------------------------------------------------
if [[ -z "${SKIP_KEYFRAMES:-}" ]]; then
  log "Stage 1: Extracting keyframes from $VIDEO_DIR"
  python benchmark/scripts/extract_keyframes.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$KEYFRAMES" \
    --strategy uniform \
    --interval "$KF_INTERVAL" \
    --max_frames "$KF_MAX" \
    --dedupe_threshold "$KF_DEDUPE"
else
  log "Stage 1: SKIPPED (SKIP_KEYFRAMES set)"
fi

# ----------------------------------------------------------------------
# Stage 2 — QA generation (Qwen API)
# ----------------------------------------------------------------------
if [[ -z "${SKIP_QA:-}" ]]; then
  [[ -z "$QWEN_AUTH_TOKEN" ]] && err "QWEN_AUTH_TOKEN not set"
  log "Stage 2: Generating QA pairs with $QWEN_MODEL"
  python benchmark/scripts/episode_qa_generation.py \
    --episodes_root "$KEYFRAMES" \
    --output_dir "$QA_RESULTS" \
    --max_images "$QA_MAX_IMAGES" \
    --model_name "$QWEN_MODEL" \
    --api_url "$QWEN_API_URL" \
    --api_key "$QWEN_AUTH_TOKEN" \
    --temperature "$QA_TEMPERATURE"
else
  log "Stage 2: SKIPPED (SKIP_QA set)"
fi

# ----------------------------------------------------------------------
# Stage 3 — Uniqueness check
# ----------------------------------------------------------------------
if [[ -z "${SKIP_CHECK:-}" ]]; then
  [[ -z "$QWEN_AUTH_TOKEN" ]] && err "QWEN_AUTH_TOKEN not set"
  log "Stage 3: QA uniqueness check"
  python benchmark/scripts/qa_uniqueness_check.py \
    --qa_results_dir "$QA_RESULTS" \
    --keyframes_root "$KEYFRAMES" \
    --model_name "$QWEN_MODEL" \
    --api_url "$QWEN_API_URL" \
    --api_key "$QWEN_AUTH_TOKEN"
else
  log "Stage 3: SKIPPED (SKIP_CHECK set)"
fi

# ----------------------------------------------------------------------
# Stage 4 — Mask generation (GPU required)
# ----------------------------------------------------------------------
if [[ -z "${SKIP_MASKS:-}" ]]; then
  log "Stage 4: Mask generation (Rex-Omni + SAM2) — requires GPU"
  python benchmark/scripts/generate_masks.py \
    --qa_check_dir "$QA_RESULTS" \
    --keyframes_root "$KEYFRAMES" \
    --output_dir "$MASKS" \
    --device cuda
else
  log "Stage 4: SKIPPED (SKIP_MASKS set)"
fi

# ----------------------------------------------------------------------
# Stage 5 — Benchmark build
# ----------------------------------------------------------------------
if [[ -z "${SKIP_BUILD:-}" ]]; then
  log "Stage 5: Building final benchmark"
  python benchmark/scripts/build_benchmark.py \
    --qa_results_dir "$QA_RESULTS" \
    --keyframes_dir "$KEYFRAMES" \
    --output_dir "$BENCHMARK"
else
  log "Stage 5: SKIPPED (SKIP_BUILD set)"
fi

log "Pipeline complete."
echo "  QA results:  $QA_RESULTS"
echo "  Masks:       $MASKS"
echo "  Benchmark:   $BENCHMARK"
