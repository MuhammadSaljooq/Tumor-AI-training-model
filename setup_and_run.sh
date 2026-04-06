#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap + run script.
# Usage examples:
#   bash setup_and_run.sh
#   bash setup_and_run.sh --models resnet50 --epochs 2 --train
#   bash setup_and_run.sh --models vit --evaluate-only

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"
CONFIG_PATH="configs/config.yaml"
DATA_DIR="data/processed"
MODELS="resnet50"
FORCE_TRAIN=0
FORCE_EVAL=0
EPOCHS=""
SKIP_PREPROCESSING=1

print_help() {
  cat <<'EOF'
setup_and_run.sh - bootstrap environment and run pipeline

Options:
  --models <names>       Space/comma separated models (default: resnet50)
                         Allowed: resnet50, vit, hybrid
  --epochs <n>           Override training epochs for this run only
  --data-dir <path>      Dataset root (default: data/processed)
  --train                Force training mode
  --evaluate-only        Force evaluate-only mode (requires checkpoints)
  --with-preprocessing   Run preprocessing stage
  -h, --help             Show help

Examples:
  bash setup_and_run.sh
  bash setup_and_run.sh --models resnet50 --epochs 2 --train
  bash setup_and_run.sh --models "resnet50 vit" --evaluate-only
EOF
}

normalize_models() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  local arr=($raw)
  local normalized=()
  for m in "${arr[@]}"; do
    case "$m" in
      resnet50|vit|hybrid) normalized+=("$m") ;;
      *)
        echo "Invalid model: $m (allowed: resnet50, vit, hybrid)"
        exit 1
        ;;
    esac
  done
  if [ "${#normalized[@]}" -eq 0 ]; then
    echo "No valid models provided."
    exit 1
  fi
  echo "${normalized[*]}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --train)
      FORCE_TRAIN=1
      shift
      ;;
    --evaluate-only)
      FORCE_EVAL=1
      shift
      ;;
    --with-preprocessing)
      SKIP_PREPROCESSING=0
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

if [ "$FORCE_TRAIN" -eq 1 ] && [ "$FORCE_EVAL" -eq 1 ]; then
  echo "Use only one of --train or --evaluate-only."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found."
  exit 1
fi

MODELS="$(normalize_models "$MODELS")"

echo "==> Setting up virtual environment"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

if [ ! -d "$DATA_DIR" ]; then
  echo "Dataset directory not found: $DATA_DIR"
  echo "Expected processed dataset under:"
  echo "  $DATA_DIR/Training/<class> and $DATA_DIR/Testing/<class>"
  echo "Tip: run quick_start.py if you want Kaggle auto-download flow."
  exit 1
fi

RUN_CONFIG="$CONFIG_PATH"
TEMP_CONFIG=""
if [ -n "$EPOCHS" ]; then
  if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "--epochs must be an integer."
    exit 1
  fi
  echo "==> Creating temporary config with epochs=$EPOCHS"
  TEMP_CONFIG="configs/.tmp_config_epochs_${EPOCHS}.yaml"
  python - "$CONFIG_PATH" "$TEMP_CONFIG" "$EPOCHS" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(Path(src).read_text())
cfg.setdefault("training", {})["epochs"] = epochs
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY
  RUN_CONFIG="$TEMP_CONFIG"
fi

IFS=' ' read -r -a MODEL_ARRAY <<< "$MODELS"
EVAL_FLAG=""
if [ "$FORCE_EVAL" -eq 1 ]; then
  EVAL_FLAG="--evaluate_only"
elif [ "$FORCE_TRAIN" -eq 0 ]; then
  # Auto-switch to evaluate-only if all requested checkpoints exist.
  all_ckpts=1
  for model in "${MODEL_ARRAY[@]}"; do
    if [ ! -f "results/checkpoints/${model}_best.pth" ]; then
      all_ckpts=0
      break
    fi
  done
  if [ "$all_ckpts" -eq 1 ]; then
    EVAL_FLAG="--evaluate_only"
    echo "==> Found checkpoints for requested models: running evaluate-only"
  fi
fi

echo "==> Running pipeline"
CMD=(python -u main.py --config "$RUN_CONFIG" --data_dir "$DATA_DIR" --models)
for model in "${MODEL_ARRAY[@]}"; do
  CMD+=("$model")
done
if [ "$SKIP_PREPROCESSING" -eq 1 ]; then
  CMD+=(--skip_preprocessing)
fi
if [ -n "$EVAL_FLAG" ]; then
  CMD+=("$EVAL_FLAG")
fi

printf 'Command: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [ -n "$TEMP_CONFIG" ] && [ -f "$TEMP_CONFIG" ]; then
  rm -f "$TEMP_CONFIG"
fi

echo "==> Done"
echo "Check outputs in:"
echo "  results/checkpoints/"
echo "  results/logs/"
echo "  results/plots/"
