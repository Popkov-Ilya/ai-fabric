#!/usr/bin/env bash
set -Eeuo pipefail

# install_qwen25_coder_14b_gguf.sh
#
# Устанавливает Hugging Face CLI (`hf`) в отдельный venv и скачивает:
# Qwen/Qwen2.5-Coder-14B-Instruct-GGUF -> qwen2.5-coder-14b-instruct-q4_k_m.gguf
#
# Результат:
#   ~/llm/models/gguf/qwen2.5-coder-14b-q4_k_m/qwen2.5-coder-14b-instruct-q4_k_m.gguf
#   ~/llm/models/current.gguf -> symlink на скачанную модель
#
# Использование:
#   bash install_qwen25_coder_14b_gguf.sh
#
# Опции:
#   --login      запустить интерактивный `hf auth login`
#   --force      перекачать файл даже если он уже есть
#
# Можно переопределить пути:
#   LLM_DIR="$HOME/llm" bash install_qwen25_coder_14b_gguf.sh

LLM_DIR="${LLM_DIR:-$HOME/llm}"
HF_VENV="${HF_VENV:-$LLM_DIR/venvs/hf}"
MODEL_ROOT="${MODEL_ROOT:-$LLM_DIR/models}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/gguf/qwen2.5-coder-14b-q4_k_m}"
CURRENT_LINK="${CURRENT_LINK:-$MODEL_ROOT/current.gguf}"

REPO_ID="${REPO_ID:-Qwen/Qwen2.5-Coder-14B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-coder-14b-instruct-q4_k_m.gguf}"

DO_LOGIN=0
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --login)
      DO_LOGIN=1
      ;;
    --force)
      FORCE=1
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Use --help" >&2
      exit 1
      ;;
  esac
done

log() {
  printf '\n\033[1;32m==>\033[0m %s\n' "$*"
}

warn() {
  printf '\n\033[1;33mWARNING:\033[0m %s\n' "$*" >&2
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
}

log "Installing system packages"
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ca-certificates curl git

log "Creating directories"
mkdir -p "$HF_VENV" "$MODEL_DIR" "$MODEL_ROOT"

if [[ ! -x "$HF_VENV/bin/python" ]]; then
  log "Creating Python venv for Hugging Face CLI: $HF_VENV"
  python3 -m venv "$HF_VENV"
fi

log "Installing/upgrading huggingface_hub CLI"
"$HF_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$HF_VENV/bin/python" -m pip install --upgrade "huggingface_hub[cli]"

HF_BIN="$HF_VENV/bin/hf"

if [[ ! -x "$HF_BIN" ]]; then
  echo "hf CLI was not found at $HF_BIN" >&2
  exit 1
fi

log "hf version"
"$HF_BIN" --version || true

if [[ -n "${HF_TOKEN:-}" ]]; then
  log "Logging in to Hugging Face using HF_TOKEN"
  "$HF_BIN" auth login --token "$HF_TOKEN"
elif [[ "$DO_LOGIN" -eq 1 ]]; then
  log "Starting interactive Hugging Face login"
  "$HF_BIN" auth login
else
  warn "Skipping Hugging Face login. This Qwen GGUF repo is public, so login is usually not required."
fi

MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

if [[ -f "$MODEL_PATH" && "$FORCE" -eq 0 ]]; then
  log "Model file already exists, skipping download"
  echo "$MODEL_PATH"
else
  if [[ -f "$MODEL_PATH" && "$FORCE" -eq 1 ]]; then
    log "Removing existing model file because --force was passed"
    rm -f "$MODEL_PATH"
  fi

  log "Downloading model"
  "$HF_BIN" download "$REPO_ID" "$MODEL_FILE" \
    --local-dir "$MODEL_DIR"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Download finished, but model file was not found:" >&2
  echo "$MODEL_PATH" >&2
  exit 1
fi

SIZE_BYTES="$(stat -c%s "$MODEL_PATH")"
if [[ "$SIZE_BYTES" -lt 1000000000 ]]; then
  warn "Downloaded file is smaller than 1 GB. It may be incomplete or a pointer file."
fi

log "Creating symlink"
ln -sfn "$MODEL_PATH" "$CURRENT_LINK"

log "Done"
cat <<EOF

Model:
  $MODEL_PATH

Current symlink:
  $CURRENT_LINK -> $MODEL_PATH

Quick checks:
  ls -lh "$MODEL_PATH"
  readlink -f "$CURRENT_LINK"

Example llama.cpp server command:
  llama-server -m "$CURRENT_LINK" -ngl 99 --ctx-size 8192 --host 127.0.0.1 --port 8080

If llama-server is not installed yet, build llama.cpp with CUDA separately.
EOF
