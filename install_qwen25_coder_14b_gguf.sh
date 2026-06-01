#!/usr/bin/env bash
set -Eeuo pipefail

# install_qwen25_coder_14b_gguf.sh
#
# Installs Hugging Face CLI into a separate venv, downloads:
# Qwen/Qwen2.5-Coder-14B-Instruct-GGUF -> qwen2.5-coder-14b-instruct-q4_k_m.gguf
#
# Also writes .llama.env for worker.py / tester.py / explainer.py.
#
# Usage:
#   bash install_qwen25_coder_14b_gguf.sh
#
# Options:
#   --login      run interactive `hf auth login`
#   --force      re-download the model file
#
# Overrides:
#   LLM_DIR="$HOME/llm" bash install_qwen25_coder_14b_gguf.sh
#   LLAMA_CTX_SIZE=32768 bash install_qwen25_coder_14b_gguf.sh
#   LLAMA_MAX_TOKENS=8192 bash install_qwen25_coder_14b_gguf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLM_DIR="${LLM_DIR:-$HOME/llm}"
HF_VENV="${HF_VENV:-$LLM_DIR/venvs/hf}"
MODEL_ROOT="${MODEL_ROOT:-$LLM_DIR/models}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/gguf/qwen2.5-coder-14b-q4_k_m}"
CURRENT_LINK="${CURRENT_LINK:-$MODEL_ROOT/current.gguf}"
LLAMA_ENV_FILE="${LLAMA_ENV_FILE:-$SCRIPT_DIR/.llama.env}"

REPO_ID="${REPO_ID:-Qwen/Qwen2.5-Coder-14B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-coder-14b-instruct-q4_k_m.gguf}"

# Qwen2.5 Coder GGUF reports n_ctx_train=131072.
# Lower LLAMA_CTX_SIZE manually if your machine runs out of memory.
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-131072}"
LLAMA_MAX_TOKENS="${LLAMA_MAX_TOKENS:-8192}"

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

install_system_packages() {
  if command -v apt >/dev/null 2>&1; then
    log "Installing system packages"
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip ca-certificates curl git
  else
    log "Skipping apt install because apt was not found"
  fi
}

install_system_packages

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

SIZE_BYTES="$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$MODEL_PATH")"
if [[ "$SIZE_BYTES" -lt 1000000000 ]]; then
  warn "Downloaded file is smaller than 1 GB. It may be incomplete or a pointer file."
fi

log "Creating symlink"
ln -sfn "$MODEL_PATH" "$CURRENT_LINK"

log "Writing worker/tester/explainer environment"
{
  printf 'export LLAMA_MODEL_PATH=%q\n' "$MODEL_PATH"
  printf 'export LLAMA_CTX_SIZE=%q\n' "$LLAMA_CTX_SIZE"
  printf 'export LLAMA_MAX_TOKENS=%q\n' "$LLAMA_MAX_TOKENS"
} > "$LLAMA_ENV_FILE"

log "Done"
cat <<EOF

Model:
  $MODEL_PATH

Current symlink:
  $CURRENT_LINK -> $MODEL_PATH

Environment file:
  $LLAMA_ENV_FILE
  LLAMA_MODEL_PATH=$MODEL_PATH
  LLAMA_CTX_SIZE=$LLAMA_CTX_SIZE
  LLAMA_MAX_TOKENS=$LLAMA_MAX_TOKENS

Run:
  python3 explainer.py
  python3 worker.py
  python3 tester.py

If full context is too heavy, rerun for example:
  LLAMA_CTX_SIZE=32768 bash install_qwen25_coder_14b_gguf.sh
EOF
