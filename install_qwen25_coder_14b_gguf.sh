#!/usr/bin/env bash
set -Eeuo pipefail

# install_qwen25_coder_14b_gguf.sh
#
# Installs Python packages into the current python3 environment, downloads:
# Qwen/Qwen2.5-Coder-14B-Instruct-GGUF -> qwen2.5-coder-14b-instruct-q4_k_m.gguf
#
# Also writes .llama.env for worker.py / tester.py / explainer.py.
#
# Run with --help to print usage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INSTALL_SCOPE="${PIP_INSTALL_SCOPE:---user}"

LLM_DIR="${LLM_DIR:-$HOME/llm}"
MODEL_ROOT="${MODEL_ROOT:-$LLM_DIR/models}"
MODEL_DIR_OVERRIDE="${MODEL_DIR:-}"
CURRENT_LINK="${CURRENT_LINK:-$MODEL_ROOT/current.gguf}"
LLAMA_ENV_FILE="${LLAMA_ENV_FILE:-$SCRIPT_DIR/.llama.env}"

DEFAULT_REPO_ID="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
DEFAULT_MODEL_FILE="qwen2.5-coder-14b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_DIR="$MODEL_ROOT/gguf/qwen2.5-coder-14b-q4_k_m"

REPO_ID="${REPO_ID:-$DEFAULT_REPO_ID}"
MODEL_FILE="${MODEL_FILE:-$DEFAULT_MODEL_FILE}"

# Qwen2.5 Coder GGUF reports n_ctx_train=131072.
# Lower LLAMA_CTX_SIZE manually if your machine runs out of memory.
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-131072}"
LLAMA_MAX_TOKENS="${LLAMA_MAX_TOKENS:-8192}"
LLAMA_N_SEQ_MAX="${LLAMA_N_SEQ_MAX:-1}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:--1}"
LLAMA_CPP_CUDA="${LLAMA_CPP_CUDA:-1}"

DO_LOGIN=0
FORCE=0
FORCE_PYTHON=0

usage() {
  cat <<'EOF'
Usage:
  bash install_qwen25_coder_14b_gguf.sh [options]

Downloads a GGUF model from Hugging Face, installs Python dependencies into the
current python3 environment, and writes .llama.env for worker.py, tester.py,
and explainer.py.

Default model:
  Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
  qwen2.5-coder-14b-instruct-q4_k_m.gguf

Options:
  --model-url URL     Hugging Face model URL or file URL.
                      Example:
                        https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
                        https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/blob/main/qwen2.5-coder-14b-instruct-q4_k_m.gguf
  --repo REPO_ID      Hugging Face repo id, for example:
                        Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
  --file MODEL_FILE   GGUF file inside the repo.
  --login             Run interactive Hugging Face login.
  --force             Re-download the model file.
  --force-python      Reinstall llama-cpp-python.
  -h, --help          Show this usage.

Environment overrides:
  LLM_DIR="$HOME/llm" bash install_qwen25_coder_14b_gguf.sh
  MODEL_DIR="/path/to/model-dir" bash install_qwen25_coder_14b_gguf.sh
  LLAMA_CTX_SIZE=32768 bash install_qwen25_coder_14b_gguf.sh
  LLAMA_MAX_TOKENS=8192 bash install_qwen25_coder_14b_gguf.sh
  LLAMA_N_GPU_LAYERS=0 bash install_qwen25_coder_14b_gguf.sh
  LLAMA_N_SEQ_MAX=1 bash install_qwen25_coder_14b_gguf.sh
  LLAMA_CPP_CUDA=0 bash install_qwen25_coder_14b_gguf.sh
  PIP_INSTALL_SCOPE=--user bash install_qwen25_coder_14b_gguf.sh
EOF
}

set_hf_model_ref() {
  local ref="$1"
  ref="${ref%%\?*}"
  ref="${ref#https://huggingface.co/}"
  ref="${ref#http://huggingface.co/}"
  ref="${ref%/}"

  if [[ "$ref" == *"/blob/"* ]]; then
    REPO_ID="${ref%%/blob/*}"
    MODEL_FILE="${ref#*/blob/}"
    MODEL_FILE="${MODEL_FILE#*/}"
    return
  fi

  if [[ "$ref" == *"/resolve/"* ]]; then
    REPO_ID="${ref%%/resolve/*}"
    MODEL_FILE="${ref#*/resolve/}"
    MODEL_FILE="${MODEL_FILE#*/}"
    return
  fi

  if [[ "$ref" == *"/tree/"* ]]; then
    REPO_ID="${ref%%/tree/*}"
    return
  fi

  REPO_ID="$ref"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model-url)
      set_hf_model_ref "${2:?Missing value for --model-url}"
      shift 2
      ;;
    --repo)
      set_hf_model_ref "${2:?Missing value for --repo}"
      shift 2
      ;;
    --file)
      MODEL_FILE="${2:?Missing value for --file}"
      shift 2
      ;;
    --login)
      DO_LOGIN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --force-python)
      FORCE_PYTHON=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MODEL_SLUG="$(printf '%s/%s' "$REPO_ID" "$MODEL_FILE" | tr '/: @' '____' | tr -cd '[:alnum:]_.-')"
if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then
  MODEL_DIR="$MODEL_DIR_OVERRIDE"
elif [[ "$REPO_ID" == "$DEFAULT_REPO_ID" && "$MODEL_FILE" == "$DEFAULT_MODEL_FILE" ]]; then
  MODEL_DIR="$DEFAULT_MODEL_DIR"
else
  MODEL_DIR="$MODEL_ROOT/gguf/$MODEL_SLUG"
fi

log() {
  printf '\n\033[1;32m==>\033[0m %s\n' "$*"
}

warn() {
  printf '\n\033[1;33mWARNING:\033[0m %s\n' "$*" >&2
}

pip_install() {
  "$PYTHON_BIN" -m pip install $PIP_INSTALL_SCOPE "$@"
}

find_hf_bin() {
  local candidate
  for candidate in \
    "$(command -v hf || true)" \
    "$HOME/.local/bin/hf" \
    "$(command -v huggingface-cli || true)" \
    "$HOME/.local/bin/huggingface-cli"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

hf_login_with_token() {
  if [[ "$(basename "$HF_BIN")" == "huggingface-cli" ]]; then
    "$HF_BIN" login --token "$HF_TOKEN"
  else
    "$HF_BIN" auth login --token "$HF_TOKEN"
  fi
}

hf_login_interactive() {
  if [[ "$(basename "$HF_BIN")" == "huggingface-cli" ]]; then
    "$HF_BIN" login
  else
    "$HF_BIN" auth login
  fi
}

hf_download() {
  "$HF_BIN" download "$REPO_ID" "$MODEL_FILE" \
    --local-dir "$MODEL_DIR"
}

install_system_packages() {
  if command -v apt >/dev/null 2>&1; then
    log "Installing system packages"
    sudo apt update
    sudo apt install -y python3 python3-pip ca-certificates curl git build-essential cmake ninja-build
  else
    log "Skipping apt install because apt was not found"
  fi
}

runtime_has_gpu_offload() {
  [[ "$("$PYTHON_BIN" - <<'PY'
try:
    from llama_cpp import llama_cpp
    supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    print("yes" if supports is not None and supports() else "no")
except Exception:
    print("no")
PY
)" == "yes" ]]
}

install_python_packages() {
  log "Installing/upgrading Python packages in current python3 environment"
  "$PYTHON_BIN" -m pip install $PIP_INSTALL_SCOPE --upgrade pip wheel setuptools
  pip_install --upgrade "huggingface_hub[cli]"

  if [[ "$LLAMA_CPP_CUDA" -eq 0 ]]; then
    log "Installing llama-cpp-python without CUDA"
    pip_install --upgrade llama-cpp-python
    return
  fi

  if runtime_has_gpu_offload && [[ "$FORCE_PYTHON" -eq 0 ]]; then
    log "llama-cpp-python already reports GPU offload support"
    return
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    cat >&2 <<EOF

CUDA toolkit was not found: nvcc is missing.

Your NVIDIA driver is visible, but building llama-cpp-python with GPU support
requires the CUDA toolkit, not just the driver.

Install CUDA toolkit / nvcc, then rerun:
  bash install_qwen25_coder_14b_gguf.sh --force-python

If you intentionally want CPU-only mode:
  LLAMA_CPP_CUDA=0 bash install_qwen25_coder_14b_gguf.sh
EOF
    exit 1
  fi

  log "Building llama-cpp-python with CUDA"
  CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on}" \
  FORCE_CMAKE=1 \
    pip_install \
      --upgrade \
      --force-reinstall \
      --no-cache-dir \
      llama-cpp-python
}

install_system_packages
install_python_packages

log "Creating directories"
mkdir -p "$MODEL_DIR" "$MODEL_ROOT"

HF_BIN="$(find_hf_bin || true)"
if [[ -z "$HF_BIN" ]]; then
  echo "hf CLI was not found after installing huggingface_hub[cli]." >&2
  echo "Check that Python user scripts are on PATH, usually: $HOME/.local/bin" >&2
  exit 1
fi

log "hf version"
"$HF_BIN" --version || true

if [[ -n "${HF_TOKEN:-}" ]]; then
  log "Logging in to Hugging Face using HF_TOKEN"
  hf_login_with_token
elif [[ "$DO_LOGIN" -eq 1 ]]; then
  log "Starting interactive Hugging Face login"
  hf_login_interactive
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
  hf_download
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Download finished, but model file was not found:" >&2
  echo "$MODEL_PATH" >&2
  exit 1
fi

SIZE_BYTES="$("$PYTHON_BIN" -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$MODEL_PATH")"
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
  printf 'export LLAMA_N_SEQ_MAX=%q\n' "$LLAMA_N_SEQ_MAX"
  printf 'export LLAMA_N_GPU_LAYERS=%q\n' "$LLAMA_N_GPU_LAYERS"
} > "$LLAMA_ENV_FILE"

log "Done"
cat <<EOF

Model:
  Repo: $REPO_ID
  File: $MODEL_FILE
  $MODEL_PATH

Current symlink:
  $CURRENT_LINK -> $MODEL_PATH

Environment file:
  $LLAMA_ENV_FILE
  LLAMA_MODEL_PATH=$MODEL_PATH
  LLAMA_CTX_SIZE=$LLAMA_CTX_SIZE
  LLAMA_MAX_TOKENS=$LLAMA_MAX_TOKENS
  LLAMA_N_SEQ_MAX=$LLAMA_N_SEQ_MAX
  LLAMA_N_GPU_LAYERS=$LLAMA_N_GPU_LAYERS

Run:
  python3 explainer.py
  python3 worker.py
  python3 tester.py

Check CUDA offload support:
  python3 -c 'from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'

If full context is too heavy, rerun for example:
  LLAMA_CTX_SIZE=32768 bash install_qwen25_coder_14b_gguf.sh
EOF
