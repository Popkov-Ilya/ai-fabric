# AI Fabric

AI Fabric is a small local LLM pipeline for turning a programming task into:

- a normalized task description,
- an explicit public API contract,
- an implementation in `output.py`,
- generated tests for that implementation.

The project is intentionally file-based and CLI-friendly. It can use either a
local GGUF model through `llama-cpp-python` or an LM Studio server through its
OpenAI-compatible API.

## Pipeline

```text
artifacts/input_task.txt
  -> explainer.py
      -> artifacts/task.txt
      -> artifacts/contract.json
  -> worker.py
      -> artifacts/output.py
      -> artifacts/actual_api.json
  -> tester.py
      -> artifacts/test_output.py
  -> repair.py
      -> artifacts/test_result.json
```

`explainer.py` prepares the task and creates the expected API contract.
`worker.py` implements the task and verifies that the generated code matches
the contract. `tester.py` writes tests from the same task and contract, instead
of adapting to whatever implementation happened to be generated. `repair.py`
runs those tests and appends failure diagnostics to `artifacts/task.txt` so the
next worker attempt has concrete feedback.

## Repository Files

- `explainer.py` reads `artifacts/input_task.txt`, fills `task_template.txt`,
  and writes `artifacts/task.txt` plus `artifacts/contract.json`.
- `worker.py` reads the task and contract, asks the LLM to implement
  `artifacts/output.py`, validates syntax, and writes `artifacts/actual_api.json`.
- `tester.py` reads the task and contract, then writes `artifacts/test_output.py`.
- `repair.py` runs generated tests and appends repair notes to the task when
  tests fail.
- `pipeline.py` runs the full sequence and repeats worker/tester/repair until
  tests pass or the repair limit is reached.
- `llm_backend.py` contains the LLM backends and environment loading.
- `task_template.txt` is the normalized task structure.
- `task_example.txt` shows the intended task format.
- `install_hf_gguf_model.sh` installs a default Hugging Face GGUF model setup
  for the `llama-cpp-python` backend.

## Artifacts

Generated and user-provided working files live under `artifacts/`:

- `input_task.txt`: raw task written by the user.
- `task.txt`: normalized task created by `explainer.py`.
- `contract.json`: expected public API created by `explainer.py`.
- `output.py`: generated implementation created by `worker.py`.
- `actual_api.json`: API extracted from `output.py` for conformance checking.
- `test_output.py`: generated tests created by `tester.py`.
- `test_result.json`: latest test command result created by `repair.py`.
- `test_stdout.txt` and `test_stderr.txt`: captured test output.
- `pipeline_result.json`: end-to-end stage report created by `pipeline.py`.

Generated outputs in `artifacts/` are ignored by Git; the small example input
files are kept in the repository.

## LLM Backend Options

Backend selection is controlled by `.llama.env` or environment variables.

### Option 1: LM Studio

Start the LM Studio local server on Windows, usually at
`http://127.0.0.1:1234/v1`, then create `.llama.env`:

```bash
export LLM_BACKEND=lm_studio
export LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
export LM_STUDIO_MODEL=local-model
export LM_STUDIO_MAX_TOKENS=8192
export LM_STUDIO_TEMPERATURE=0.1
export LM_STUDIO_TIMEOUT=600
```

`LM_STUDIO_MODEL` can usually be left as `local-model` for LM Studio. If you
run another OpenAI-compatible server, set it to the model id expected by that
server.

### Option 2: llama-cpp-python

Install dependencies manually:

```bash
python -m pip install -r requirements.txt
```

Then create `.llama.env`:

```bash
export LLM_BACKEND=llama_cpp
export LLAMA_MODEL_PATH=/path/to/model.gguf
export LLAMA_CTX_SIZE=131072
export LLAMA_MAX_TOKENS=8192
export LLAMA_N_SEQ_MAX=1
export LLAMA_N_GPU_LAYERS=-1
export LLAMA_TEMPERATURE=0.1
```

Or use the bundled installer on a compatible Unix-like shell:

```bash
bash install_hf_gguf_model.sh
```

## Usage

Create the artifacts directory and write a raw task:

```bash
mkdir -p artifacts
```

Put your task in `artifacts/input_task.txt`, then run:

```bash
python pipeline.py
```

Or run the stages manually:

```bash
python explainer.py
python worker.py
python tester.py
python repair.py
```

After that, inspect or run:

```bash
python artifacts/test_output.py
```

To reuse an existing `artifacts/task.txt` and `artifacts/contract.json`:

```bash
python pipeline.py --skip-explainer
```

To control repair attempts:

```bash
python pipeline.py --max-repairs 2 --test-timeout 60
```

## Contract Format

`artifacts/contract.json` uses this shape:

```json
{
  "module": "output",
  "source_file": "output.py",
  "functions": [
    {
      "name": "two_sum",
      "async": false,
      "arguments": [
        {
          "name": "nums",
          "kind": "positional_or_keyword",
          "required": true
        },
        {
          "name": "target",
          "kind": "positional_or_keyword",
          "required": true
        }
      ],
      "returns": "list[int]"
    }
  ]
}
```

`worker.py` checks that every contract function exists in `output.py` with the
same name, async flag, and argument structure. Extra helper functions are
allowed.

## Current Limitations

- Tests are run by `repair.py` and `pipeline.py`, but failures are repaired by
  appending diagnostic text to the task rather than editing code directly.
- `contract.json` is generated by the LLM, so it can still need human review.
- Contract conformance currently checks function shape, not type annotations or
  runtime behavior.
- The generated tests come from the same model/backend unless you switch the
  backend configuration between stages.
