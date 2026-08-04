# Medical QA Selective-Revision Evaluation

This repository contains the evaluation code for a medical question-answering benchmark that studies answer revision under source-status and evidence-quality interventions.

The three experiment scripts used during development were consolidated into one evaluator. Model-specific loading behavior is kept in `model_registry.py`, while benchmark parsing, inference, metrics, resuming, auditing, and output generation are implemented in `evaluate.py`.

## Default behavior

The default model roster contains the ten unique models used in the experiments. Models are evaluated sequentially.

The following expensive conversational extensions are disabled by default:

- persistence/escalation follow-up: enable with `--run_multi_turn_persistence`
- delayed memory evaluation: enable with `--run_memory_tests`

Same-turn source-conflict and evidence-versus-authority tests remain enabled because they are part of the main evaluation protocol. Disable them with `--no_same_turn_conflict_tests`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some gated model repositories require prior access approval and an authenticated Hugging Face session:

```bash
export HF_TOKEN="..."
```

Do not place access tokens in committed shell scripts or command history. The evaluator redacts `--hf_token` from its saved run configuration, but using the `HF_TOKEN` environment variable is preferred.

## Dataset format

`--input_files` accepts comma-separated JSON or JSONL paths and shell-style globs. A schema example can be generated without loading a model:

```bash
python3 evaluate.py --write_template example_schema.jsonl
```

## Main run

```bash
DATASET_DIR="./data"
OUTPUT_DIR="./outputs"

python3 evaluate.py \
  --input_files "$DATASET_DIR/*.json" \
  --output_dir "$OUTPUT_DIR" \
  --inference_mode score \
  --batch_size 4 \
  --device cuda \
  --device_map auto \
  --resume
```

The same command is available as a reusable script:

```bash
DATASET_DIR="./data" OUTPUT_DIR="./outputs" ./scripts/run_evaluation.sh
```

Additional evaluator arguments can be appended to the script. For example, to run one model:

```bash
./scripts/run_evaluation.sh \
  --models "google/medgemma-27b-text-it"
```

To enable the optional conversational extensions:

```bash
./scripts/run_evaluation.sh \
  --run_multi_turn_persistence \
  --run_memory_tests \
  --memory_max_examples 300
```

## Validation-only run

Use `--validate_only` to parse the datasets, validate the benchmark structure, and write audit files without loading any model:

```bash
python3 evaluate.py \
  --input_files "./data/*.json" \
  --output_dir "./validation_output" \
  --validate_only
```

## Code checks

```bash
./scripts/validate_code.sh
```

This checks Python syntax, CLI defaults, the model registry, and schema-template generation. It does not download or execute model weights.

## Main outputs

Each model receives a separate output directory containing prediction rows, aggregate metrics, uncertainty summaries, audit artifacts, and failure-analysis files. `run_config.json` records the arguments and environment metadata needed to reconstruct a run. Resume mode identifies completed predictions by question and variant identifiers.

## Anonymous artifact checklist

Before uploading the repository:

1. Do not commit datasets, generated outputs, model caches, access tokens, local environment files, or cluster job logs.
2. Keep machine-specific paths outside the repository and pass them through environment variables.
3. Inspect the Git history, not only the current files, for names, email addresses, usernames, institutions, and private paths.
4. Create a fresh anonymous repository rather than reusing a repository with identifying commit metadata.
5. Add author and citation metadata only after the anonymous review period.
