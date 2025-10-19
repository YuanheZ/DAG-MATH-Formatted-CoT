# openai_api

Small utilities for running OpenAI Batch API jobs that return strict, structured outputs for DAG‑style math solutions.

- Reads the system prompt from `openai_api/fewshot_instructions.txt`.
- Uses the Responses API (`/v1/responses`) with `text.format` set to a strict JSON Schema derived from pydantic models.

## Quick Start

1. Prepare an inputs JSONL where each line has:

```json
{"id": "123", "problem": "<problem text>"}
```

2. Run the batch job end‑to‑end:

```bash
python -m openai_api.run_batch --input path/to/problems.jsonl --outdir out --model o4-mini
```

Environment: set `OPENAI_API_KEY` or pass `--api-key`.

Outputs for the generic runner are written to `--outdir` as `<id>.json`, each validated against the schema.

## Files

- `models.py`: pydantic models and JSON Schema builder (strict, no extras).
- `prompting.py`: loads `fewshot_instructions.txt` and builds per‑item user prompts.
- `batch_utils.py`: JSONL line builder, batch submit/poll, download, parse, validate.
- `run_batch.py`: CLI wrapper to tie it all together.

## Run on a HF Dataset

Create N samples per problem for a single dataset (each dataset here has 30 problems):

```bash
python -m openai_api.run_hf_dataset_batch \
  --dataset MathArena/aime_2025 \
  --split test \
  --samples-per-problem 3 \
  --outdir out/aime_2025 \
  --model o4-mini \
  --retry-mode max --max-retries 2
```

Outputs and notes:
- System prompt is read from `openai_api/fewshot_instructions.txt`.
- User prompt format is exactly: `Problem: [problem]\n\nSolution:`.
- Structured output schema:
  - `SolutionStep { step_id: int, thinking: str, direct_dependent_steps: list[int] | null, text: str }`
  - `DAGSolution { steps: list[SolutionStep] }`
- If `direct_dependent_steps` is empty, it is saved as `null` in JSON as required.
- Retries: `--retry-mode until-success` keeps resubmitting failures until all succeed; `--retry-mode max --max-retries K` caps attempts.
- Per-problem merged files (matching the example style) are written to:
  - `<outdir>/<dataset_underscored>/problem_<problem_id>_all_samples.json`
  - Each file is a JSON array; elements have fields: `problem_id`, `sample_id`, `problem_text`, `steps`, `final_answer`, and `metadata { reasoning_content, model }`.
