from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import openai

try:
    from .prompting import load_system_prompt, build_user_prompt
    from .batch_utils import (
        make_responses_batch_line,
        submit_batch_and_wait,
        download_file_text,
        parse_batch_results,
        validate_solution,
    )
    from .models import coerce_empty_dependencies_to_null
except ImportError:  # allow running as a script: python openai_api/run_batch.py
    import sys, os
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from openai_api.prompting import load_system_prompt, build_user_prompt
    from openai_api.batch_utils import (
        make_responses_batch_line,
        submit_batch_and_wait,
        download_file_text,
        parse_batch_results,
        validate_solution,
    )
    from openai_api.models import coerce_empty_dependencies_to_null


def _load_inputs_jsonl(path: str) -> List[Dict[str, Any]]:
    """Expect each line: {"id": <str|int>, "problem": <str>}.

    Returns a list of dictionaries with canonical keys: id(str), problem(str).
    """
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Accept a few variants for convenience
            _id = rec.get("id")
            if _id is None:
                _id = rec.get("problem_id")
            problem = rec.get("problem")
            if problem is None:
                # If user provided AoS messages, try to extract
                msgs = rec.get("messages")
                if isinstance(msgs, list) and msgs:
                    # use first user message content as the problem
                    for m in msgs:
                        if isinstance(m, dict) and m.get("role") == "user":
                            problem = m.get("content")
                            break
            if _id is None or problem is None:
                # Skip malformed lines but keep going
                continue
            items.append({"id": str(_id), "problem": str(problem)})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OpenAI Batch for DAG-MATH structured outputs")
    ap.add_argument("--input", required=True, help="Path to JSONL with fields: id, problem")
    ap.add_argument("--outdir", required=True, help="Directory to write per-id JSON outputs")
    ap.add_argument("--model", default="o4-mini", help="OpenAI model for /v1/responses")
    ap.add_argument("--system-prompt-path", default=None, help="Override path for fewshot_instructions.txt")
    ap.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"], help="Optional reasoning effort hint")
    ap.add_argument("--api-key", default=None, help="OpenAI API key (overrides env if provided)")
    ap.add_argument("--poll-interval", type=int, default=5, help="Poll interval seconds for batch status")
    args = ap.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(args.system_prompt_path)
    rows = _load_inputs_jsonl(args.input)
    if not rows:
        raise SystemExit("No valid input records found in --input JSONL.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmpf:
        jsonl_path = tmpf.name
        for row in rows:
            cid = f"problem_{row['id']}"
            user_prompt = build_user_prompt(row["problem"])
            req = make_responses_batch_line(
                custom_id=cid,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            tmpf.write(json.dumps(req) + "\n")

    client = openai.OpenAI()
    output_file_id, batch = submit_batch_and_wait(client, jsonl_path, poll_interval_s=int(args.poll_interval))

    try:
        os.remove(jsonl_path)
    except Exception:
        pass

    if not output_file_id:
        raise SystemExit("Batch did not complete successfully; see logs above.")

    raw = download_file_text(client, output_file_id)
    if not raw:
        raise SystemExit("Failed to download batch output content.")

    results = parse_batch_results(raw)
    if not results:
        raise SystemExit("No parsable results found in batch output.")

    # Validate and write per-id files
    for row in rows:
        cid = f"problem_{row['id']}"
        parsed = results.get(cid)
        if not parsed:
            continue
        try:
            parsed = coerce_empty_dependencies_to_null(parsed)
            _sol = validate_solution(parsed)
        except Exception as e:
            print(f"{cid}: schema validation failed: {e}")
            continue
        out_path = Path(args.outdir) / f"{row['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
