from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from datasets import load_dataset

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
except ImportError:  # allow running as a script: python openai_api/run_hf_dataset_batch.py
    import sys
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


def extract_problem_text(example: Dict[str, Any]) -> Optional[str]:
    """Attempt to extract the problem statement from a dataset example.

    We try a set of common field names used in math datasets.
    """
    candidates = [
        "problem",
        "question",
        "prompt",
        "content",
        "text",
        "problem_text",
        "Problem",
    ]
    for key in candidates:
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # As a last resort, try to build from choices
    return None


def build_lines_for_dataset(
    *,
    ds_name: str,
    split: str,
    samples_per_problem: int,
    system_prompt: str,
    model: str,
    reasoning_effort: Optional[str],
    temperature: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (jsonl_lines, index_records).

    index_records: list of dicts with keys: cid, dataset, index, sample
    """
    dataset = load_dataset(ds_name, split=split)
    lines: List[Dict[str, Any]] = []
    index_records: List[Dict[str, Any]] = []
    for i, ex in enumerate(dataset):
        problem = extract_problem_text(ex)
        if not problem:
            continue
        user_prompt = build_user_prompt(problem)
        for s in range(samples_per_problem):
            cid = f"ds:{ds_name}|idx:{i}|sample:{s+1}"
            req = make_responses_batch_line(
                custom_id=cid,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
            lines.append(req)
            index_records.append({
                "cid": cid,
                "dataset": ds_name,
                "index": i,
                "sample": s + 1,
                "problem_text": problem,
            })
    return lines, index_records


def run_batches_with_retries(
    *,
    lines: List[Dict[str, Any]],
    index_records: List[Dict[str, Any]],
    outdir: Path,
    retry_mode: str,
    max_retries: int,
    poll_interval: int,
) -> Dict[str, Dict[str, Any]]:
    """Submit batch(es), retry failed items per settings, return cid->parsed dicts.

    retry_mode: "until-success" or "max"
    """
    client = openai.OpenAI()
    pending = [r["cid"] for r in index_records]
    by_cid = {r["cid"]: r for r in index_records}

    retry_counts: Dict[str, int] = {}
    results: Dict[str, Dict[str, Any]] = {}
    log_path = outdir / "retry_log.txt"
    outdir.mkdir(parents=True, exist_ok=True)

    def log_retry(cid: str, reason: str):
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{cid}\t{retry_counts.get(cid,0)}\t{reason}\n")

    while pending:
        # write a temp JSONL for current pending set
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmpf:
            jsonl_path = tmpf.name
            for cid in pending:
                # find the original request template by cid
                # We rebuild from `lines` for simplicity
                # (lines are small for 30*N)
                for ln in lines:
                    if ln.get("custom_id") == cid:
                        tmpf.write(json.dumps(ln) + "\n")
                        break

        output_file_id, _batch = submit_batch_and_wait(client, jsonl_path, poll_interval_s=poll_interval)
        try:
            os.remove(jsonl_path)
        except Exception:
            pass

        if not output_file_id:
            # whole batch failed; bump retries for all and possibly stop
            next_pending: List[str] = []
            for cid in pending:
                retry_counts[cid] = retry_counts.get(cid, 0) + 1
                log_retry(cid, "batch_failed")
                if retry_mode == "until-success" or retry_counts[cid] <= max_retries:
                    next_pending.append(cid)
            if not next_pending:
                break
            pending = next_pending
            continue

        raw = download_file_text(client, output_file_id)
        if not raw:
            # treat as batch failure
            next_pending = []
            for cid in pending:
                retry_counts[cid] = retry_counts.get(cid, 0) + 1
                log_retry(cid, "output_download_failed")
                if retry_mode == "until-success" or retry_counts[cid] <= max_retries:
                    next_pending.append(cid)
            if not next_pending:
                break
            pending = next_pending
            continue

        parsed_map = parse_batch_results(raw)

        # Decide per-cid success/failure
        next_pending: List[str] = []
        for cid in pending:
            obj = parsed_map.get(cid)
            if not obj:
                retry_counts[cid] = retry_counts.get(cid, 0) + 1
                log_retry(cid, "missing_or_malformed")
                if retry_mode == "until-success" or retry_counts[cid] <= max_retries:
                    next_pending.append(cid)
                continue
            # normalize [] -> null
            obj = coerce_empty_dependencies_to_null(obj)
            # validate
            try:
                _ = validate_solution(obj)
            except Exception as e:
                retry_counts[cid] = retry_counts.get(cid, 0) + 1
                log_retry(cid, f"schema_error: {str(e).splitlines()[0]}")
                if retry_mode == "until-success" or retry_counts[cid] <= max_retries:
                    next_pending.append(cid)
                continue
            results[cid] = obj
        pending = next_pending

        if retry_mode == "max" and not pending:
            break

        if retry_mode == "max" and pending:
            # check if any pending exceed max retries
            still_pending = []
            for cid in pending:
                if retry_counts.get(cid, 0) >= max_retries:
                    # give up
                    continue
                still_pending.append(cid)
            pending = still_pending

        if retry_mode == "until-success" and pending:
            # keep looping until all succeed
            pass

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Batch API on a HF dataset with N samples/problem and structured outputs")
    ap.add_argument("--dataset", required=True, help='Hugging Face dataset name, e.g., "MathArena/aime_2025"')
    ap.add_argument("--split", default="train", help="Dataset split (default: train)")
    ap.add_argument("--samples-per-problem", type=int, required=True, help="N samples for each problem")
    ap.add_argument("--outdir", required=True, help="Output directory for results and logs")
    ap.add_argument("--model", default="o4-mini", help="Model for /v1/responses")
    ap.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"], help="Optional reasoning effort setting")
    ap.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature")
    ap.add_argument("--retry-mode", choices=["until-success", "max"], default="max", help="Retry strategy")
    ap.add_argument("--max-retries", type=int, default=2, help="Max retries when --retry-mode=max")
    ap.add_argument("--poll-interval", type=int, default=30, help="Batch status poll interval seconds")
    ap.add_argument("--api-key", default=None, help="OpenAI API key (overrides env)")
    ap.add_argument("--system-prompt-path", default=None, help="Override path to fewshot_instructions.txt")
    args = ap.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(args.system_prompt_path)

    # Build requests
    lines, idx = build_lines_for_dataset(
        ds_name=args.dataset,
        split=args.split,
        samples_per_problem=int(args.samples_per_problem),
        system_prompt=system_prompt,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
    )
    if not lines:
        raise SystemExit("No requests were built (dataset empty or unrecognized fields).")

    # Submit + retries
    cid_to_obj = run_batches_with_retries(
        lines=lines,
        index_records=idx,
        outdir=outdir,
        retry_mode=args.retry_mode,
        max_retries=int(args.max_retries),
        poll_interval=int(args.poll_interval),
    )

    total_expected = len(idx)
    total_ok = len(cid_to_obj)
    all_ok = (total_ok == total_expected)

    # Build per-problem merged arrays in the same style as
    # api/results/qwen3_30b_moe/<dataset>/problem_<id>_all_samples.json
    per_problem: Dict[int, List[Dict[str, Any]]] = {}
    expected_per_problem: Dict[int, int] = {}
    for meta in idx:
        expected_per_problem[meta["index"]] = expected_per_problem.get(meta["index"], 0) + 1
        cid = meta["cid"]
        obj = cid_to_obj.get(cid)
        if not obj:
            continue
        steps = obj.get("steps") or []
        final_ans = None
        if isinstance(steps, list) and steps:
            last_text = steps[-1].get("text") if isinstance(steps[-1], dict) else None
            if isinstance(last_text, str) and last_text.strip():
                final_ans = last_text
        sample_entry = {
            "problem_id": meta["index"],
            "sample_id": meta["sample"],
            "problem_text": meta.get("problem_text", ""),
            "steps": steps,
            "final_answer": final_ans,
            "metadata": {
                "reasoning_content": "",
                "model": args.model,
            },
        }
        per_problem.setdefault(meta["index"], []).append(sample_entry)

    dataset_dir = outdir / args.dataset.replace('/', '_')
    dataset_dir.mkdir(parents=True, exist_ok=True)

    complete = 0
    incomplete = []
    for pid, samples in per_problem.items():
        if len(samples) == expected_per_problem.get(pid, 0):
            # sort by sample_id for determinism
            samples.sort(key=lambda x: x.get("sample_id", 0))
            out_path = dataset_dir / f"problem_{pid}_all_samples.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            complete += 1
        else:
            incomplete.append((pid, len(samples), expected_per_problem.get(pid, 0)))

    if incomplete:
        print(f"Wrote {complete} complete problem files; {len(incomplete)} incomplete (missing samples).")
        print("Incomplete problems:")
        for pid, have, need in incomplete:
            print(f"  problem_{pid}: {have}/{need} samples available")
    else:
        print(f"All problems complete. Wrote {complete} files to {dataset_dir}")


if __name__ == "__main__":
    main()
