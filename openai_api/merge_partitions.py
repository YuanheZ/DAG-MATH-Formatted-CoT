from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROBLEM_FILE_RE = re.compile(r"problem_(\d+)_all_samples\.json$")


def find_problem_ids(input_dirs: Sequence[Path]) -> List[int]:
    ids = set()
    for d in input_dirs:
        for p in d.glob("problem_*_all_samples.json"):
            m = PROBLEM_FILE_RE.search(p.name)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def load_samples(file_path: Path) -> List[dict]:
    try:
        text = file_path.read_text(encoding="utf-8")
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []


def merge_per_problem(
    *,
    problem_id: int,
    input_dirs: Sequence[Path],
    renumber: bool,
) -> List[dict]:
    merged: List[dict] = []
    for d in input_dirs:
        fp = d / f"problem_{problem_id}_all_samples.json"
        if not fp.exists():
            continue
        items = load_samples(fp)
        # Sort by sample_id for determinism within each source
        items.sort(key=lambda x: x.get("sample_id", 0))
        merged.extend(items)

    if renumber:
        for idx, item in enumerate(merged, start=1):
            item["sample_id"] = idx
            # Ensure problem_id field is set correctly
            item["problem_id"] = problem_id
    else:
        # still enforce consistent problem_id
        for item in merged:
            item["problem_id"] = problem_id

    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-problem sample arrays from multiple partitions into one file per problem.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input directories that contain problem_<id>_all_samples.json",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory where merged problem_<id>_all_samples.json will be written",
    )
    ap.add_argument(
        "--problems",
        default=None,
        help="Optional comma-separated list of problem ids to merge (default: auto-detect all)",
    )
    ap.add_argument(
        "--keep-sample-ids",
        action="store_true",
        help="Do not renumber sample_id; by default, sample_id is reassigned sequentially",
    )
    args = ap.parse_args()

    input_dirs = [Path(p).resolve() for p in args.inputs]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.problems:
        problems = sorted({int(x) for x in args.problems.split(",") if x.strip().isdigit()})
    else:
        problems = find_problem_ids(input_dirs)

    if not problems:
        raise SystemExit("No problems found to merge.")

    total_written = 0
    for pid in problems:
        merged = merge_per_problem(problem_id=pid, input_dirs=input_dirs, renumber=(not args.keep_sample_ids))
        if not merged:
            print(f"problem_{pid}: no samples found; skipping")
            continue
        out_path = outdir / f"problem_{pid}_all_samples.json"
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"problem_{pid}: wrote {len(merged)} samples -> {out_path}")
        total_written += 1

    print(f"Done. Wrote {total_written} merged problem files to {outdir}")


if __name__ == "__main__":
    main()
