# Benchmark Overview

This folder contains Omni problem JSON files where each file contains the gold-standard DAG-MATH formatted CoT that forms a directed acyclic graph (DAG). Files are named `problem_<id>_final.json` (e.g., `problem_0_final.json`). There are 2,894 JSON files in total.

## Top‑Level Schema
Each JSON file is a list with a single object describing the problem:
- problem_id: integer identifier of the problem.
- domain: list of strings describing the topic taxonomy.
- difficulty: numeric difficulty indicator from 1 (easiest) to 6 (hardest).
- problem_text: problem statement.
- sample_id: sample identifier for the solution trace.
- final_answer: final answer string.
- steps: list of step objects (see below).

## Step Schema (Normalized)
Each step object captures a node in the reasoning DAG plus its narrative:
- step_id: unique integer for the step within the problem (appears first).
- edge: inference from premises (previous nodes) to the conclusion in the node.
- direct_dependent_steps: null or list of `step_id` integers that this step directly depends on (parents).
- node: brief conclusion of the current step.

Key order within each step is normalized as: `step_id`, `edge`, `direct_dependent_steps`, `node`.

## Example Format
```json
[
  {
    "problem_id": 0,
    "problem_text": "...",
    "steps": [
      {
        "step_id": 4,
        "edge": "...",
        "direct_dependent_steps": [1,2],
        "node": "..."
      }
    ]
  }
]
```