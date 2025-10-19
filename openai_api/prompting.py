from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_PROMPT_PATH = Path(__file__).with_name("fewshot_instructions.txt")


def load_system_prompt(path: Optional[str] = None) -> str:
    """Load the system prompt from `openai_api/fewshot_instructions.txt`.

    If `path` is provided, it is used; otherwise we look for the file next to
    this module. If that is missing, we raise a clear error with guidance.
    """
    target = Path(path) if path else DEFAULT_PROMPT_PATH
    if not target.exists():
        alt_root = Path.cwd() / "openai_api" / "fewshot_instructions.txt"
        if alt_root.exists():
            target = alt_root
        else:
            raise FileNotFoundError(
                f"System prompt not found. Expected at: {DEFAULT_PROMPT_PATH} or {alt_root}"
            )
    return target.read_text(encoding="utf-8")


def build_user_prompt(problem_text: str) -> str:
    """Create the user prompt in the exact required form.

    Format:
    Problem: [problem statement]

    Solution:
    """
    return f"Problem: {problem_text.strip()}\n\nSolution:"
