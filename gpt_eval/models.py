from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel


class SolutionStep(BaseModel):
    step_id: int
    thinking: str
    # Allow null (None) or an array of ints
    direct_dependent_steps: Optional[List[int]]
    text: str


class DAGSolution(BaseModel):
    steps: List[SolutionStep]


def _enforce_required_all_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Adds `required` for every object node that has properties recursively."""

    def _recurse(node: Dict[str, Any]):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            props = node.get("properties")
            if isinstance(props, dict):
                node["required"] = list(props.keys())
                for v in props.values():
                    _recurse(v)
        items = node.get("items")
        if isinstance(items, dict):
            _recurse(items)
        elif isinstance(items, list):
            for it in items:
                _recurse(it)
        for key in ("oneOf", "anyOf", "allOf"):
            arr = node.get(key)
            if isinstance(arr, list):
                for el in arr:
                    _recurse(el)
        for defs_key in ("$defs", "definitions"):
            defs = node.get(defs_key)
            if isinstance(defs, dict):
                for v in defs.values():
                    _recurse(v)

    _recurse(schema)
    return schema


def _enforce_no_additional_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Sets `additionalProperties: false` for all object nodes recursively."""

    def _recurse(node: Dict[str, Any]):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
            props = node.get("properties")
            if isinstance(props, dict):
                for v in props.values():
                    _recurse(v)
        items = node.get("items")
        if isinstance(items, dict):
            _recurse(items)
        elif isinstance(items, list):
            for it in items:
                _recurse(it)
        for key in ("oneOf", "anyOf", "allOf"):
            arr = node.get(key)
            if isinstance(arr, list):
                for el in arr:
                    _recurse(el)
        for defs_key in ("$defs", "definitions"):
            defs = node.get(defs_key)
            if isinstance(defs, dict):
                for v in defs.values():
                    _recurse(v)

    _recurse(schema)
    return schema


def build_text_format_schema() -> Dict[str, Any]:
    """Returns a Responses API text.format schema dict for strict structured outputs.

    The returned object is suitable for:

    body = {
        "model": ...,
        "input": [...],
        "text": {"format": build_text_format_schema()},
    }
    """

    schema = DAGSolution.model_json_schema()
    schema = _enforce_required_all_properties(schema)
    schema = _enforce_no_additional_properties(schema)
    return {
        "type": "json_schema",
        "name": "DAGSolution",
        "schema": schema,
        "strict": True,
    }


def coerce_empty_dependencies_to_null(obj: Dict[str, Any]) -> Dict[str, Any]:
    """In-place style normalization: [] -> null for direct_dependent_steps.

    Returns the same dict reference for convenience.
    """
    if not isinstance(obj, dict):
        return obj
    steps = obj.get("steps")
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict):
                dds = s.get("direct_dependent_steps")
                if isinstance(dds, list) and len(dds) == 0:
                    s["direct_dependent_steps"] = None
    return obj
