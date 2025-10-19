from __future__ import annotations

import json
import time
from typing import Dict, Any, List, Optional, Tuple

import openai

from .models import build_text_format_schema, DAGSolution


def make_responses_batch_line(
    *,
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {"format": build_text_format_schema()},
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if temperature is not None:
        body["temperature"] = float(temperature)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def submit_batch_and_wait(
    batch_client: openai.OpenAI,
    jsonl_path: str,
    *,
    poll_interval_s: int = 5,
) -> Tuple[Optional[str], Optional[dict]]:
    with open(jsonl_path, "rb") as fh:
        file_obj = batch_client.files.create(file=fh, purpose="batch")

    batch = batch_client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    batch_id = batch.id
    print(f"Submitted batch {batch_id} with requests file {file_obj.id}.")

    while True:
        time.sleep(poll_interval_s)
        batch = batch_client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        print(f"Batch {batch_id} status: {status}")
        if status in {"completed", "failed", "expired", "canceled"}:
            break

    if getattr(batch, "status", None) != "completed":
        print(f"Batch {batch_id} did not complete successfully: {getattr(batch, 'status', None)}")
        return None, batch

    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id is None:
        output_ids = getattr(batch, "output_file_ids", None)
        if output_ids and isinstance(output_ids, list):
            output_file_id = output_ids[0]

    return output_file_id, batch


def download_file_text(api_client: openai.OpenAI, file_id: str) -> str:
    try:
        resp = api_client.files.content(file_id)
        if hasattr(resp, "read"):
            return resp.read().decode("utf-8")
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, (bytes, bytearray)):
            return resp.decode("utf-8")
        try:
            return b"".join(list(resp)).decode("utf-8")
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to download file content for {file_id}: {e}")
    return ""


def _extract_parsed_from_body(body: dict) -> Optional[dict]:
    if not isinstance(body, dict):
        return None
    # Preferred: output_parsed
    if "output_parsed" in body and isinstance(body["output_parsed"], dict):
        return body["output_parsed"]
    # Fallback to possible JSON in output content
    output = body.get("output")
    if isinstance(output, list) and output:
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "json" in first and isinstance(first["json"], dict):
                    return first["json"]
            if item.get("type") in {"output_text", "message"}:
                txt = item.get("text")
                if txt and isinstance(txt, str):
                    try:
                        return json.loads(txt)
                    except Exception:
                        pass
                c = item.get("content")
                if isinstance(c, list) and c and isinstance(c[0], dict):
                    t = c[0].get("text")
                    if t and isinstance(t, str):
                        try:
                            return json.loads(t)
                        except Exception:
                            pass
    txt = body.get("output_text")
    if txt and isinstance(txt, str):
        try:
            return json.loads(txt)
        except Exception:
            return None
    return None


def parse_batch_results(raw_jsonl_text: str) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for line in raw_jsonl_text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = rec.get("custom_id")
        body = None
        if isinstance(rec.get("response"), dict):
            body = rec["response"].get("body")
        if body is None and isinstance(rec.get("body"), dict):
            body = rec.get("body")
        parsed = _extract_parsed_from_body(body) if body else None
        if custom_id and parsed:
            results[custom_id] = parsed
    return results


def validate_solution(obj: dict) -> DAGSolution:
    """Validate a parsed output against the DAGSolution pydantic model."""
    return DAGSolution(**obj)
