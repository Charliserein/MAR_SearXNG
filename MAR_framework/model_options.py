"""Shared Qwen request settings; keep generation bounded and JSON-compatible."""
import json
import os


def model_options():
    extra = json.loads(os.getenv(
        "MAR_MODEL_EXTRA_BODY", '{"chat_template_kwargs":{"enable_thinking":false}}'))
    if not isinstance(extra, dict):
        raise ValueError("MAR_MODEL_EXTRA_BODY must be a JSON object")
    maximum = int(os.getenv("MAR_MODEL_MAX_TOKENS", "2048"))
    if maximum < 1:
        raise ValueError("MAR_MODEL_MAX_TOKENS must be positive")
    return {"extra_body": extra, "max_tokens": maximum, "timeout": 180,
            "max_retries": 1}
