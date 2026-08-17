"""One bounded OpenAI-compatible request path for PAWEval."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


MAX_RESPONSE_BYTES = 1_000_000


@dataclass(frozen=True)
class VLMConfig:
    base_url: str
    model: str
    api_key_env: str
    timeout_s: float = 120.0
    max_tokens: int = 2048
    attempts: int = 2


class ProviderFailure(RuntimeError):
    """The configured VLM could not provide one structured response."""


def complete(config: VLMConfig, messages: list[dict[str, Any]]) -> str:
    """Request one completion with a fixed two-attempt limit."""

    key = os.environ.get(config.api_key_env)
    if not key:
        raise ProviderFailure("missing_credentials")
    if config.attempts < 1 or config.timeout_s <= 0:
        raise ProviderFailure("invalid_provider_config")
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "Return one valid PAWEval JSON object and no prose."},
            {"role": "user", "content": messages},
        ],
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_error = "provider_failed"
    for attempt in range(config.attempts):
        try:
            return _post(config, key, payload)
        except ProviderFailure as exc:
            last_error = str(exc)
            if last_error not in {"timeout", "network_error", "rate_limited", "server_error"}:
                break
            if attempt < config.attempts - 1:
                time.sleep(1.0)
    raise ProviderFailure(last_error)


def _post(config: VLMConfig, api_key: str, payload: dict[str, Any]) -> str:
    url = config.base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except TimeoutError as exc:
        raise ProviderFailure("timeout") from exc
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderFailure("rate_limited") from exc
        if exc.code >= 500:
            raise ProviderFailure("server_error") from exc
        raise ProviderFailure("client_error") from exc
    except urllib.error.URLError as exc:
        raise ProviderFailure("network_error") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProviderFailure("response_too_large")
    try:
        response = json.loads(body)
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ProviderFailure("malformed_provider_response") from exc
    if not isinstance(content, str):
        raise ProviderFailure("malformed_provider_response")
    return content
