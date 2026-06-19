from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 90
    max_retries: int = 3

    def chat(self, *, model: str, messages: list[dict], temperature: float = 0.2, max_tokens: int | None = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        token_limit = _resolve_max_tokens(max_tokens)
        payload = {"model": model, "temperature": temperature, "messages": messages}
        if token_limit is not None:
            payload["max_tokens"] = token_limit
        body = json.dumps(payload).encode("utf-8")
        return _post_json_with_retries(
            url,
            body=body,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )["choices"][0]["message"]["content"]

    def embed(self, *, model: str, text: str) -> list[float]:
        url = self.base_url.rstrip("/") + "/embeddings"
        body = json.dumps({"model": model, "input": text}).encode("utf-8")
        data = _post_json_with_retries(
            url,
            body=body,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        return list(map(float, data["data"][0]["embedding"]))


def _resolve_max_tokens(max_tokens: int | None) -> int | None:
    raw = max_tokens if max_tokens is not None else os.getenv("OPENAI_MAX_TOKENS", "900")
    try:
        value = int(raw)
    except Exception:
        value = 900
    if value <= 0:
        return None
    return value


def _post_json_with_retries(
    url: str,
    *,
    body: bytes,
    api_key: str,
    timeout_s: int,
    max_retries: int,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 401: bad key; 429: rate limit/quota; 5xx: transient.
            if e.code in (401, 403):
                raise RuntimeError(
                    "OpenAI: non autorizzato (401/403). Verifica che OPENAI_API_KEY sia corretta e attiva."
                ) from e
            if e.code == 429 or 500 <= e.code <= 599:
                last_err = e
                if attempt >= max_retries:
                    break
                retry_after = 0.0
                try:
                    ra = e.headers.get("Retry-After")
                    if ra:
                        retry_after = float(ra)
                except Exception:
                    retry_after = 0.0
                # Exponential backoff with a floor; honor Retry-After when provided.
                sleep_s = max(retry_after, 1.0 * (2**attempt))
                time.sleep(min(sleep_s, 20.0))
                continue
            # Other HTTP errors: raise with body for debugging.
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"OpenAI HTTP error {e.code}. {detail}".strip()) from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(min(1.0 * (2**attempt), 10.0))
            continue

    raise RuntimeError("OpenAI: troppe richieste (429) o errore temporaneo. Riprova tra poco.") from last_err
