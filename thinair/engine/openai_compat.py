"""One transport: OpenAI-compatible chat completions over stdlib ``urllib``.

No third-party dependencies.  A genuinely different wire format is a new
module here, selected by ``ModelDef.transport`` -- never a behavioral model
folder.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from . import EngineError

__all__ = ["OpenAICompatEngine", "engine_for"]

DEFAULT_BASE_URL = "http://localhost:8000/v1"

#: A reasoning model on a modest endpoint routinely thinks for longer than two
#: minutes before its answer comes back.  Cutting it off there is not a safety
#: margin: the read dies mid-thought, spends a transport retry, and the retry
#: starts the same long thought from scratch -- so a slow endpoint fails three
#: times as slowly as it would have succeeded once, and the ledger records a
#: transport error where a reading belonged.  Wait by default; ``THINAIR_TIMEOUT``
#: shortens the leash where a caller wants one.
DEFAULT_TIMEOUT = 900.0


class OpenAICompatEngine:
    """``complete(messages, schema, temperature, max_tokens) -> (text, meta)``."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "", definition: Any = None, timeout: float | None = None,
                 max_tokens: int | None = None) -> None:
        self.configured = bool(base_url or os.environ.get("THINAIR_BASE_URL"))
        self.base_url = (base_url or os.environ.get("THINAIR_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("THINAIR_API_KEY") or ""
        self.model = model
        self.definition = definition
        env_timeout = os.environ.get("THINAIR_TIMEOUT")
        self.timeout = float(
            timeout if timeout is not None
            else (env_timeout if env_timeout else DEFAULT_TIMEOUT))
        env_max = os.environ.get("THINAIR_MAX_TOKENS")
        self.max_tokens = max_tokens or (int(env_max) if env_max else None)
        self.transport_retries = 3

    # -- quirk handling ----------------------------------------------------
    def _shape_messages(self, messages: list[dict]) -> list[dict]:
        definition = self.definition
        out = [dict(m) for m in messages]
        if definition is not None and definition.has("no_system_role"):
            systems = [m["content"] for m in out if m.get("role") == "system"]
            out = [m for m in out if m.get("role") != "system"]
            if systems:
                head = "\n\n".join(systems)
                if out and out[0].get("role") == "user":
                    out[0]["content"] = head + "\n\n" + out[0]["content"]
                else:
                    out.insert(0, {"role": "user", "content": head})
        if definition is not None and definition.has("single_user_turn"):
            joined = "\n\n".join(f"[{m['role']}] {m['content']}" for m in out)
            out = [{"role": "user", "content": joined}]
        return out

    def _response_format(self, schema: dict | None) -> dict | None:
        definition = self.definition
        mode = definition.structured_output if definition is not None else "prompted"
        if definition is not None and definition.has("no_json_mode"):
            return None
        if schema is None or mode == "prompted":
            return None
        if mode == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "thinair_opinion", "strict": False,
                                    "schema": schema}}
        return {"type": "json_object"}

    # -- the one network call ---------------------------------------------
    def complete(self, messages: list[dict], schema: dict | None = None,
                 temperature: float = 0.2, max_tokens: int | None = None,
                 think: bool = False, **extra: Any) -> tuple[str, dict]:
        definition = self.definition
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._shape_messages(messages),
        }
        if definition is None or not definition.has("no_temperature"):
            payload["temperature"] = float(temperature)
        limit = max_tokens or self.max_tokens
        if definition is not None:
            limit = limit or definition.defaults.get("max_tokens")
        if limit:
            payload["max_tokens"] = int(limit)
        response_format = self._response_format(schema)
        if response_format:
            payload["response_format"] = response_format
        if definition is not None:
            payload.update(definition.request_extra)
            payload.update(definition.think_payload(think))
        payload.update({k: v for k, v in extra.items() if v is not None})

        started = time.time()
        raw = self._post(f"{self.base_url}/chat/completions", payload)

        try:
            choice = raw["choices"][0]["message"]
            text = choice.get("content") or ""
        except Exception:  # pragma: no cover - defensive
            raise EngineError(f"unexpected response shape: {str(raw)[:300]}") from None
        usage = raw.get("usage") or {}
        meta = {
            "model": raw.get("model") or self.model,
            "temperature": float(temperature),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency": round(time.time() - started, 4),
            "transport": "openai_compat",
        }
        return text, meta


    # The one place this package touches a socket.  It is a separate method so
    # that a test can substitute a recorder and still exercise payload
    # shaping, quirk handling and response parsing with zero network.
    def _post(self, url: str, payload: dict) -> dict:
        # Two guards, both *before* the socket, so a misfire is a one-line
        # stack trace instead of a silent hang and three slow retries:
        if os.environ.get("THINAIR_OFFLINE", "") not in ("", "0"):
            raise EngineError(
                f"THINAIR_OFFLINE is set: a model consultation was about to "
                f"spend a call ({url}, model {self.model!r}).  Serve the "
                f"cell from the record or by assignment, or unset "
                f"THINAIR_OFFLINE to allow the call.")
        if self.model in ("", "default") and not self.configured:
            raise EngineError(
                "no endpoint is configured: the belief resolved to model "
                "'default' against the fallback base URL, which is almost "
                "never a served model.  Set THINAIR_MODEL and "
                "THINAIR_BASE_URL (and THINAIR_API_KEY if the endpoint "
                "wants one), or name the model: model(\"deepseek-v4-flash\").")
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last: Exception | None = None
        for attempt in range(1, self.transport_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers,
                                             method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as fh:
                    return json.loads(fh.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:      # pragma: no cover - network
                detail = exc.read().decode("utf-8", "replace")[:400]
                last = EngineError(f"HTTP {exc.code} from {url}: {detail}")
                if exc.code < 500:
                    raise last from None
            except Exception as exc:                    # pragma: no cover - network
                last = EngineError(f"{type(exc).__name__} calling {url}: {exc}")
            if attempt < self.transport_retries:
                time.sleep(0.25 * attempt)
        raise last or EngineError(f"no completion from {url}")  # pragma: no cover


def engine_for(model: str, definition: Any = None, base_url: str | None = None,
               api_key: str | None = None, max_tokens: int | None = None) -> Any:
    """Pick a transport from ``ModelDef.transport``.  v1 ships exactly one."""
    transport = getattr(definition, "transport", "openai_compat")
    if transport != "openai_compat":
        raise EngineError(
            f"no transport named {transport!r} in v1; add one under thinair/engine/")
    return OpenAICompatEngine(base_url=base_url, api_key=api_key, model=model,
                              definition=definition, max_tokens=max_tokens)
