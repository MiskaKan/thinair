"""The Qwen family — thinair's default engine.

Everything here is what Qwen wants specifically; everything it does not
override it inherits from the base `Engine`, which speaks plain
OpenAI-compatible JSON. Three knobs, and that is the whole family:

- the default model name a bare `Thing` talks to,
- `enable_thinking: False` in the chat template — the knob behind the
  direct tier, and the reason `_think_toggle` has anything to toggle,
- the sampling Qwen recommends for each of its two modes.
"""

from __future__ import annotations

from .. import Engine


class QwenEngine(Engine):
    """Qwen3-family models on an OpenAI-compatible server (vLLM, SGLang,
    llama.cpp and friends). Selected for any model name containing
    "qwen"; also the family whose model name is the package default."""

    default_model = "Qwen3.6-35B-A3B-oQ8-mtp"
    model_names = ("qwen",)

    def direct_payload(self, payload):
        # Qwen-recommended sampling for non-thinking mode
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["top_p"] = 0.8

    def thinking_payload(self, payload):
        # thinking mode: wider nucleus, and a presence penalty so a
        # quantized model is less tempted to circle its own thoughts
        payload["top_p"] = 0.95
        payload["presence_penalty"] = 1.0
