"""The DeepSeek family.

DeepSeek's reasoning models think by default and report it in the
standard `reasoning_content` channel, which the base engine already
reads — so a single knob is the whole family: the chat-template kwarg
that turns the thinking off for the direct tier. Sampling is left to the
base; measured against DeepSeek-V4-Flash-0731 (an IQ3_XXS quant on
llama.cpp), a wider nucleus and a presence penalty changed neither the
length of the reasoning nor whether the answer arrived, so this family
claims nothing it cannot show.

One deployment note. Served by llama.cpp, this family's template wants
`enable_thinking`; DeepSeek's own vLLM/SGLang recipes for V3.1 and later
use `{"thinking": False}` instead. They are not interchangeable here —
`thinking: False` leaves a stray `</think>` in the answer channel and,
once a JSON grammar is attached, fails the sampler outright (HTTP 400).
A deployment that wants the other spelling overrides `direct_payload`
with that one line.
"""

from __future__ import annotations

from .. import Engine


class DeepSeekEngine(Engine):
    """DeepSeek models on an OpenAI-compatible server. Selected for any
    model name containing "deepseek" — including the served path of a
    GGUF quant, which is how llama.cpp names them."""

    default_model = "DeepSeek-V4-Flash-0731"
    model_names = ("deepseek",)

    def direct_payload(self, payload):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    def query_envelope(self, value_schema=None):
        """The reply contract, minus the cases this family cannot decode
        under a grammar.

        An open slot — the `{}` a schema uses for "any JSON value", which
        is what an untyped read asks for — collapses this model on the
        direct tier: measured over five samples of one attribute read, a
        constrained envelope produced an empty object or a second
        envelope nested inside `value` five times out of five, while the
        same prompt under plain JSON mode answered correctly five times
        out of five. Returning None here drops the constraint and leaves
        the prompt to steer, which it does well.

        Only this envelope, and only when a slot is genuinely open: with
        a typed slot the grammar costs nothing (four of four either way,
        on a boolean read and a string collapse), and the plan protocol's
        action envelope — open by nature, since a `set` may carry any
        value — measurably keeps the model inside the protocol, so it is
        left constrained."""
        envelope = super().query_envelope(value_schema)
        return None if _has_open_slot(envelope) else envelope


def _has_open_slot(schema):
    """True when any part of a JSON Schema is the empty schema `{}` —
    the one that constrains nothing."""
    if isinstance(schema, dict):
        return not schema or any(_has_open_slot(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_open_slot(v) for v in schema)
    return False
