"""Qwen3-35B family served over an OpenAI-compatible endpoint."""

from thinair.models import ModelDef

MODEL = ModelDef(
    match=("qwen3-35b", "qwen/qwen3-35b*", "qwen3-35b-*"),
    defaults=dict(temperature=0.2, max_tokens=4096),
    structured_output="json_schema",
    think=dict(on={"enable_thinking": True}, off={"enable_thinking": False}),
    request_extra={},
    quirks=(),
    prompt_dialect="default",
    version="1",
)
