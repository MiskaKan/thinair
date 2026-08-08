"""gpt-oss open-weight models served over an OpenAI-compatible endpoint.

Declares ``no_system_role``: the served chat template rejects a system turn, so
the transport folds the system message into the first user turn.
"""

from thinair.models import ModelDef

MODEL = ModelDef(
    match=("gpt-oss", "gpt-oss-*", "openai/gpt-oss*"),
    defaults=dict(temperature=0.3, max_tokens=8192),
    structured_output="json_mode",
    think=dict(on={"reasoning_effort": "high"}, off={"reasoning_effort": "low"}),
    request_extra={},
    quirks=("no_system_role",),
    prompt_dialect="default",
    version="1",
)
