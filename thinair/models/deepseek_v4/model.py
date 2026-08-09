"""DeepSeek V4 (flash and pro) served over an OpenAI-compatible endpoint.

``structured_output="prompted"``: the ds4 server accepts ``response_format`` in
the payload and ignores it, which is worse than rejecting it -- the reply comes
back as prose and only the prompt's own instruction recovers the JSON.  So we
never send it.

**These are reasoning models, and they take their time.**  A single reading
routinely thinks for several minutes before the answer arrives -- that is the
model working, not the transport hanging.  Leave ``THINAIR_TIMEOUT`` at its
generous default (900 s), and set ``THINAIR_PROGRESS=1`` to watch each call
start and land instead of guessing.  Cutting the leash short is the classic
own goal: the read dies mid-thought, the retry restarts the same long thought,
and the ledger records a transport error where a reading belonged.

The two served ids are separate beliefs by construction: the model name is part
of every belief id (invariant 6), which is what makes flash-vs-pro agreement
computable from the ledger alone.
"""

from thinair.models import ModelDef

MODEL = ModelDef(
    match=("deepseek-v4", "deepseek-v4-*", "deepseek/deepseek-v4*"),
    defaults=dict(temperature=0.2, max_tokens=8192),
    structured_output="prompted",
    think=dict(on={"reasoning_effort": "high"}, off={"reasoning_effort": "low"}),
    request_extra={},
    quirks=(),
    prompt_dialect="default",
    version="1",
)
