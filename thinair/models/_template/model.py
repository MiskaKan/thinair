"""Copy-me starting point for a new model folder.

Recipe:
    1. copy ``models/_template`` to ``models/<your_model>``
    2. rename nothing else; fill in ``MODEL`` below
    3. run ``pytest tests/test_models.py``
Done.

This folder is skipped by auto-discovery (its name starts with ``_``), so an
unfinished copy never breaks the registry.
"""

from thinair.models import ModelDef

MODEL = ModelDef(
    # Names and globs this def claims.  Must not overlap another built-in def.
    match=("my-model", "vendor/my-model*"),
    # Request defaults; the belief id records the temperature actually used.
    defaults=dict(temperature=0.2, max_tokens=4096),
    # "json_schema" | "json_mode" | "prompted"
    structured_output="prompted",
    # Payloads that switch a thinking mode on/off, if the server has one.
    think=dict(on={}, off={}),
    # Server-specific payload knobs merged into every request.
    request_extra={},
    # Declared, handled by the transport.  See models.KNOWN_QUIRKS.
    quirks=(),
    # A named override set in engine/prompts.py.
    prompt_dialect="default",
    # Part of every belief id: retuning this folder mints fresh ids (invariant 6).
    version="1",
)
