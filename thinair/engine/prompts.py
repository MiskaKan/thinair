"""All prompt construction, isolated and testable.

Template versions are part of every ``ModelBelief`` id (invariant 6): editing a
prompt here without bumping the version would silently re-label one belief's
opinions as another's.  Don't.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["ATTRIBUTE_TEMPLATE", "EPISODE_TEMPLATE", "DIALECTS",
           "attribute_messages", "episode_messages", "episode_observation",
           "render_snapshot", "response_schema", "episode_schema",
           "template_version"]

#: Bump on *any* edit to the corresponding builder (invariant 6).
ATTRIBUTE_TEMPLATE = "extract-v3"
EPISODE_TEMPLATE = "episode-v1"

TRUNCATION_MARKER = " …[truncated]"
MAX_RENDER = 2000


def template_version(kind: str = "attribute") -> str:
    return EPISODE_TEMPLATE if kind == "episode" else ATTRIBUTE_TEMPLATE


DIALECTS = {
    "default": {},
    "terse": {"preamble": "Answer with JSON only."},
}


def _dialect(name: str) -> dict:
    return DIALECTS.get(name, DIALECTS["default"])


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _short(value: Any, limit: int = MAX_RENDER) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + TRUNCATION_MARKER
    return text


def render_snapshot(e: Any, limit: int = MAX_RENDER) -> str:
    """The sealed snapshot as text.  State only -- never a transcript."""
    lines: list[str] = []
    # Everything framework-side is a dunder so that no domain attribute is
    # ever shadowed.
    lines.append(f"entity: {e.__entity__} ({e.__class_name__})")
    purpose = e.__purpose__
    if purpose:
        lines.append(f'purpose: "{purpose.strip()}"')
    if e.__value__ is not None or e.__p__:
        lines.append(f"this value: {_short(e.__value__, limit)}  (p={e.__p__:.2f})")
    if e.__provenance__:
        lines.append("provenance: " + "; ".join(e.__provenance__))
    frozen, believed = [], []
    for attr in e.__attrs__():
        opinion = e.__opinion__(attr)
        if opinion is None:
            continue
        rendered = f"  {attr} = {_short(opinion.value, limit)}"
        if opinion.frozen:
            frozen.append(rendered + "   (given)")
        else:
            believed.append(rendered + f"   (p={opinion.p:.2f}, per {opinion.belief})")
    if frozen:
        lines.append("given attributes:")
        lines.extend(frozen)
    if believed:
        lines.append("believed attributes:")
        lines.extend(believed)
    contracts = e.__contracts__
    if contracts:
        lines.append("declared attributes (contracts):")
        for name, contract in contracts.items():
            lines.append(f"  {name}: {contract.describe()}")
        undeclared = [a for a in contracts if e.__opinion__(a) is None]
        if undeclared:
            lines.append("  (not yet determined: " + ", ".join(undeclared) + ")")
    if e.__call_arguments__:
        lines.append("called with:")
        arguments = e.__call_arguments__
        if isinstance(arguments, dict):
            for name, value in arguments.items():
                lines.append(f"  {name} = {_short(value, limit)}")
        else:
            for i, value in enumerate(arguments):
                lines.append(f"  #{i} = {_short(value, limit)}")
    if e.__methods__:
        lines.append("real methods you may call:")
        for signature in e.__methods__:
            lines.append(f"  {signature}")
    for name, sub in (e.__arguments__ or {}).items():
        lines.append(f"argument {name}:")
        lines.append("  " + render_snapshot(sub, limit=limit // 2).replace("\n", "\n  "))
    return "\n".join(lines)


def _objections(objections: list[dict]) -> list[str]:
    """Prior rounds' vetoes, quoted verbatim."""
    out = []
    for i, objection in enumerate(objections, start=1):
        out.append(
            f"Attempt {i} proposed {_short(objection.get('value'))} and was "
            f"rejected by {objection.get('belief')}: "
            f"{objection.get('reason') or 'no reason given'}")
    return out


# --------------------------------------------------------------------------
# attribute proposal
# --------------------------------------------------------------------------

_ATTR_SYSTEM = """\
You are one belief among several about a Python object. You do not decide the
truth; you state what you believe and how strongly.

Reply with a single JSON object and nothing else:
  {"value": <the value you believe>, "p": <your probability, 0.0-1.0>}

Rules:
- "p" is an honest calibrated probability, not a formality. Low is fine.
- Ground the value in the state you are shown. Do not invent numbers or names
  that are absent from it.
- If the state genuinely does not determine the value, still answer, with a low p.
"""


def attribute_messages(e: Any, attr: str, contract: Any = None,
                       objections: list[dict] | None = None,
                       dialect: str = "default") -> list[dict]:
    """Messages asking one belief for one attribute of one entity."""
    body = [f"State of the object:", render_snapshot(e), ""]
    ask = f'What is this entity\'s "{attr}"?'
    if contract is not None:
        ask += f"  It is declared as: {contract.describe()}"
    body.append(ask)
    if objections:
        body.append("")
        body.append("Earlier attempts at this same attribute were rejected:")
        body.extend("  - " + line for line in _objections(objections))
        body.append("Do not repeat a rejected value. Address the objection.")
    preamble = _dialect(dialect).get("preamble")
    system = _ATTR_SYSTEM if not preamble else _ATTR_SYSTEM + "\n" + preamble
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n".join(body)}]


def response_schema(schema: Any = None) -> dict:
    """The ``{"value", "p"}`` envelope, constrained by a Schema when present."""
    value_schema = {}
    if schema is not None and hasattr(schema, "json_schema"):
        value_schema = schema.json_schema()
    return {
        "type": "object",
        "properties": {
            "value": value_schema or {},
            "p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["value", "p"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------

_EPISODE_SYSTEM = """\
You are working out what one method call would do to a Python object. You act
in a bounded loop. Every reply is a single JSON object, one action, nothing else.

Actions:
  {"action": "get", "attr": "<name>"}
      Read an attribute of the object. You get its value and its probability.
  {"action": "call", "method": "<name>", "args": [...], "kwargs": {...}}
      Run a real method or function that already exists. You get its result.
  {"action": "return", "changes": {"<attr>": <value>, ...}, "value": <result>,
   "p": <0.0-1.0>}
      Finish. "changes" are attribute writes you propose (use {} for none),
      "value" is what the call returns, "p" is your honest confidence.

Rules:
- Proposed changes must target declared attributes. They are validated after you
  return; if one is rejected you will be told why and asked again.
- You cannot freeze, pin, or certify anything. You state beliefs; code decides.
- You may not call another undefined (imagined) method.
- Budget: %(actions)d actions. Spend them; then return.
"""


def episode_messages(e: Any, expression: str, returns: Any = None,
                     action_budget: int = 8, dialect: str = "default") -> list[dict]:
    body = ["State of the object:", render_snapshot(e), "",
            f"The call to work out: {expression}"]
    if returns is not None:
        body.append(f"Its return value is declared as: {returns.describe()}")
    body.append("Begin. One action per reply.")
    system = _EPISODE_SYSTEM % {"actions": action_budget}
    preamble = _dialect(dialect).get("preamble")
    if preamble:
        system += "\n" + preamble
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n".join(body)}]


def episode_observation(kind: str, detail: Any, budget_left: int) -> dict:
    text = f"{kind}: {_short(detail)}\nActions left: {budget_left}."
    return {"role": "user", "content": text}


def episode_schema(returns: Any = None) -> dict:
    """A permissive envelope: the action grammar is a union, not one shape.

    Note there is deliberately no ``frozen`` slot anywhere in this grammar --
    a model can never pin its own answer (invariant 4).
    """
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "call", "return"]},
            "attr": {"type": "string"},
            "method": {"type": "string"},
            "args": {"type": "array"},
            "kwargs": {"type": "object"},
            "changes": {"type": "object"},
            "value": {},
            "p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["action"],
    }
