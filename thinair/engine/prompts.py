"""All prompt construction, isolated and testable.

Template versions are part of every ``ModelBelief`` id (invariant 6): editing a
prompt here without bumping the version would silently re-label one belief's
opinions as another's.  Don't.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["ATTRIBUTE_TEMPLATE", "EPISODE_TEMPLATE", "AGENT_TEMPLATE",
           "DIALECTS", "attribute_messages", "episode_messages",
           "episode_observation", "render_snapshot", "response_schema",
           "episode_schema", "template_version"]

#: Bump on *any* edit to the corresponding builder (invariant 6).
ATTRIBUTE_TEMPLATE = "extract-v4"
EPISODE_TEMPLATE = "episode-v3"
#: The open agentic turn -- "you are this entity; act" -- is its own
#: template, versioned separately: a variant, never an edit.
AGENT_TEMPLATE = "agent-v2"

TRUNCATION_MARKER = " …[truncated]"
MAX_RENDER = 2000


def template_version(kind: str = "attribute") -> str:
    if kind == "episode":
        return EPISODE_TEMPLATE
    if kind == "agent":
        return AGENT_TEMPLATE
    return ATTRIBUTE_TEMPLATE


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


def _render_boundary(e: Any, limit: int) -> str:
    """What crosses an entity boundary: identity, purpose, public cells only.

    No panel, no ledger slice, no contracts, no methods, no belief
    attribution -- another mind's mechanisms are its own business.
    """
    name = e.__class_name__ or ""
    lines = [f"entity: {e.__entity__}" + (f" ({name})" if name else "")]
    purpose = e.__purpose__
    if purpose:
        lines.append(f'purpose: "{purpose.strip()}"')
    cells = e.__attrs__()
    if cells:
        lines.append("public state:")
        for attr, opinion in cells.items():
            lines.append(f"  {attr} = {_short(opinion.value, limit)}"
                         f"   (p={opinion.p:.2f})")
    else:
        lines.append("public state: (nothing visible)")
    return "\n".join(lines)


def render_snapshot(e: Any, limit: int = MAX_RENDER) -> str:
    """The sealed snapshot as text.  State only -- never a transcript."""
    if getattr(e, "__boundary__", False):
        return _render_boundary(e, limit)
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
    peers = getattr(e, "__peers__", None) or {}
    if peers:
        lines.append("entities you hold references to (their public view):")
        for view in peers.values():
            lines.append("  " + render_snapshot(view, limit=limit // 2)
                         .replace("\n", "\n  "))
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
    """The ``{"value", "p"}`` envelope, constrained by a SchemaBelief when present."""
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
  {"action": "tell", "entity": "<entity id>", "method": "<verb>", "args": [...]}
      Send an addressed message to another entity -- a call delivered on that
      entity's own later turn. Nothing comes back now; any reply arrives
      later as a call to this object. Queued; it is sent only if you finish.
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


_AGENT_SYSTEM = """\
You are the entity shown below -- not an assistant describing it, the entity
itself, deciding its own next move. You act in a bounded loop. Every reply is
a single JSON object, one action, nothing else.

Actions:
  {"action": "get", "attr": "<name>"}
      Read one of your attributes. You get its value and its probability.
  {"action": "call", "method": "<name>", "args": [...], "kwargs": {...}}
      Run one of your real methods. You get its result.
  {"action": "tell", "entity": "<entity id>", "method": "<verb>", "args": [...]}
      Send an addressed message to another entity -- name a verb that says
      what you want of it, with any arguments. It is delivered as a call
      that entity answers on its own later turn -- nothing comes back now;
      any reply arrives later as a call to you. Queued; it is sent only if
      you end your turn normally.
  {"action": "return", "changes": {"<attr>": <value>, ...}, "value": <note>,
   "p": <0.0-1.0>}
      End your turn. "changes" are writes to your own attributes (use {} for
      none), "value" is a short note on what you did, "p" is your honest
      confidence.

Rules:
- You may only write your own declared, unfrozen attributes. Acting on
  another entity is never yours to do directly: to reach one, "tell" it,
  using its exact entity id, and it will decide for itself on its own turn.
- Other entities see only your public attributes; what you see of them is
  only their public state.
- Tell an entity only when you want something of it; a turn that needs
  nothing from nobody simply returns.
- Proposed changes are validated after you return; if one is rejected you
  will be told why and asked again.
- You cannot freeze, pin, or certify anything. You state beliefs; code decides.
- You may not call another undefined (imagined) method on yourself.
- If there is nothing worth doing, return {} changes and say so.
- Budget: %(actions)d actions. Spend them; then return.
"""


def episode_messages(e: Any, expression: str, returns: Any = None,
                     action_budget: int = 8, dialect: str = "default",
                     objections: list[dict] | None = None,
                     acting: bool = False) -> list[dict]:
    lead = "Your state:" if acting else "State of the object:"
    body = [lead, render_snapshot(e), ""]
    if acting:
        body.append(f"Your turn. It was prompted by: {expression}")
    else:
        body.append(f"The call to work out: {expression}")
    if returns is not None:
        body.append(f"Its return value is declared as: {returns.describe()}")
    if objections:
        body.append("")
        body.append("Earlier attempts this turn were rejected:")
        body.extend("  - " + line for line in _objections(objections))
        body.append("Do not repeat a rejected proposal. Address the objection.")
    body.append("Begin. One action per reply.")
    system = (_AGENT_SYSTEM if acting else _EPISODE_SYSTEM) % {
        "actions": action_budget}
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
            "action": {"type": "string",
                       "enum": ["get", "call", "tell", "return"]},
            "attr": {"type": "string"},
            "entity": {"type": "string"},
            "method": {"type": "string"},
            "args": {"type": "array"},
            "kwargs": {"type": "object"},
            "changes": {"type": "object"},
            "value": {},
            "p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["action"],
    }
