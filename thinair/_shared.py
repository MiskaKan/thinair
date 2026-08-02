"""The internals both halves of thinair touch.

`thinair` holds the object model (Thing, its strata, the plan runtime);
`thinair.engine` holds everything model-specific. A handful of things sit
between them — the ambient per-operation state, the backend defaults, and
the debug rendering primitives — and live here so neither half has to
import the other. Nothing in this module imports from either.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import traceback

_DEFAULT_BASE_URL = os.environ.get("THINAIR_BASE_URL", "http://127.0.0.1:8000/v1")
_DEFAULT_API_KEY = os.environ.get("THINAIR_API_KEY", "1234")
_DEFAULT_MAX_TOKENS = int(os.environ.get("THINAIR_MAX_TOKENS", "65536"))

# the ambient state of one inference: what is required of it, whether it
# narrates itself, and where it sits in the stack of operations
_required = contextvars.ContextVar("thing_required_confidence", default=None)
_debugging = contextvars.ContextVar(
    "thing_debug", default=os.environ.get("THINAIR_DEBUG", "") not in ("", "0")
)
_purpose = contextvars.ContextVar("thing_purpose", default="inference")
_op_stack = contextvars.ContextVar("thing_op_stack", default=())
_op_debug = contextvars.ContextVar("thing_op_debug", default=None)

# every file of the package, so a call site is reported as the user's own
# line and never as thinair's own plumbing
_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


def _snip(text, limit=100):
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_schema(schema):
    if isinstance(schema, type):
        return schema.__name__
    if isinstance(schema, dict):
        return "{" + ", ".join(f'"{k}": {_render_schema(v)}' for k, v in schema.items()) + "}"
    if isinstance(schema, list):
        return "[" + (_render_schema(schema[0]) + ", ..." if schema else "...") + "]"
    return json.dumps(schema)


def _told_event_line(event):
    """One journal event rendered for the telling. A huge call result is
    elided down to a preview — the record keeps it whole. A trailing
    delete says what it means: the question re-opened, not answered."""
    if event.get("event") == "condense":
        return "(the older story, retold): " + str(event.get("summary", ""))
    if event.get("event") == "delete":
        return (
            json.dumps(event, default=repr, ensure_ascii=False)
            + " — the recorded value was retired; the question is open "
            "again, to be answered afresh (deletion is not absence)"
        )
    text = json.dumps(event, default=repr, ensure_ascii=False)
    if len(text) <= 600:
        return text
    slim = dict(event)
    if "result" in slim:
        rendered = json.dumps(slim["result"], default=repr, ensure_ascii=False)
        if len(rendered) > 300:
            slim["result"] = rendered[:300] + f"… ({len(rendered)} chars, elided)"
    text = json.dumps(slim, default=repr, ensure_ascii=False)
    return text if len(text) <= 900 else text[:900] + "…"


def _render_messages(messages):
    out = []
    for message in messages:
        out.append(f"[{message['role']}]")
        for line in str(message.get("content", "")).splitlines() or [""]:
            out.append(f"  {line}")
    return "\n".join(out)


def _meta_line(meta):
    bits = []
    if meta.get("prompt_tokens") is not None or meta.get("completion_tokens") is not None:
        bits.append(
            f"tokens {meta.get('prompt_tokens', '?')} in → "
            f"{meta.get('completion_tokens', '?')} out"
        )
    finish = meta.get("finish_reason")
    if finish == "length":
        bits.append("hit the token budget")
    elif finish == "cut":
        bits.append("cut (loop detected)")
    elif finish:
        bits.append(finish)
    if meta.get("interventions"):
        bits.append(f"{meta['interventions']} intervention(s)")
    if meta.get("direct"):
        bits.append("direct (no thinking)")
    if meta.get("json_mode") is False:
        bits.append("freeform (server JSON mode off)")
    return " · ".join(bits)


def _call_site():
    """The user's own frames (thinair's filtered out), outermost first,
    indented down to the exact line that triggered this inference."""
    frames = [
        f
        for f in traceback.extract_stack()
        if not os.path.realpath(f.filename).startswith(_PACKAGE_DIR + os.sep)
    ][-5:]
    lines = []
    for i, frame in enumerate(frames):
        loc = f"{os.path.basename(frame.filename)}:{frame.lineno} {frame.name}"
        if frame.line:
            loc += f"  {frame.line.strip()}"
        lines.append("  " * i + loc)
    return "\n".join(lines)


def _debug_box(title, sections, footer="", opener="┌", close=True, tag=True):
    """One titled box. `opener="├"` continues the operation above it and
    `close=False` leaves the bottom open, so a whole operation reads as
    a single tall structure closed by its final block (or, for plans,
    by the final step's `▸` line). `tag=False` renders a bare segment
    title without the `thinair ·` prefix; `title=None` emits labeled
    sections only. Boxes born inside another operation say so in the
    title and shift right with depth."""
    chain = _op_stack.get()
    pad = "  " * len(chain)
    lines = []
    if title is not None:
        if tag:
            if chain:
                title = f"{title} · in {' › '.join(chain)}"
            title = f"thinair · {title}"
        lines.append(f"{opener}─ {title} " + "─" * max(1, 56 - len(title)))
    for label, body in sections:
        if label:
            lines.append(f"├─ {label} " + "─" * max(1, 56 - len(label)))
        for line in str(body).splitlines() or [""]:
            lines.append(f"│ {line}")
    if close:
        lines.append(f"└─ {footer}" if footer else "└" + "─" * 59)
    elif footer:
        lines.append(f"├─ {footer}")
    print("\n".join(pad + line for line in lines), file=sys.stderr)
