"""The model folder registry.

Adding a model costs one folder, one file, zero edits elsewhere.  Folders are
**data**, never behavior: a ``ModelDef`` is consumed by the one transport in
``engine/`` and by ``engine/prompts.py``.  A genuinely new wire format is a new
transport, selected by ``ModelDef.transport``.

Imported only by ``ModelBelief`` (invariant 7).
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import os
import pkgutil
from dataclasses import dataclass, field, replace

__all__ = ["ModelDef", "resolve", "registry", "register", "load_path",
           "GENERIC", "KNOWN_QUIRKS", "STRUCTURED_OUTPUT_MODES"]

KNOWN_QUIRKS = frozenset({
    "no_system_role",       # fold the system message into the first user turn
    "no_temperature",       # server rejects the temperature knob
    "no_json_mode",         # ignore response_format entirely
    "single_user_turn",     # collapse the conversation into one user message
    "no_stop",              # server rejects the stop knob
})

STRUCTURED_OUTPUT_MODES = ("json_schema", "json_mode", "prompted")


@dataclass(frozen=True)
class ModelDef:
    """One declarative model definition -- the whole content of a model folder."""

    match: tuple[str, ...]
    defaults: dict = field(default_factory=dict)
    structured_output: str = "prompted"
    think: dict = field(default_factory=dict)
    request_extra: dict = field(default_factory=dict)
    quirks: tuple[str, ...] = ()
    prompt_dialect: str = "default"
    version: str = "1"
    transport: str = "openai_compat"
    name: str = ""
    origin: str = "builtin"

    def __post_init__(self) -> None:
        if isinstance(self.match, str):
            object.__setattr__(self, "match", (self.match,))
        object.__setattr__(self, "match", tuple(self.match))
        object.__setattr__(self, "quirks", tuple(self.quirks))
        if not self.match:
            raise ValueError("a ModelDef must claim at least one name or glob")
        if self.structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"structured_output must be one of {STRUCTURED_OUTPUT_MODES}, "
                f"got {self.structured_output!r}")
        for q in self.quirks:
            if q not in KNOWN_QUIRKS:
                raise ValueError(f"unknown quirk {q!r}; known: {sorted(KNOWN_QUIRKS)}")
        temp = self.defaults.get("temperature")
        if temp is not None and not (0.0 <= float(temp) <= 2.0):
            raise ValueError(f"default temperature out of range: {temp!r}")
        mt = self.defaults.get("max_tokens")
        if mt is not None and int(mt) <= 0:
            raise ValueError(f"default max_tokens must be positive: {mt!r}")
        if self.think:
            on, off = self.think.get("on", {}), self.think.get("off", {})
            if not isinstance(on, dict) or not isinstance(off, dict):
                raise TypeError("think.on / think.off must be payload dicts")
            for k in set(on) & set(off):
                if on[k] == off[k]:
                    raise ValueError(
                        f"think payloads are not disjoint on {k!r}: both {on[k]!r}")
        for pattern in self.match:
            try:
                fnmatch.translate(pattern)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"un-compilable glob {pattern!r}: {exc}") from None
        if not str(self.version):
            raise ValueError("version participates in belief identity (invariant 6)")

    # -- request shaping ---------------------------------------------------
    def claims(self, model_name: str) -> bool:
        low = model_name.lower()
        return any(fnmatch.fnmatchcase(low, p.lower()) for p in self.match)

    def specificity(self, model_name: str) -> tuple:
        """Higher sorts first: literal beats glob, longer glob beats shorter."""
        low = model_name.lower()
        best = (0, 0)
        for p in self.match:
            if not fnmatch.fnmatchcase(low, p.lower()):
                continue
            literal = 0 if any(c in p for c in "*?[") else 1
            best = max(best, (literal, len(p)))
        external = 1 if self.origin == "external" else 0
        return (external, *best)

    def think_payload(self, think: bool) -> dict:
        if not self.think:
            return {}
        return dict(self.think.get("on" if think else "off", {}))

    def has(self, quirk: str) -> bool:
        return quirk in self.quirks

    def identity(self) -> str:
        """The part of a belief id that this def contributes (invariant 6)."""
        return f"{self.name or self.match[0]}/v{self.version}"


# ``structured_output="prompted"``: the prompt already demands JSON, and an
# unknown server given ``response_format`` may mangle or ignore it (the ds4
# server does) -- worse than never sending it.  A folder that *knows* its
# server honors json_schema/json_mode opts in; the fallback stays quiet.
GENERIC = ModelDef(
    match=("*",),
    defaults=dict(temperature=0.2, max_tokens=2048),
    structured_output="prompted",
    version="2",            # v1 sent json_mode; behavior moved, so the id moves
    name="generic-openai-compat",
)

_REGISTRY: dict[str, ModelDef] = {}
_LOADED_PATHS: set[str] = set()


def _overlaps(a: ModelDef, b: ModelDef) -> str | None:
    """Detect claim collisions between two same-origin defs."""
    for pa in a.match:
        for pb in b.match:
            if pa.lower() == pb.lower():
                return pa
            # a glob swallowing another def's literal claim
            if not any(c in pb for c in "*?[") and fnmatch.fnmatchcase(pb.lower(), pa.lower()):
                return f"{pa} ⊇ {pb}"
            if not any(c in pa for c in "*?[") and fnmatch.fnmatchcase(pa.lower(), pb.lower()):
                return f"{pb} ⊇ {pa}"
    return None


def register(definition: ModelDef, name: str | None = None) -> ModelDef:
    """Register one def; overlapping same-origin claims raise."""
    if not isinstance(definition, ModelDef):
        raise TypeError("expected a ModelDef")
    if name and not definition.name:
        definition = replace(definition, name=name)
    key = definition.name or definition.match[0]
    for other in _REGISTRY.values():
        if other.name == definition.name and other.origin == definition.origin:
            continue
        if other.origin != definition.origin:
            continue  # external defs may override built-in claims
        clash = _overlaps(definition, other)
        if clash:
            raise ValueError(
                f"model def {key!r} overlaps {other.name!r} on {clash!r}; "
                "claims must be disjoint")
    _REGISTRY[key] = definition
    return definition


def _discover_builtin() -> None:
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue  # skips _template: a copy-me starting point, not a model
        module = importlib.import_module(f"{__name__}.{info.name}.model")
        definition = getattr(module, "MODEL", None)
        if definition is None:
            raise ValueError(f"models/{info.name}/model.py defines no MODEL")
        register(definition, name=info.name)


def load_path(paths) -> list[ModelDef]:
    """Load user-side model folders (``THINAIR_MODELS_PATH`` / ``models_path=``)."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = []
    for root in paths:
        root = os.fspath(root)
        if not os.path.isdir(root) or root in _LOADED_PATHS:
            continue
        _LOADED_PATHS.add(root)
        for entry in sorted(os.listdir(root)):
            if entry.startswith("_") or entry.startswith("."):
                continue
            path = os.path.join(root, entry, "model.py")
            if not os.path.isfile(path):
                continue
            spec = importlib.util.spec_from_file_location(f"_thinair_model_{entry}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            definition = getattr(module, "MODEL", None)
            if definition is None:
                raise ValueError(f"{path} defines no MODEL")
            out.append(register(replace(definition, origin="external"), name=entry))
    return out


def registry() -> dict[str, ModelDef]:
    return dict(_REGISTRY)


def resolve(model_name: str, models_path=None) -> ModelDef:
    """Most-specific claim wins; no claim -> the generic OpenAI-compat fallback."""
    if models_path:
        load_path(models_path)
    env = os.environ.get("THINAIR_MODELS_PATH")
    if env:
        load_path([p for p in env.split(os.pathsep) if p])
    candidates = [d for d in _REGISTRY.values() if d.claims(model_name)]
    if not candidates:
        return GENERIC
    candidates.sort(key=lambda d: d.specificity(model_name), reverse=True)
    return candidates[0]


_discover_builtin()
