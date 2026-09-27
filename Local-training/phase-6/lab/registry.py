"""
Node type system (ComfyUI-style): every node class declares typed input and
output ports plus widget parameters; the frontend builds its palette from
`schema()`, so adding a Python class here is all it takes to add a node.
"""

from dataclasses import dataclass, field
from typing import Any


# Port data types. Wires are only allowed between equal types (or into ANY).
TYPES = {
    "SAMPLE":   "#e0a030",  # MMAD question + image record
    "IMAGE":    "#64b5f6",  # PIL.Image (RGB)
    "QUESTION": "#9ccc65",  # {question, options, answer, subtask, ...}
    "REFS":     "#4db6ac",  # normal reference image paths
    "ANOMALY":  "#ef5350",  # {map, score, tau_map, tau_score, detector}
    "REGION":   "#ff8a65",  # {boxes, fired, region_words}
    "TEXT":     "#ce93d8",  # prompt / free text
    "RESPONSE": "#ba68c8",  # raw MLLM output {text, probs, latency}
    "ANSWER":   "#f06292",  # parsed option letter
    "VERDICT":  "#fff176",  # scored answer
    "ANY":      "#bdbdbd",
}


@dataclass
class Port:
    name: str
    type: str
    optional: bool = False


@dataclass
class Param:
    name: str
    kind: str                       # choice | int | float | bool | text | textarea | upload
    default: Any = None
    choices: Any = None             # list, or callable -> list (resolved when schema is served)
    min: float = None
    max: float = None
    step: float = None
    help: str = ""
    control: bool = False           # int widgets: show fixed/increment/randomize after run
    sweepable: bool = True

    def to_json(self):
        choices = self.choices() if callable(self.choices) else self.choices
        return {k: v for k, v in {
            "name": self.name, "kind": self.kind, "default": self.default, "choices": choices,
            "min": self.min, "max": self.max, "step": self.step, "help": self.help,
            "control": self.control, "sweepable": self.sweepable,
        }.items() if v is not None}


@dataclass
class Result:
    outputs: dict = field(default_factory=dict)     # port name -> value
    ui: dict = field(default_factory=dict)          # {images:[{url,label}], text, html_table, metrics:{}}
    records: list = field(default_factory=list)     # [{kind, label, ...}] collected by the batch runner


class Node:
    TYPE = ""
    TITLE = ""
    CATEGORY = ""
    DESCRIPTION = ""
    INPUTS: list = []
    OUTPUTS: list = []
    PARAMS: list = []
    CACHEABLE = True

    def run(self, ctx, inputs: dict, params: dict) -> Result:   # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def defaults(cls):
        return {p.name: p.default for p in cls.PARAMS}

    @classmethod
    def schema(cls):
        return {
            "type": cls.TYPE, "title": cls.TITLE, "category": cls.CATEGORY,
            "description": cls.DESCRIPTION.strip(),
            "inputs": [p.__dict__ for p in cls.INPUTS],
            "outputs": [p.__dict__ for p in cls.OUTPUTS],
            "params": [p.to_json() for p in cls.PARAMS],
        }


REGISTRY: dict = {}


def register(cls):
    REGISTRY[cls.TYPE] = cls
    return cls


def schema():
    return {"types": TYPES, "nodes": [c.schema() for c in REGISTRY.values()]}


def coerce_params(cls, params: dict):
    """Fill defaults and cast widget values to the declared kinds."""
    out = {}
    given = params or {}
    for p in cls.PARAMS:
        v = given.get(p.name, p.default)
        try:
            if p.kind == "int":
                v = int(float(v))
            elif p.kind == "float":
                v = float(v)
            elif p.kind == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif v is None:
                v = p.default
        except (TypeError, ValueError):
            v = p.default
        out[p.name] = v
    return out
