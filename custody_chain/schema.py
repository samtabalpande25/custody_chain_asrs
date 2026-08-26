"""Custody episode schema.

A custody episode is an audit record that is also a training episode. The premise
of this project is that these are the same object: a facility operating under
regulatory obligation already logs state, action, and outcome with sensor
evidence, and that log is a well-formed offline RL trajectory.

The seal is what makes it a *chain*. Each episode's hash covers the previous
episode's hash, so the ledger is append-only in a checkable way: you cannot
insert, delete, or edit an episode after the fact without breaking every seal
downstream of it. That property is what a compliance function is actually buying.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Outcome = Literal["clean", "gated", "aborted"]

GENESIS = "0" * 64

# Action classes are declared per deployment. Autonomy is certified per class,
# never for the policy as a whole.
DEFAULT_SKILL_CLASSES = ("navigate", "handoff", "reseat", "isolate", "inspect")

# Aviation deployment (see adapters/asrs.py). Declared separately because
# autonomy is certified per class, and a class from one domain is not
# evidence for a class in another.
ASRS_SKILL_CLASSES = (
    "separation", "clearance", "ground_movement", "approach", "systems",
)


@dataclass
class Step:
    """One decision point, with the evidence that justified it."""

    t: float
    observation: dict[str, Any]
    action: str
    action_probs: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    cost: float = 0.0
    guardrail: str | None = None          # "gate" | "halt" | None
    evidence: list[str] = field(default_factory=list)


@dataclass
class CustodyEpisode:
    """An audit record and an offline RL trajectory. Same bytes, two readers."""

    episode_id: str
    site: str
    skill_class: str
    steps: list[Step]
    outcome: Outcome
    prev_seal: str = GENESIS
    seal: str | None = None

    # Present when the record was produced by a model reading unstructured prose
    # rather than logged directly. Covered by the seal, so the extraction cannot
    # be swapped out without breaking the chain.
    extraction: dict[str, Any] | None = None

    # ---- derived ---------------------------------------------------------
    @property
    def episode_return(self) -> float:
        return round(sum(s.reward for s in self.steps), 8)

    @property
    def episode_cost(self) -> float:
        return round(sum(s.cost for s in self.steps), 8)

    @property
    def guardrail_events(self) -> int:
        return sum(1 for s in self.steps if s.guardrail)

    @property
    def succeeded(self) -> bool:
        return self.outcome == "clean"

    # ---- sealing ---------------------------------------------------------
    def digest(self) -> str:
        """Hash covering the record *and* the previous seal."""
        payload = asdict(self)
        payload.pop("seal", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def seal_record(self, prev_seal: str | None = None) -> str:
        if prev_seal is not None:
            self.prev_seal = prev_seal
        self.seal = self.digest()
        return self.seal

    def verify(self) -> bool:
        return self.seal is not None and self.digest() == self.seal

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["episode_return"] = self.episode_return
        d["episode_cost"] = self.episode_cost
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustodyEpisode":
        d = dict(d)
        d.pop("episode_return", None)
        d.pop("episode_cost", None)
        steps = [Step(**s) for s in d.pop("steps", [])]
        return cls(steps=steps, **d)


def validate(ep: CustodyEpisode, skill_classes=DEFAULT_SKILL_CLASSES) -> list[str]:
    """Return a list of problems. Empty list means the record is admissible.

    'Admissible' is the operative word: a step with no sensor evidence cannot be
    audited, so under this scheme it cannot be used as training data either. That
    rule is strict on purpose -- it is the difference between this and an
    ordinary RL dataset.
    """
    problems: list[str] = []

    if not ep.steps:
        problems.append(f"{ep.episode_id}: no steps")
    if ep.skill_class not in skill_classes:
        problems.append(f"{ep.episode_id}: undeclared skill class {ep.skill_class!r}")
    if ep.outcome not in ("clean", "gated", "aborted"):
        problems.append(f"{ep.episode_id}: bad outcome {ep.outcome!r}")

    for i, s in enumerate(ep.steps):
        if s.guardrail not in (None, "gate", "halt"):
            problems.append(f"{ep.episode_id}[{i}]: bad guardrail {s.guardrail!r}")
        if not s.evidence:
            problems.append(f"{ep.episode_id}[{i}]: action logged with no evidence")
        if s.cost < 0:
            problems.append(f"{ep.episode_id}[{i}]: negative cost")

    return problems
