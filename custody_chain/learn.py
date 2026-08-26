"""Offline policy fitting, with provenance.

The learning here is deliberately the least novel part: first-visit Monte-Carlo
return-to-go over sealed episodes. It is a textbook offline estimator, chosen
because it is auditable by hand -- a reviewer can recompute any value in this
table with a spreadsheet.

The contribution is the bookkeeping around it. Every fitted value carries the
list of episode ids and seals that produced it, so a policy update can answer
"why did you learn this?" with a checkable answer rather than an assurance.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .ledger import Ledger
from .schema import CustodyEpisode, Step

# A state abstraction maps a logged observation to the discrete state the policy
# reasons over. Deployments declare their own; this is the seam where domain
# knowledge enters.
StateFn = Callable[[Step], str]


def default_state_fn(step: Step) -> str:
    """Sort the observation dict into a stable string key."""
    return "·".join(f"{k}={step.observation[k]}" for k in sorted(step.observation))


@dataclass
class ValueEstimate:
    """One fitted (state, action) value and the evidence behind it."""

    state: str
    action: str
    mean_return: float
    n: int
    mean_cost: float = 0.0
    episode_ids: list[str] = field(default_factory=list)
    episode_seals: list[str] = field(default_factory=list)


@dataclass
class Policy:
    """A fitted policy plus the provenance that makes it defensible."""

    version: str
    skill_class: str
    discount: float
    estimates: dict[str, ValueEstimate] = field(default_factory=dict)
    ledger_head: str = ""
    n_episodes: int = 0
    fingerprint: str = ""

    # ---- use -------------------------------------------------------------
    def best_action(self, state: str) -> str | None:
        cands = [e for e in self.estimates.values() if e.state == state]
        if not cands:
            return None
        return max(cands, key=lambda e: e.mean_return).action

    def provenance(self, state: str, action: str) -> ValueEstimate | None:
        return self.estimates.get(f"{state}|{action}")

    def states(self) -> list[str]:
        return sorted({e.state for e in self.estimates.values()})

    def support_summary(self, thin_below: int = 5) -> dict[str, Any]:
        """How much evidence actually sits behind this table.

        A fitted value with two observations is not a finding, and a table that
        reports it beside a value with two hundred invites the reader to treat
        them alike. Gottesman et al. (2019) make this the central caution about
        offline RL on observational records: the estimator will happily produce
        a number for a state-action pair the clinicians almost never chose.

        This does not fix the problem -- only more data or a coarser state
        abstraction does that. It refuses to hide it.
        """
        if not self.estimates:
            return {"values": 0, "thin": 0, "thin_fraction": 0.0,
                    "median_n": 0, "obs_per_value": 0.0}

        ns = sorted(e.n for e in self.estimates.values())
        thin = sum(1 for n in ns if n < thin_below)
        return {
            "values": len(ns),
            "thin": thin,
            "thin_fraction": round(thin / len(ns), 4),
            "thin_below": thin_below,
            "median_n": ns[len(ns) // 2],
            "obs_per_value": round(sum(ns) / len(ns), 2),
        }

    # ---- integrity -------------------------------------------------------
    def compute_fingerprint(self) -> str:
        """Hash of the fitted table plus the ledger head it was fitted from.

        Two claims become checkable: this policy is the one that was certified,
        and it was fitted from this exact ledger state.
        """
        payload = {
            "version": self.version,
            "skill_class": self.skill_class,
            "discount": self.discount,
            "ledger_head": self.ledger_head,
            "estimates": {
                k: [round(v.mean_return, 8), v.n, sorted(v.episode_seals)]
                for k, v in sorted(self.estimates.items())
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(blob.encode()).hexdigest()
        return self.fingerprint

    # ---- io --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "skill_class": self.skill_class,
            "discount": self.discount,
            "ledger_head": self.ledger_head,
            "n_episodes": self.n_episodes,
            "fingerprint": self.fingerprint,
            "estimates": {k: asdict(v) for k, v in self.estimates.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        d = json.loads(Path(path).read_text())
        est = {k: ValueEstimate(**v) for k, v in d.pop("estimates", {}).items()}
        return cls(estimates=est, **d)


def fit(
    ledger: Ledger,
    skill_class: str,
    version: str = "v1.0",
    discount: float = 0.95,
    state_fn: StateFn = default_state_fn,
    require_intact_chain: bool = True,
) -> Policy:
    """Fit a policy from sealed episodes of one skill class.

    Refuses to fit from a broken chain by default. Training on records whose
    integrity cannot be established would produce a policy whose provenance
    claim is worthless, which is the one thing this project exists to avoid.
    """
    if require_intact_chain:
        breaks = ledger.verify_chain()
        if breaks:
            raise ValueError(
                "refusing to fit from a broken ledger:\n  "
                + "\n  ".join(str(b) for b in breaks)
            )

    episodes = ledger.by_skill(skill_class)
    acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ret": 0.0, "cost": 0.0, "n": 0, "ids": [], "seals": []}
    )

    for ep in episodes:
        seen: set[str] = set()
        rewards = [s.reward for s in ep.steps]
        costs = [s.cost for s in ep.steps]

        for i, step in enumerate(ep.steps):
            k = f"{state_fn(step)}|{step.action}"
            if k in seen:          # first-visit
                continue
            seen.add(k)

            g = c = 0.0
            disc = 1.0
            for j in range(i, len(rewards)):
                g += disc * rewards[j]
                c += disc * costs[j]
                disc *= discount

            a = acc[k]
            a["ret"] += g
            a["cost"] += c
            a["n"] += 1
            a["ids"].append(ep.episode_id)
            if ep.seal:
                a["seals"].append(ep.seal)

    estimates = {}
    for k, a in acc.items():
        state, action = k.split("|", 1)
        estimates[k] = ValueEstimate(
            state=state,
            action=action,
            mean_return=round(a["ret"] / a["n"], 8),
            mean_cost=round(a["cost"] / a["n"], 8),
            n=a["n"],
            episode_ids=a["ids"],
            episode_seals=a["seals"],
        )

    pol = Policy(
        version=version,
        skill_class=skill_class,
        discount=discount,
        estimates=estimates,
        ledger_head=ledger.head,
        n_episodes=len(episodes),
    )
    pol.compute_fingerprint()
    return pol
