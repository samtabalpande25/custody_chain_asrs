"""The custody ledger: an append-only, hash-linked store of sealed episodes.

Stored as JSONL so it is greppable, diffable, and readable by an auditor with no
tooling. That is a deliberate choice -- a binary format would be faster and would
also be the first thing a compliance reviewer objects to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import DEFAULT_SKILL_CLASSES, GENESIS, CustodyEpisode, validate


@dataclass
class ChainBreak:
    """A point where the ledger stops being trustworthy."""

    index: int
    episode_id: str
    reason: str

    def __str__(self) -> str:
        return f"[{self.index}] {self.episode_id}: {self.reason}"


class Ledger:
    """Append-only collection of sealed custody episodes."""

    def __init__(
        self,
        episodes: list[CustodyEpisode] | None = None,
        skill_classes: tuple[str, ...] = DEFAULT_SKILL_CLASSES,
    ):
        self.episodes: list[CustodyEpisode] = episodes or []
        # Declared per deployment. A clinical ledger and a robotics ledger have
        # nothing in common here, and neither should inherit the other's list.
        self.skill_classes = skill_classes

    # ---- construction ----------------------------------------------------
    def append(self, ep: CustodyEpisode, strict: bool = True) -> CustodyEpisode:
        """Seal an episode onto the end of the chain.

        With strict=True an inadmissible record is refused rather than stored.
        An organisation may want strict=False during migration, but shipping
        that way defeats the purpose.
        """
        problems = validate(ep, skill_classes=self.skill_classes)
        if problems and strict:
            raise ValueError(
                f"refusing to seal inadmissible episode:\n  " + "\n  ".join(problems)
            )
        prev = self.episodes[-1].seal if self.episodes else GENESIS
        ep.seal_record(prev_seal=prev)
        self.episodes.append(ep)
        return ep

    # ---- integrity -------------------------------------------------------
    def verify_chain(self) -> list[ChainBreak]:
        """Walk the chain. An empty list means the ledger is intact.

        Two distinct failures are reported separately because they mean
        different things to an investigator: a bad seal means *this* record was
        edited; a bad link means a record was inserted or removed around it.
        """
        breaks: list[ChainBreak] = []
        expected_prev = GENESIS

        for i, ep in enumerate(self.episodes):
            if not ep.verify():
                breaks.append(ChainBreak(i, ep.episode_id, "seal does not match content"))
            if ep.prev_seal != expected_prev:
                breaks.append(
                    ChainBreak(i, ep.episode_id, "broken link to previous episode")
                )
            expected_prev = ep.seal or ""

        return breaks

    @property
    def head(self) -> str:
        return self.episodes[-1].seal if self.episodes else GENESIS

    # ---- io --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep.to_dict(), separators=(",", ":")) + "\n")
        return p

    @classmethod
    def load(
        cls,
        path: str | Path,
        skill_classes: tuple[str, ...] | None = None,
    ) -> "Ledger":
        """Load a ledger. Declared skill classes default to those present.

        A stored ledger already establishes which classes were admissible when
        it was written, so re-declaring them on load would let a caller silently
        narrow the set and reject records that were legitimately sealed.
        """
        eps = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    eps.append(CustodyEpisode.from_dict(json.loads(line)))
        classes = skill_classes or tuple(sorted({e.skill_class for e in eps}))
        return cls(eps, skill_classes=classes or DEFAULT_SKILL_CLASSES)

    # ---- views -----------------------------------------------------------
    def by_skill(self, skill_class: str) -> list[CustodyEpisode]:
        return [e for e in self.episodes if e.skill_class == skill_class]

    def by_site(self, site: str) -> list[CustodyEpisode]:
        return [e for e in self.episodes if e.site == site]

    def skill_classes(self) -> list[str]:
        return sorted({e.skill_class for e in self.episodes})

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)
