"""Two things the evidence field has been conflating.

The finding
-----------
Run the stub against any narrative and every step passes `validate`'s evidence
check. Run a model against real ASRS prose and many steps fail it. The model is
not doing worse -- it is being honest. The stub writes `log_line:3` for every
step it emits, unconditionally, and that string has been satisfying a check
whose docstring says "a step with no sensor evidence cannot be audited."

`log_line:3` is not sensor evidence. It is a pointer to the sentence the claim
came from. Both are worth having and they support different claims:

  LOCATOR   where in the source this step was read from. Always derivable,
            never absent, and it supports exactly one claim: "an extraction of
            this source produced this step." That is a provenance claim.

  CITATION  what the log itself points at as proof -- an FDR trace, a chart
            time, a radar tape, a photograph. Supports the much stronger claim
            "this happened, and here is the artifact that shows it." That is an
            evidentiary claim, and prose written from memory usually cannot
            make it.

Collapsing them means a fabricated locator silently discharges an evidentiary
requirement. That is the failure mode this whole repo exists to prevent, sitting
inside `validate`.

What this changes for ASRS
--------------------------
ASRS narratives are voluntary recollections filed after the fact. Most steps in
them will never have a citation, because the reporter was not required to
produce one. That is a property of the corpus, not a defect in the reading.

So ASRS fails the citation requirement the way MIMIC failed the guardrail
requirement -- structurally, and worth saying out loud rather than routing
around. `evidence_regime` records which standard an episode can actually meet,
in the same spirit as `collection_regime="voluntary_immunised"`.

A deployment with charted evidence sets `require_citation=True` and gets the
strict rule. ASRS cannot, and the bundle should say so rather than implying the
episodes carry evidence they do not.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import CustodyEpisode, Step

LOCATOR_PREFIX = "src"
CITATION_PREFIX = "cite"

#: What kind of evidentiary standard an episode's steps can actually meet.
EVIDENCE_REGIMES = (
    "cited",        # every step points at an artifact outside the narrative
    "located",      # steps are traceable to source text, nothing more
    "mixed",        # some steps cited, some only located
)


def locator(episode_id: str, index: int, line: int | None = None) -> str:
    """A pointer to where in the source a step was read from.

    Deterministic and recomputable: given the same source text and the same
    reading, an auditor lands on the same sentence. That is the whole claim it
    makes, and it should not be mistaken for a larger one.
    """
    tail = f":line:{line}" if line is not None else ""
    return f"{LOCATOR_PREFIX}:{episode_id}:step:{index}{tail}"


def citation(episode_id: str, mention: str) -> str:
    """A reference to something the log cites as proof of what it describes."""
    slug = re.sub(r"\s+", "_", mention.strip())
    return f"{CITATION_PREFIX}:{episode_id}:{slug}"


def is_citation(ref: str) -> bool:
    return ref.startswith(f"{CITATION_PREFIX}:")


def relabel_episode(
    ep: CustodyEpisode,
    source_lines: int | None = None,
) -> CustodyEpisode:
    """Split an episode's evidence into locators and citations.

    Existing references that look like the old synthesised pointers become
    locators. Everything else is treated as a citation -- the model only ever
    put things there because the log mentioned them.

    Every step ends up with a locator, so `validate` still passes. What changes
    is that passing no longer implies the step is evidenced, and
    `citation_coverage` reports the difference instead of hiding it.
    """
    for i, s in enumerate(ep.steps):
        refs, cites = [], []
        for ref in s.evidence:
            tail = ref.split(":", 2)[-1]
            if tail.startswith("log_line:") or tail.startswith("step:"):
                continue                      # a locator, about to be rewritten
            cites.append(citation(ep.episode_id, tail))
        line = i if source_lines is None or i < source_lines else None
        refs.append(locator(ep.episode_id, i, line))
        s.evidence = refs + cites

    ep.extraction = dict(ep.extraction or {})
    ep.extraction["evidence_regime"] = regime(ep)
    ep.extraction["citation_coverage"] = citation_coverage(ep)
    return ep


def citation_coverage(ep: CustodyEpisode) -> dict[str, Any]:
    """How many steps point at something outside the narrative."""
    total = len(ep.steps)
    cited = sum(1 for s in ep.steps if any(is_citation(r) for r in s.evidence))
    return {
        "steps": total,
        "cited": cited,
        "uncited": total - cited,
        "fraction": round(cited / total, 4) if total else 0.0,
    }


def regime(ep: CustodyEpisode) -> str:
    cov = citation_coverage(ep)
    if not cov["steps"] or cov["cited"] == 0:
        return "located"
    return "cited" if cov["uncited"] == 0 else "mixed"


def evidence_problems(
    ep: CustodyEpisode,
    require_citation: bool = False,
) -> list[str]:
    """Admissibility checks for evidence, with the strict rule made optional.

    `require_citation=False` is the ASRS setting and it is a real weakening:
    episodes admitted under it are traceable to a source, not evidenced by one.
    Anything built on them inherits that, and the bundle should say so.

    `require_citation=True` is the rule `validate`'s docstring describes. A
    facility with charted or sensor-referenced logs should run it, and this repo
    should stop claiming to enforce it when it does not.
    """
    problems: list[str] = []
    for i, s in enumerate(ep.steps):
        if not any(r.startswith(f"{LOCATOR_PREFIX}:") for r in s.evidence):
            problems.append(
                f"{ep.episode_id}[{i}]: no source locator — the step cannot be "
                "traced back to the text it was read from"
            )
        if require_citation and not any(is_citation(r) for r in s.evidence):
            problems.append(
                f"{ep.episode_id}[{i}]: no citation — the log asserts this "
                "action without pointing at anything that shows it happened"
            )
    return problems


def corpus_summary(episodes: list[CustodyEpisode]) -> str:
    """What standard this corpus can actually meet. Print it, do not bury it."""
    steps = sum(len(e.steps) for e in episodes)
    cited = sum(citation_coverage(e)["cited"] for e in episodes)
    by_regime: dict[str, int] = {}
    for e in episodes:
        by_regime[regime(e)] = by_regime.get(regime(e), 0) + 1

    pct = (100.0 * cited / steps) if steps else 0.0
    lines = [
        f"  steps                  {steps}",
        f"  with a citation        {cited}  ({pct:.1f}%)",
        f"  located only           {steps - cited}",
    ]
    for r in EVIDENCE_REGIMES:
        if r in by_regime:
            lines.append(f"  episodes {r:<13} {by_regime[r]}")
    if pct < 50:
        lines.append(
            "\n  Most steps are traceable to the narrative but not evidenced by\n"
            "  anything outside it. That is a property of voluntary recollection,\n"
            "  not a defect in the reading. Do not present these episodes as\n"
            "  meeting the evidence standard in schema.validate's docstring."
        )
    return "\n".join(lines)
