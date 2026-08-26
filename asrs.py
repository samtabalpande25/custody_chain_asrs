"""Adapter for the NASA Aviation Safety Reporting System (ASRS).

Why this adapter and not another structured dataset
---------------------------------------------------
Every other input path in this repo sidesteps the thing that actually blocks
deployment. The synthetic generator emits structured records by construction.
The MIMIC adapter consumes a cohort someone else preprocessed. Neither one has
ever exercised `custody_chain.extract` against a log a human wrote for other
humans, which is the form real operational history actually takes.

ASRS is the strongest available test of that path:

  * The narratives are operational prose written by pilots, controllers, and
    maintenance technicians describing what they did and why.
  * NASA de-identifies before publication, so there is no data use agreement
    and no credentialing wait.
  * Analysts hand-code each report against a controlled vocabulary, so every
    narrative arrives alongside an independent structured reading of the same
    event.

That last property is what makes ASRS more than a convenient corpus. It gives
this adapter a clean split that the MIMIC path could not have.

The split: coded fields are the record, prose is the hypothesis
--------------------------------------------------------------
The language model reads the narrative and proposes a *sequence of actions*.
It does not decide anything that carries a safety or value claim. Specifically:

  reward     never extracted, from prose or from codes. Reward design belongs
             to a deployment, not to a dataset adapter. Every step lands here
             with reward 0.0 and that is deliberate.

  outcome    taken from the ASRS analyst-coded `Events / Result` field, never
             from the model's reading.

  guardrail  taken from the coded fields only. A model-proposed guardrail is
             discarded, because "an intervention occurred" is a safety claim
             and a model's reading of a log is not evidence for one.

Anything the model did produce is sealed into the episode through
`extraction`, so its contribution to the record is visible and tamper-evident
in exactly the way a directly-logged record's is not required to be.

What this adapter is honest about
---------------------------------
ASRS is *voluntary, confidential, and non-punitive* -- it is not filed under
compulsion. That is a real departure from this project's premise, which is
about records kept under obligation. It is recorded on every episode as
`collection_regime="voluntary_immunised"` rather than quietly elided. The
argument for using it anyway is that ASRS's immunity structure gives reporters
a positive incentive to file completely, which reaches the same integrity
property by a different route -- but that is an argument, not a fact, and the
field is there so a reader can discount it.

NASA also states plainly that it does not verify or investigate reports, and
that the presence of records on a topic cannot be used to infer how common
that topic is. Nothing downstream of this adapter should be read as a
prevalence estimate.

Getting the data
----------------
    https://asrs.arc.nasa.gov/search/database.html

Run a query in the ASRS Database Online, export to CSV, and point this adapter
at the file. No account, no DUA. The export carries a two-row header (a group
row above a field row); `read_asrs_csv` handles that.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..extract import CompleteFn, extract_episode, extraction_problems, rule_based_completion
from ..schema import CustodyEpisode, Step, validate

# ---------------------------------------------------------------- vocabulary

#: Action classes for the aviation deployment. Autonomy is certified per class,
#: so these are declared here rather than inherited from the hospital defaults.
ASRS_SKILL_CLASSES = (
    "separation",        # conflicts, loss of separation, TCAS events
    "clearance",         # altitude/track/speed deviations, procedural
    "ground_movement",   # incursions, excursions, taxi events
    "approach",          # inflight events, weather, unstable approach
    "systems",           # aircraft equipment problems
)

#: `Events / Anomaly` substrings -> skill class. First match wins, so order
#: matters: more specific patterns sit above more general ones.
_ANOMALY_TO_SKILL: tuple[tuple[str, str], ...] = (
    ("ground incursion", "ground_movement"),
    ("ground excursion", "ground_movement"),
    ("ground event", "ground_movement"),
    ("conflict", "separation"),
    ("airborne conflict", "separation"),
    ("deviation - altitude", "clearance"),
    ("deviation - track", "clearance"),
    ("deviation - speed", "clearance"),
    ("deviation / discrepancy - procedural", "clearance"),
    ("clearance", "clearance"),
    ("aircraft equipment", "systems"),
    ("equipment problem", "systems"),
    ("inflight event", "approach"),
    ("weather", "approach"),
)

#: `Events / Result` substrings that mean the episode terminated early.
_ABORT_RESULTS = (
    "evasive action",
    "go around",
    "missed approach",
    "diverted",
    "rejected takeoff",
    "emergency",
    "returned to departure",
)

#: `Events / Result` substrings that mean something intervened but the episode
#: continued. These are the guardrail events -- the field MIMIC could not fill.
_GATE_RESULTS = (
    "issued advisory",
    "issued alert",
    "issued new clearance",
    "returned to clearance",
    "provided assistance",
    "requested",
)


# ---------------------------------------------------------------- csv reading

def _norm(s: str) -> str:
    """Squash all whitespace. For header cells, where newlines are noise."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def _norm_text(s: str) -> str:
    """Squash horizontal whitespace but keep line breaks.

    Narrative structure is the only segmentation signal a log carries, so
    flattening it would destroy the sequence this adapter exists to recover.
    """
    s = re.sub(r"[ \t]+", " ", s or "")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def read_asrs_csv(path: str) -> list[dict[str, str]]:
    """Read an ASRS Database Online CSV export into flat dicts.

    The export puts a group row above the field row, so a column is named by
    the pair -- ``Events`` / ``Result`` becomes ``Events / Result``. Group
    cells are only written once per span and blank thereafter, so the group is
    carried forward across the row.
    """
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))

    if len(rows) < 2:
        raise ValueError(f"{path}: expected a two-row ASRS header, got {len(rows)} row(s)")

    groups, fields = rows[0], rows[1]
    names, carried = [], ""
    for i, fname in enumerate(fields):
        gname = _norm(groups[i]) if i < len(groups) else ""
        if gname:
            carried = gname
        fname = _norm(fname)
        names.append(f"{carried} / {fname}" if carried and fname else (fname or carried))

    out = []
    for raw in rows[2:]:
        if not any(_norm(c) for c in raw):
            continue
        out.append({names[i]: _norm_text(v) for i, v in enumerate(raw) if i < len(names)})
    return out


def _col(row: dict[str, str], *needles: str) -> str:
    """Fetch the first column whose name contains all of `needles`.

    ASRS export headers vary between query builds, so matching on substrings
    is more durable than pinning exact column names.
    """
    low = {k.lower(): v for k, v in row.items()}
    for key, val in low.items():
        if all(n.lower() in key for n in needles):
            if val:
                return val
    return ""


# ---------------------------------------------------------------- mapping

def skill_class_for(anomaly: str) -> str:
    """Map a coded anomaly to a declared skill class.

    Returns ``"unmapped"`` when nothing matches, which fails `validate` on
    purpose: an episode whose action class nobody declared should surface as a
    problem rather than be silently filed under a default.
    """
    low = anomaly.lower()
    for needle, skill in _ANOMALY_TO_SKILL:
        if needle in low:
            return skill
    return "unmapped"


def outcome_for(result: str) -> str:
    """Episode outcome, from the analyst-coded result. Never from the model."""
    low = result.lower()
    if any(n in low for n in _ABORT_RESULTS):
        return "aborted"
    if any(n in low for n in _GATE_RESULTS):
        return "gated"
    return "clean"


def guardrail_for(result: str, detector: str) -> str | None:
    """Guardrail event, from coded fields only.

    A halt is the crew or the system terminating the plan. A gate is an
    external authority intervening while the plan continued. Both are
    intervention *records*, which is the whole reason this dataset is worth
    the trouble -- it populates the field the MIMIC path had to leave empty.
    """
    low = f"{result} {detector}".lower()
    if any(n in low for n in _ABORT_RESULTS):
        return "halt"
    if any(n in low for n in _GATE_RESULTS):
        return "gate"
    return None


# ---------------------------------------------------------------- conversion

@dataclass
class AdapterStats:
    """What the conversion actually did. Printed, and carried into the bundle."""

    rows: int = 0
    converted: int = 0
    skipped_no_narrative: int = 0
    unmapped_skill: int = 0
    inadmissible: int = 0
    with_guardrail: int = 0
    unclassified_actions: int = 0
    flagged_uncertain: int = 0
    problems: list[str] = field(default_factory=list)

    def render(self) -> str:
        pct = (100.0 * self.with_guardrail / self.converted) if self.converted else 0.0
        return "\n".join([
            f"  rows read              {self.rows}",
            f"  episodes converted     {self.converted}",
            f"  skipped (no narrative) {self.skipped_no_narrative}",
            f"  unmapped skill class   {self.unmapped_skill}",
            f"  with guardrail event   {self.with_guardrail}  ({pct:.1f}%)",
            f"  unclassified actions   {self.unclassified_actions}",
            f"  flagged uncertain      {self.flagged_uncertain}",
            f"  inadmissible as-is     {self.inadmissible}",
        ])


def row_to_episode(
    row: dict[str, str],
    complete_fn: CompleteFn = rule_based_completion,
    model_name: str = "rule-based-stub",
    weights_sha256: str | None = None,
    decode_params: dict[str, Any] | None = None,
) -> tuple[CustodyEpisode | None, list[str]]:
    """Convert one ASRS row into a candidate custody episode.

    Returns ``(episode, problems)``. The episode is not sealed here -- sealing
    happens on ledger append, so the ledger keeps control of chain order.
    """
    acn = _col(row, "acn") or _col(row, "accession")
    narrative = _col(row, "narrative")
    if not narrative:
        return None, ["no narrative text"]

    episode_id = f"ASRS-{acn or 'UNKNOWN'}"
    anomaly = _col(row, "events", "anomaly") or _col(row, "anomaly")
    result = _col(row, "events", "result") or _col(row, "result")
    detector = _col(row, "events", "detector") or _col(row, "detector")
    locale = _col(row, "locale") or "ZZZ"

    ep, _ = extract_episode(
        log_text=narrative,
        episode_id=episode_id,
        site=locale,
        skill_class=skill_class_for(anomaly),
        complete_fn=complete_fn,
        model_name=model_name,
        evidence_prefix="asrs",
    )

    # Extraction provenance for on-prem models. Sealed with everything else,
    # so swapping the checkpoint breaks the episode's seal.
    if weights_sha256:
        ep.extraction["weights_sha256"] = weights_sha256
    if decode_params:
        ep.extraction["decode_params"] = dict(decode_params)

    # --- coded fields override the model's reading, never the reverse -------
    ep.outcome = outcome_for(result)

    coded_guardrail = guardrail_for(result, detector)
    for s in ep.steps:
        s.guardrail = None          # discard model-proposed guardrails
        s.reward = 0.0              # the adapter never invents reward
        s.cost = 0.0
    if coded_guardrail and ep.steps:
        ep.steps[-1].guardrail = coded_guardrail
        ep.steps[-1].evidence.append(f"asrs:{episode_id}:coded_result")

    # Provenance for the coded reading, so a reader can tell which fields came
    # from an analyst and which from a model.
    ep.extraction["coded_fields"] = {
        "anomaly": anomaly,
        "result": result,
        "detector": detector,
        "source": "ASRS analyst coding",
    }
    ep.extraction["collection_regime"] = "voluntary_immunised"

    problems = validate(ep, skill_classes=ASRS_SKILL_CLASSES)
    problems += extraction_problems(ep)
    return ep, problems


def convert(
    path: str,
    complete_fn: CompleteFn = rule_based_completion,
    model_name: str = "rule-based-stub",
    limit: int | None = None,
    weights_sha256: str | None = None,
    decode_params: dict[str, Any] | None = None,
) -> tuple[list[CustodyEpisode], AdapterStats]:
    """Convert an ASRS CSV export into candidate custody episodes.

    Every returned episode is a *candidate*. Extractions default to
    unreviewed, and unreviewed extractions are inadmissible -- so a fresh
    conversion is expected to be unfittable until a human has been through it.
    That is the intended behaviour, not a bug to route around.
    """
    stats = AdapterStats()
    episodes: list[CustodyEpisode] = []

    for row in read_asrs_csv(path):
        stats.rows += 1
        if limit is not None and stats.converted >= limit:
            break

        ep, problems = row_to_episode(
            row, complete_fn, model_name, weights_sha256, decode_params
        )
        if ep is None:
            stats.skipped_no_narrative += 1
            continue

        stats.converted += 1
        if ep.skill_class == "unmapped":
            stats.unmapped_skill += 1
        if ep.guardrail_events:
            stats.with_guardrail += 1
        stats.unclassified_actions += sum(1 for s in ep.steps if s.action == "unclassified")
        stats.flagged_uncertain += len(ep.extraction.get("uncertain", []))
        if problems:
            stats.inadmissible += 1
            stats.problems.extend(problems[:3])

        episodes.append(ep)

    return episodes, stats


def iter_admissible(
    episodes: list[CustodyEpisode],
    reviewer: str | None = None,
) -> Iterator[CustodyEpisode]:
    """Yield the episodes a ledger will actually accept.

    Passing ``reviewer`` marks each remaining extraction as human-reviewed and
    re-seals it. Do that only if a human genuinely read them: the review flag
    is the record of an amendment, and a false one is worse than none.
    """
    from ..extract import mark_reviewed

    for ep in episodes:
        if validate(ep, skill_classes=ASRS_SKILL_CLASSES):
            continue
        if reviewer:
            mark_reviewed(ep, reviewer)
        elif extraction_problems(ep):
            continue
        yield ep
