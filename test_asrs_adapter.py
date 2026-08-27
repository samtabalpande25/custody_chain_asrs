"""Tests for the ASRS adapter.

The properties worth pinning are not "does it parse a CSV" but the boundaries
this adapter enforces between what a model may decide and what it may not.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from custody_chain.adapters import asrs                       # noqa: E402
from custody_chain.evidence import (                          # noqa: E402
    EVIDENCE_REGIMES,
    evidence_problems,
    is_citation,
)
from custody_chain.extract import extraction_problems, mark_reviewed  # noqa: E402
from custody_chain.schema import validate                     # noqa: E402

_FIXTURE_CANDIDATES = (
    ROOT / "asrs_sample.csv",
    ROOT / "tests" / "fixtures" / "asrs_sample.csv",
    ROOT / "fixtures" / "asrs_sample.csv",
)
FIXTURE = str(next(p for p in _FIXTURE_CANDIDATES if p.exists()))


@pytest.fixture(scope="module")
def episodes():
    eps, _ = asrs.convert(FIXTURE)
    return eps


# ---------------------------------------------------------------- parsing
def test_two_row_header_is_flattened():
    rows = asrs.read_asrs_csv(FIXTURE)
    assert rows, "no rows parsed"
    assert any("Events / Result" in k for k in rows[0])


def test_narrative_line_breaks_survive_parsing():
    rows = asrs.read_asrs_csv(FIXTURE)
    narratives = [asrs._col(r, "narrative") for r in rows]
    assert any("\n" in n for n in narratives), (
        "line breaks were flattened — narrative structure is the only "
        "segmentation signal a log carries"
    )


def test_multiline_narrative_yields_multiple_steps(episodes):
    assert any(len(e.steps) > 1 for e in episodes)


# ---------------------------------------------------------------- mapping
@pytest.mark.parametrize("anomaly,expected", [
    ("Airborne Conflict", "separation"),
    ("Ground Incursion Runway", "ground_movement"),
    ("Deviation - Altitude Overshoot", "clearance"),
    ("Aircraft Equipment Problem Less Severe", "systems"),
])
def test_anomaly_maps_to_declared_skill_class(anomaly, expected):
    assert asrs.skill_class_for(anomaly) == expected


def test_unrecognised_anomaly_is_not_silently_defaulted():
    """An undeclared action class must surface, not land in a catch-all."""
    assert asrs.skill_class_for("Something Entirely Unmapped") == "unmapped"
    ep = next(e for e in asrs.convert(FIXTURE)[0] if e.skill_class == "unmapped")
    assert any("undeclared skill class" in p
               for p in validate(ep, skill_classes=asrs.ASRS_SKILL_CLASSES))


# ------------------------------------------------- the model's boundaries
def test_adapter_never_invents_reward(episodes):
    """Reward design belongs to a deployment, never to a dataset adapter."""
    assert all(s.reward == 0.0 and s.cost == 0.0
               for e in episodes for s in e.steps)


def test_guardrails_come_from_coded_fields_not_the_model(episodes):
    """'An intervention occurred' is a safety claim. A model may not make one."""
    for e in episodes:
        flagged = [s for s in e.steps if s.guardrail]
        assert len(flagged) <= 1, "guardrail should come from the episode-level code"
        if flagged:
            assert any("coded_result" in ev for ev in flagged[-1].evidence)


def test_outcome_comes_from_analyst_coding():
    assert asrs.outcome_for("Flight Crew Executed Go Around / Missed Approach") == "aborted"
    assert asrs.outcome_for("Air Traffic Control Issued New Clearance") == "gated"
    assert asrs.outcome_for("General None Reported / Taken") == "clean"


def test_guardrail_events_are_actually_populated(episodes):
    """The field the MIMIC path had to leave empty. If this is 0, the whole
    reason for preferring ASRS over another structured dataset is gone."""
    assert sum(e.guardrail_events for e in episodes) > 0


# ---------------------------------------------------------- admissibility
def test_fresh_extractions_are_inadmissible(episodes):
    assert all(extraction_problems(e) for e in episodes), (
        "a model's reading of a log is a hypothesis, not a record"
    )


def test_review_is_an_amendment_that_reseals():
    eps, _ = asrs.convert(FIXTURE)
    ep = eps[0]
    ep.seal_record()
    before = ep.seal

    mark_reviewed(ep, "test-reviewer")

    assert ep.seal != before, "review is an amendment and must re-seal"
    assert ep.extraction["review_status"] == "human_reviewed"
    assert ep.extraction["reviewer"] == "test-reviewer"


def test_review_alone_does_not_clear_flagged_uncertainty():
    """Signing off is not the same as resolving. An extraction the model was
    unsure about stays inadmissible until the uncertainty itself is dealt
    with — otherwise review becomes a rubber stamp."""
    eps, _ = asrs.convert(FIXTURE)
    ep = next(e for e in eps if e.extraction.get("uncertain"))
    mark_reviewed(ep, "test-reviewer")
    assert any("uncertain" in p for p in extraction_problems(ep))


def test_iter_admissible_reviewer_does_not_bypass_uncertainty():
    """The gate that uses extraction_problems must ask it even after review.

    A previous `elif` skipped the check whenever --reviewer was set, which
    made review a rubber stamp. The helper test above is not enough: it
    guarded the function, not the iterator that admits to the ledger.
    """
    eps, _ = asrs.convert(FIXTURE)
    assert any(e.extraction.get("uncertain") for e in eps)
    admitted = list(asrs.iter_admissible(eps, reviewer="test-reviewer"))
    assert admitted == [], (
        "unresolved uncertainty must still refuse, even with a reviewer"
    )


def test_resolved_and_reviewed_extraction_becomes_admissible():
    eps, _ = asrs.convert(FIXTURE)
    ep = next(e for e in eps
              if not validate(e, skill_classes=asrs.ASRS_SKILL_CLASSES))
    ep.extraction = dict(ep.extraction)
    ep.extraction["uncertain"] = []          # a human resolved them
    for s in ep.steps:
        if s.action == "unclassified":
            s.action = "hold"                # ...and classified the action
    mark_reviewed(ep, "test-reviewer")
    assert extraction_problems(ep) == []


def test_iter_admissible_after_resolve_and_review():
    """Review plus actually resolving the flags is what opens the gate."""
    eps, _ = asrs.convert(FIXTURE)
    for ep in eps:
        if validate(ep, skill_classes=asrs.ASRS_SKILL_CLASSES):
            continue
        ep.extraction = dict(ep.extraction)
        ep.extraction["uncertain"] = []
        for s in ep.steps:
            if s.action == "unclassified":
                s.action = "hold"
    admitted = list(asrs.iter_admissible(eps, reviewer="test-reviewer"))
    assert {e.episode_id for e in admitted} == {
        "ASRS-1900001", "ASRS-1900002", "ASRS-1900003",
        "ASRS-1900004", "ASRS-1900006",
    }


def test_extraction_provenance_is_sealed_into_the_episode():
    eps, _ = asrs.convert(FIXTURE, model_name="local-llama",
                          weights_sha256="b71c4e" + "0" * 58)
    ep = eps[0]
    ep.seal_record()
    assert ep.verify()
    ep.extraction["weights_sha256"] = "d00d" + "0" * 60
    assert not ep.verify(), "swapping the checkpoint digest must break the seal"


def test_local_model_without_weights_digest_is_flagged():
    eps, _ = asrs.convert(FIXTURE, model_name="local-llama")
    assert any("weights digest" in p for p in extraction_problems(eps[0]))


def test_hosted_model_needs_no_weights_digest():
    eps, _ = asrs.convert(FIXTURE, model_name="claude-sonnet-4-6")
    assert not any("weights digest" in p for p in extraction_problems(eps[0]))


# ------------------------------------------------------------- evidence
def test_stub_reference_is_a_locator_not_a_citation(episodes):
    """The stub writes a pointer to the sentence, unconditionally.

    Treating that as evidence made validate's evidence check vacuous on every
    dataset: a fabricated locator was discharging an evidentiary requirement.
    """
    ref = episodes[0].steps[0].evidence[0]
    assert ref.startswith("src:")
    assert not is_citation(ref)


def test_citation_requirement_is_not_vacuous(episodes):
    """Lax passes, strict names the steps. If both are 0, the check is asleep."""
    lax = sum(len(evidence_problems(e, require_citation=False)) for e in episodes)
    strict = sum(len(evidence_problems(e, require_citation=True)) for e in episodes)
    assert lax == 0
    assert strict > 0, "a corpus of recollection cannot satisfy the strict rule"


def test_evidence_regime_is_recorded(episodes):
    assert all(e.extraction["evidence_regime"] in EVIDENCE_REGIMES
               for e in episodes)


# ---------------------------------------------------------------- honesty
def test_voluntary_collection_regime_is_recorded(episodes):
    """ASRS is voluntary, not compelled. The premise departure is on the
    record rather than quietly elided."""
    assert all(e.extraction["collection_regime"] == "voluntary_immunised"
               for e in episodes)


def test_coded_and_model_derived_fields_are_distinguishable(episodes):
    assert all(e.extraction["coded_fields"]["source"] == "ASRS analyst coding"
               for e in episodes)
