"""Tests for the custody chain.

Run with: python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from custody_chain import audit, certify, learn  # noqa: E402
from custody_chain.ledger import Ledger  # noqa: E402
from custody_chain.schema import GENESIS, CustodyEpisode, Step, validate  # noqa: E402


def make_step(action="proceed", reward=0.1, cost=0.0, guardrail=None, **obs):
    return Step(
        t=0.0,
        observation=obs or {"blocked_ahead": 0, "staff_near": 0},
        action=action,
        reward=reward,
        cost=cost,
        guardrail=guardrail,
        evidence=["cam/0001.jpg"],
    )


def make_episode(eid="HA-0001", skill="navigate", outcome="clean", steps=None):
    return CustodyEpisode(
        episode_id=eid,
        site="Hospital A",
        skill_class=skill,
        steps=steps if steps is not None else [make_step()],
        outcome=outcome,
    )


# ---------------------------------------------------------------- schema
def test_wellformed_episode_is_admissible():
    assert validate(make_episode()) == []


def test_step_without_evidence_is_inadmissible():
    ep = make_episode()
    ep.steps[0].evidence = []
    assert any("no evidence" in p for p in validate(ep))


def test_undeclared_skill_class_is_rejected():
    assert any("undeclared skill class" in p for p in validate(make_episode(skill="teleport")))


def test_return_is_derived_not_stored():
    ep = make_episode(steps=[make_step(reward=0.5), make_step(reward=0.25)])
    assert ep.episode_return == pytest.approx(0.75)


# ---------------------------------------------------------------- sealing
def test_seal_verifies_then_detects_edit():
    ep = make_episode()
    ep.seal_record()
    assert ep.verify()
    ep.steps[0].reward = 99.0
    assert not ep.verify()


def test_unsealed_episode_does_not_verify():
    assert not make_episode().verify()


# ---------------------------------------------------------------- ledger
def test_ledger_links_episodes_into_a_chain():
    led = Ledger()
    a = led.append(make_episode("HA-0001"))
    b = led.append(make_episode("HA-0002"))
    assert a.prev_seal == GENESIS
    assert b.prev_seal == a.seal
    assert led.verify_chain() == []


def test_ledger_refuses_inadmissible_episode():
    ep = make_episode()
    ep.steps[0].evidence = []
    with pytest.raises(ValueError, match="inadmissible"):
        Ledger().append(ep)


def test_editing_a_stored_episode_breaks_the_chain():
    led = Ledger()
    for i in range(5):
        led.append(make_episode(f"HA-{i:04d}"))
    assert led.verify_chain() == []

    led.episodes[2].steps[0].reward = 42.0
    breaks = led.verify_chain()
    assert breaks and breaks[0].index == 2


def test_removing_an_episode_breaks_the_link():
    led = Ledger()
    for i in range(5):
        led.append(make_episode(f"HA-{i:04d}"))
    del led.episodes[2]
    assert any("broken link" in b.reason for b in led.verify_chain())


def test_ledger_roundtrips_through_disk(tmp_path):
    led = Ledger()
    for i in range(4):
        led.append(make_episode(f"HA-{i:04d}"))
    path = led.save(tmp_path / "l.jsonl")

    back = Ledger.load(path)
    assert len(back) == 4
    assert back.head == led.head
    assert back.verify_chain() == []


# ---------------------------------------------------------------- learning
def _populated_ledger(n=40, skill="navigate"):
    led = Ledger()
    for i in range(n):
        good = i % 2 == 0
        led.append(
            CustodyEpisode(
                episode_id=f"HA-{i:04d}",
                site="Hospital A",
                skill_class=skill,
                steps=[
                    make_step("proceed" if good else "wait",
                              reward=0.5 if good else -0.3,
                              blocked_ahead=0, staff_near=0)
                ],
                outcome="clean" if good else "aborted",
            )
        )
    return led


def test_fit_produces_values_with_provenance():
    led = _populated_ledger()
    pol = learn.fit(led, skill_class="navigate")

    est = pol.provenance("blocked_ahead=0·staff_near=0", "proceed")
    assert est is not None
    assert est.n == 20
    assert len(est.episode_ids) == 20
    assert all(s for s in est.episode_seals)


def test_fit_prefers_the_better_action():
    pol = learn.fit(_populated_ledger(), skill_class="navigate")
    assert pol.best_action("blocked_ahead=0·staff_near=0") == "proceed"


def test_fit_refuses_a_broken_ledger():
    led = _populated_ledger()
    led.episodes[3].steps[0].reward = 99.0
    with pytest.raises(ValueError, match="broken ledger"):
        learn.fit(led, skill_class="navigate")


def test_fingerprint_changes_with_the_data():
    a = learn.fit(_populated_ledger(20), skill_class="navigate")
    b = learn.fit(_populated_ledger(40), skill_class="navigate")
    assert a.fingerprint != b.fingerprint


def test_policy_roundtrips_through_disk(tmp_path):
    pol = learn.fit(_populated_ledger(), skill_class="navigate")
    back = learn.Policy.load(pol.save(tmp_path / "p.json"))
    assert back.fingerprint == pol.fingerprint
    assert back.best_action("blocked_ahead=0·staff_near=0") == "proceed"


# ---------------------------------------------------------------- certification
def test_wilson_bound_is_conservative_on_small_samples():
    # 4/4 successes is a raw 100%; the honest reading is far lower.
    assert certify.wilson_lower_bound(4, 4) < 0.60
    assert certify.wilson_lower_bound(400, 400) > 0.98


def test_wilson_bound_handles_empty_sample():
    assert certify.wilson_lower_bound(0, 0) == 0.0


def test_small_sample_is_not_promoted_however_clean():
    led = Ledger()
    for i in range(4):
        led.append(make_episode(f"HA-{i:04d}"))
    report = certify.evaluate(led, "navigate", current_tier="observe")
    assert not report.promoted
    assert any("episodes" in r for r in report.reasons)


def test_promotion_granted_on_sufficient_clean_record():
    led = Ledger()
    for i in range(60):
        led.append(make_episode(f"HA-{i:04d}"))
    report = certify.evaluate(led, "navigate", current_tier="observe")
    assert report.promoted
    assert report.proposed_tier == "propose"


def test_halts_block_promotion():
    led = Ledger()
    for i in range(60):
        led.append(
            make_episode(
                f"HA-{i:04d}",
                steps=[make_step(guardrail="halt" if i % 2 else None)],
            )
        )
    report = certify.evaluate(led, "navigate", current_tier="observe")
    assert not report.promoted
    assert any("halt rate" in r for r in report.reasons)


# ---------------------------------------------------------------- audit
def test_bundle_reports_integrity_and_limitations():
    led = _populated_ledger()
    pol = learn.fit(led, skill_class="navigate")
    rep = certify.evaluate(led, "navigate")
    bundle = audit.build(led, pol, rep)

    assert bundle["chain_integrity"]["intact"] is True
    assert bundle["policy"]["fingerprint"] == pol.fingerprint
    assert bundle["learned_values"][0]["caused_by_episodes"]
    # The bundle must never imply it is a safety case.
    assert any("not establish safety" in l for l in bundle["limitations"])


def test_bundle_surfaces_a_broken_chain():
    led = _populated_ledger()
    pol = learn.fit(led, skill_class="navigate")
    rep = certify.evaluate(led, "navigate")
    led.episodes[1].steps[0].reward = 7.0     # tamper after fitting
    bundle = audit.build(led, pol, rep)

    assert bundle["chain_integrity"]["intact"] is False
    assert bundle["chain_integrity"]["breaks"]


# ---------------------------------------------------------------- support diagnostics
def test_support_summary_flags_thin_evidence():
    led = Ledger()
    # 30 episodes, each in its own state -> one observation per value
    for i in range(30):
        led.append(
            CustodyEpisode(
                episode_id=f"HA-{i:04d}",
                site="Hospital A",
                skill_class="navigate",
                steps=[make_step("proceed", reward=0.1, unique_state=i)],
                outcome="clean",
            )
        )
    pol = learn.fit(led, skill_class="navigate")
    sup = pol.support_summary()

    assert sup["values"] == 30
    assert sup["median_n"] == 1
    assert sup["thin_fraction"] == 1.0


def test_support_summary_is_clean_when_evidence_is_thick():
    pol = learn.fit(_populated_ledger(80), skill_class="navigate")
    assert pol.support_summary()["thin_fraction"] == 0.0


def test_support_summary_handles_empty_policy():
    sup = learn.Policy(version="v0", skill_class="navigate", discount=0.95).support_summary()
    assert sup["values"] == 0


def test_bundle_carries_support_and_causal_caveat():
    led = _populated_ledger()
    pol = learn.fit(led, skill_class="navigate")
    rep = certify.evaluate(led, "navigate")
    bundle = audit.build(led, pol, rep)

    assert "support" in bundle["policy"]
    assert any("causal validity" in l for l in bundle["limitations"])


# ---------------------------------------------------------------- declared skill classes
def test_ledger_accepts_a_deployment_specific_skill_class():
    led = Ledger(skill_classes=("vasopressor_titration",))
    led.append(make_episode("ICU-0001", skill="vasopressor_titration"))
    assert len(led) == 1


def test_loaded_ledger_keeps_the_classes_it_was_written_with(tmp_path):
    led = Ledger(skill_classes=("vasopressor_titration",))
    led.append(make_episode("ICU-0001", skill="vasopressor_titration"))
    path = led.save(tmp_path / "icu.jsonl")

    back = Ledger.load(path)
    assert back.verify_chain() == []
    back.append(make_episode("ICU-0002", skill="vasopressor_titration"))
