"""The evidence bundle: the artifact you hand a regulator.

This is the commercial product of the whole scheme. Everything upstream exists so
that this file can be produced: a self-contained, checkable answer to the question
that currently blocks autonomous deployment in regulated facilities -- "on what
basis does this system do what it does, and how do you know?"

The bundle answers three questions and refuses to answer a fourth:

  1. What did the system do?        -> the sealed episode chain
  2. What did it learn, and why?    -> per-value provenance to specific episodes
  3. What is it now allowed to do?  -> the certification decision and its test
  4. Is it safe?                    -> not a question this can answer, and the
                                       bundle says so rather than implying it can.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .certify import CertificationReport
from .learn import Policy
from .ledger import Ledger


def build(
    ledger: Ledger,
    policy: Policy,
    report: CertificationReport,
    top_n: int = 25,
) -> dict[str, Any]:
    """Assemble an evidence bundle for one policy version."""
    breaks = ledger.verify_chain()

    ranked = sorted(
        policy.estimates.values(), key=lambda e: (-e.n, -e.mean_return)
    )[:top_n]

    return {
        "bundle_version": "1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": {
            "version": policy.version,
            "skill_class": policy.skill_class,
            "fingerprint": policy.fingerprint,
            "fitted_from_ledger_head": policy.ledger_head,
            "n_episodes_used": policy.n_episodes,
            "estimator": f"first-visit Monte-Carlo, discount {policy.discount}",
            "support": policy.support_summary(),
        },
        "chain_integrity": {
            "episodes": len(ledger),
            "head": ledger.head,
            "intact": not breaks,
            "breaks": [str(b) for b in breaks],
        },
        "certification": {
            "skill_class": report.skill_class,
            "decision": "PROMOTE" if report.promoted else "HOLD",
            "from_tier": report.current_tier,
            "to_tier": report.proposed_tier,
            "test": "Wilson score lower bound, 95%",
            "n_episodes": report.n_episodes,
            "success_rate": round(report.success_rate, 4),
            "success_lower_bound": round(report.success_lower_bound, 4),
            "halt_rate": round(report.halt_rate, 4),
            "reasons": report.reasons,
        },
        "learned_values": [
            {
                "state": e.state,
                "action": e.action,
                "mean_return": e.mean_return,
                "mean_cost": e.mean_cost,
                "n_observations": e.n,
                "caused_by_episodes": e.episode_ids,
                "episode_seals": e.episode_seals,
            }
            for e in ranked
        ],
        "limitations": [
            "This bundle establishes provenance and integrity. It does not "
            "establish safety, and must not be read as a safety case.",
            "Guardrails are fixed and human-owned. Nothing in this bundle "
            "certifies the guardrail envelope itself.",
            "The certification test measures observed success on the sealed "
            "record. It cannot speak to conditions absent from that record.",
            "Reward hacking against audit metrics is an open risk: if the "
            "record is the reward source, a policy can learn to produce "
            "well-formed records rather than good outcomes.",
            "Chain integrity is not causal validity. A perfectly sealed record "
            "can still support a wrong policy through unmeasured confounding. "
            "See Gottesman et al., Nature Medicine 25:16-18 (2019).",
        ],
    }


def write(bundle: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle, indent=2))
    return p


def render_text(bundle: dict[str, Any]) -> str:
    """Human-readable rendering, for the reviewer who will not open JSON."""
    p, c, cert = bundle["policy"], bundle["chain_integrity"], bundle["certification"]
    out = [
        "=" * 66,
        f"EVIDENCE BUNDLE — policy {p['version']} — skill class {p['skill_class']}",
        "=" * 66,
        f"generated        {bundle['generated_utc']}",
        f"fingerprint      {p['fingerprint'][:32]}…",
        f"estimator        {p['estimator']}",
        f"support          median {p['support']['median_n']} obs/value, "
        f"{p['support']['thin']}/{p['support']['values']} thin",
        "",
        "CHAIN INTEGRITY",
        f"  episodes       {c['episodes']}",
        f"  head           {c['head'][:32]}…",
        f"  intact         {'yes' if c['intact'] else 'NO — see breaks'}",
    ]
    for b in c["breaks"]:
        out.append(f"    ! {b}")

    out += [
        "",
        "CERTIFICATION",
        f"  decision       {cert['decision']} ({cert['from_tier']} -> {cert['to_tier']})",
        f"  test           {cert['test']}",
        f"  success        {cert['success_rate']:.1%} "
        f"(lower bound {cert['success_lower_bound']:.1%}) over {cert['n_episodes']} episodes",
        f"  halt rate      {cert['halt_rate']:.1%}",
    ]
    for r in cert["reasons"]:
        out.append(f"    - {r}")

    out += ["", "LEARNED VALUES (most-observed first)"]
    for v in bundle["learned_values"][:10]:
        out.append(
            f"  {v['action']:<10} {v['state'][:34]:<34} "
            f"{v['mean_return']:+.3f}  n={v['n_observations']:<4} "
            f"from {len(v['caused_by_episodes'])} episodes"
        )

    out += ["", "LIMITATIONS"]
    for lim in bundle["limitations"]:
        out.append(f"  - {lim}")
    out.append("=" * 66)
    return "\n".join(out)
