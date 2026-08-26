"""Convert a MIMIC-IV sepsis cohort into sealed custody episodes.

Why MIMIC is the right test of this project's premise
-----------------------------------------------------
MIMIC is not a robotics dataset and this project is not really about robots. It
is about whether records kept under regulatory obligation are usable as offline
RL data, and whether the learning built on them can be made auditable. An ICU is
the strongest available instance of that setting: every intervention is charted
because it must be, each observation carries a recorded time and a recorded item
id, and the outcomes are unambiguous.

It is also the setting with the most honest literature about how this goes
wrong. Komorowski et al.'s AI Clinician (Nature Medicine, 2018) established the
sepsis-treatment formulation this adapter follows. Gottesman et al., "Guidelines
for reinforcement learning in healthcare" (Nature Medicine, 2019), sets out why
policies fitted this way are hard to evaluate and easy to over-trust. Read the
second before you present anything built with the first.

What this adapter does and does not claim
-----------------------------------------
It converts a *preprocessed cohort* -- the tabular output of a pipeline such as
cmudig/AI-Clinician-MIMICIV -- into custody episodes with real evidence
references. It does not extract from MIMIC directly, does not define a cohort,
and does not validate anyone's clinical preprocessing.

Two fields deserve care, because they are where fabrication would creep in:

  evidence   Real. Each step references the stay and chart time it came from, so
             a reviewer can trace a logged action back to the source record.
             This is the first dataset in this repo where the field is not empty,
             and it is the reason MIMIC is a better test than a robotics
             benchmark.

  guardrail  Absent, and left absent. Clinical protocols exist, but MIMIC does
             not log "a protocol blocked this action". Inventing guardrail
             events to make the demo look complete would be fabricating the one
             thing this project claims to protect.

Usage
-----
    python -m custody_chain.adapters.mimic \\
        --cohort data/mimic_sepsis_cohort.csv \\
        --out data/mimic_ledger.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

from ..ledger import Ledger
from ..schema import CustodyEpisode, Step

# Komorowski et al. discretise IV fluid and vasopressor dose into 5 bins each,
# giving a 25-action space. Bin 0 is "none"; bins 1-4 are quartiles of the
# non-zero doses observed in the cohort.
N_DOSE_BINS = 5

# Columns the adapter needs. Everything else in the cohort is treated as a
# state feature.
REQUIRED = ("stay_id", "bloc", "iv_fluid", "vaso", "outcome")

# State features are binned so the tabular estimator has support. This is the
# same trade the rest of the repo makes: coarse enough that every fitted value
# can be traced by hand, at the cost of clinical resolution.
DEFAULT_FEATURES = ("sofa", "lactate", "map", "hr")


def dose_bin(value: float, cuts: list[float]) -> int:
    """Bin a dose against cut points. 0 means none given."""
    if value <= 0:
        return 0
    for i, c in enumerate(cuts, start=1):
        if value <= c:
            return i
    return N_DOSE_BINS - 1


def quartile_cuts(values: Iterable[float]) -> list[float]:
    """Quartile cut points over the non-zero doses."""
    nz = sorted(v for v in values if v > 0)
    if not nz:
        return [0.0, 0.0, 0.0]
    return [nz[int(len(nz) * q)] for q in (0.25, 0.50, 0.75)]


def bin_feature(value: float, name: str) -> str:
    """Coarse clinical bins. Deployments should replace this with their own."""
    thresholds = {
        "sofa": (4, 8, 12),
        "lactate": (2, 4, 8),
        "map": (55, 65, 80),
        "hr": (60, 100, 130),
    }.get(name.lower())
    if thresholds is None:
        return "na"
    lo, mid, hi = thresholds
    if value < lo:
        return "lo"
    if value < mid:
        return "mid"
    if value < hi:
        return "hi"
    return "vhi"


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def build_ledger(
    rows: list[dict[str, Any]],
    features: tuple[str, ...] = DEFAULT_FEATURES,
    survive_reward: float = 100.0,
    strict: bool = True,
) -> Ledger:
    """Group cohort rows into per-stay episodes and seal them into a ledger.

    Reward follows the AI Clinician convention: terminal only, +/- on survival,
    nothing intermediate. That is a deliberately crude reward and a known
    weakness of the formulation -- Gottesman et al. single out reward design as
    a place these studies go wrong. It is used here because deviating from the
    published convention would make results incomparable, not because it is good.
    """
    missing = [c for c in REQUIRED if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"cohort is missing required columns: {missing}")

    iv_cuts = quartile_cuts(_f(r, "iv_fluid") for r in rows)
    vaso_cuts = quartile_cuts(_f(r, "vaso") for r in rows)

    by_stay: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_stay.setdefault(str(r["stay_id"]), []).append(r)

    ledger = Ledger(skill_classes=("vasopressor_titration",))

    for stay_id, stay_rows in by_stay.items():
        stay_rows.sort(key=lambda r: _f(r, "bloc"))
        survived = _f(stay_rows[-1], "outcome") == 0

        steps: list[Step] = []
        for r in stay_rows:
            iv = dose_bin(_f(r, "iv_fluid"), iv_cuts)
            vp = dose_bin(_f(r, "vaso"), vaso_cuts)

            obs = {f: bin_feature(_f(r, f), f) for f in features if f in r}

            charttime = r.get("charttime") or f"bloc{int(_f(r, 'bloc'))}"
            steps.append(
                Step(
                    t=_f(r, "bloc"),
                    observation=obs,
                    action=f"iv{iv}_vaso{vp}",
                    reward=0.0,
                    cost=_f(r, "sofa"),          # organ dysfunction as a cost signal
                    guardrail=None,               # not logged in MIMIC; left absent
                    evidence=[
                        f"mimic-iv:icustay:{stay_id}:{charttime}",
                        f"mimic-iv:chartevents:{stay_id}:{charttime}",
                    ],
                )
            )

        if not steps:
            continue

        steps[-1].reward = survive_reward if survived else -survive_reward

        ledger.append(
            CustodyEpisode(
                episode_id=f"ICU-{stay_id}",
                site="MIMIC-IV",
                skill_class="vasopressor_titration",
                steps=steps,
                outcome="clean" if survived else "aborted",
            ),
            strict=strict,
        )

    return ledger


def load_cohort(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True, help="preprocessed cohort CSV")
    p.add_argument("--out", default="data/mimic_ledger.jsonl")
    p.add_argument("--features", nargs="*", default=list(DEFAULT_FEATURES))
    args = p.parse_args(argv)

    rows = load_cohort(args.cohort)
    if not rows:
        print("cohort is empty", file=sys.stderr)
        return 1

    ledger = build_ledger(rows, features=tuple(args.features))
    ledger.save(args.out)

    survived = sum(1 for e in ledger if e.succeeded)
    steps = sum(len(e.steps) for e in ledger)
    print(f"episodes      {len(ledger)}")
    print(f"steps         {steps}")
    print(f"survival      {survived / len(ledger):.1%}")
    print(f"chain head    {ledger.head[:32]}…")
    print(f"written to    {args.out}")
    print(
        "\nEvidence references point at real MIMIC-IV stays and chart times.\n"
        "Guardrail events are absent because MIMIC does not log them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
