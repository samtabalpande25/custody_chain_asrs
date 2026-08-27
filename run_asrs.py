#!/usr/bin/env python
"""Run the full pipeline over a NASA ASRS export.

    python examples/run_asrs.py --csv data/asrs_export.csv --limit 200

This is the only path in the repo where a language model reads prose that a
human wrote for other humans. Everything else consumes records that were
already structured, which means everything else quietly assumes away the step
that actually blocks deployment.

The run is expected to end in a refusal on first pass. Extractions default to
unreviewed, unreviewed extractions are inadmissible, and `fit` will not train
on records whose integrity is unestablished. Pass --reviewer only once a human
has genuinely read the extractions; the flag records an amendment, and a false
amendment is worse than no chain at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from custody_chain import audit, certify, learn                    # noqa: E402
from custody_chain.adapters import asrs                            # noqa: E402
from custody_chain.extract import anthropic_completion, rule_based_completion  # noqa: E402
from custody_chain.ledger import Ledger                            # noqa: E402


def build_completion(args):
    """Pick the reader, and describe it honestly for the seal."""
    if args.model == "stub":
        return rule_based_completion, "rule-based-stub", None
    if args.model.startswith("claude-"):
        return anthropic_completion(args.model), args.model, None
    # Local checkpoint: the name is not an identity, the digest is.
    if not args.weights_sha256:
        sys.exit(
            f"refusing to run: local model {args.model!r} needs --weights-sha256.\n"
            "  A model name does not identify which weights did the reading."
        )
    raise SystemExit(
        "wire your local provider here — return a complete_fn(prompt) -> str"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ASRS Database Online CSV export")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skill", default="separation", choices=list(asrs.ASRS_SKILL_CLASSES))
    ap.add_argument("--model", default="stub",
                    help="'stub', a hosted model id (claude-*), or a local checkpoint name")
    ap.add_argument("--weights-sha256", default=None,
                    help="SHA-256 of the checkpoint; required for local models")
    ap.add_argument("--reviewer", default=None,
                    help="mark extractions reviewed and re-seal — only if a human read them")
    ap.add_argument("--out", default="data/asrs_ledger.jsonl")
    args = ap.parse_args()

    complete_fn, model_name, decode = build_completion(args)

    print(f"reading {args.csv}")
    episodes, stats = asrs.convert(
        args.csv,
        complete_fn=complete_fn,
        model_name=model_name,
        limit=args.limit,
        weights_sha256=args.weights_sha256,
        decode_params=decode,
    )
    print("\nextraction")
    print(stats.render())

    if stats.problems:
        print("\n  sample problems:")
        for p in stats.problems[:5]:
            print(f"    ! {p}")

    # ---- ledger --------------------------------------------------------
    led = Ledger(skill_classes=asrs.ASRS_SKILL_CLASSES)
    admitted = rejected = 0
    for ep in asrs.iter_admissible(episodes, reviewer=args.reviewer):
        try:
            led.append(ep)
            admitted += 1
        except Exception as exc:                       # ledger refused it
            rejected += 1
            if rejected <= 3:
                print(f"    ! ledger refused {ep.episode_id}: {exc}")

    print(f"\nledger\n  admitted {admitted}\n  refused  {rejected}")
    if not admitted:
        if args.reviewer:
            print(
                "\n  nothing admitted. Review is necessary, not sufficient:\n"
                "  --reviewer was set, but extraction_problems still refused\n"
                "  every candidate (unresolved uncertainty, unclassified\n"
                "  actions, missing evidence, or an undeclared skill class).\n"
                "  Resolve those flags on the record; do not clear them by\n"
                "  passing a name."
            )
        else:
            print(
                "\n  nothing admitted. This is the expected first-pass result:\n"
                "  a model's reading of a log is a hypothesis until a human checks it.\n"
                "  Review the extractions, then re-run with --reviewer YOUR_NAME."
            )
        return 1

    path = led.save(args.out)
    print(f"  saved {path}")

    breaks = led.verify_chain()
    print(f"  chain {'INTACT' if not breaks else 'BROKEN'}")

    # ---- learn, certify, audit ----------------------------------------
    n_skill = sum(1 for e in led.episodes if e.skill_class == args.skill)
    if not n_skill:
        print(f"\nno '{args.skill}' episodes admitted — nothing to fit")
        return 0

    pol = learn.fit(led, skill_class=args.skill)
    rep = certify.evaluate(led, args.skill)
    bundle = audit.build(led, pol, rep)

    out = pathlib.Path(args.out).with_name("asrs_bundle.json")
    out.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"\nfit       {args.skill}: {n_skill} episode(s)")
    print("certify")
    print(rep.summary() if hasattr(rep, "summary") else rep)
    print(f"bundle    {out}")

    print(
        "\nNote: every reward in this ledger is 0.0. The adapter does not invent\n"
        "reward, so the fitted values are structurally meaningful and numerically\n"
        "empty until a deployment supplies a reward specification. That is the\n"
        "correct division of labour, not an oversight."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
