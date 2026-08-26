"""Command-line interface.

The CLI is the evaluable surface. An organisation assessing this runs four
commands in order and sees whether the claims hold:

    custody verify   ledger.jsonl
    custody fit      ledger.jsonl --skill navigate -o policy.json
    custody certify  ledger.jsonl --skill navigate --tier observe
    custody audit    ledger.jsonl --policy policy.json --skill navigate

Exit codes are meaningful so this can sit in CI: a broken chain or a failed
certification gate is a non-zero exit, which means a pipeline can refuse to
promote a policy that has not earned it.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import audit as audit_mod
from . import certify as certify_mod
from . import learn as learn_mod
from .ledger import Ledger


def cmd_verify(args) -> int:
    ledger = Ledger.load(args.ledger)
    breaks = ledger.verify_chain()

    print(f"episodes  {len(ledger)}")
    print(f"head      {ledger.head[:32]}…")

    if not breaks:
        print("chain     INTACT")
        return 0

    print(f"chain     BROKEN — {len(breaks)} problem(s)")
    for b in breaks:
        print(f"  ! {b}")
    return 1


def cmd_fit(args) -> int:
    ledger = Ledger.load(args.ledger)
    try:
        policy = learn_mod.fit(
            ledger,
            skill_class=args.skill,
            version=args.version,
            discount=args.discount,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"skill class      {policy.skill_class}")
    print(f"episodes used    {policy.n_episodes}")
    print(f"states fitted    {len(policy.states())}")
    print(f"values fitted    {len(policy.estimates)}")
    print(f"fingerprint      {policy.fingerprint[:32]}…")

    sup = policy.support_summary()
    print(f"median obs/value {sup['median_n']}")
    if sup["thin_fraction"] > 0.5:
        print(
            f"\nWARNING: {sup['thin']} of {sup['values']} fitted values rest on "
            f"fewer than {sup['thin_below']} observations.\n"
            "The state abstraction is too fine for this much data. Most of these "
            "numbers are noise.\nCoarsen the state or collect more episodes before "
            "treating this table as findings."
        )

    if args.out:
        policy.save(args.out)
        print(f"written to       {args.out}")
    return 0


def cmd_certify(args) -> int:
    ledger = Ledger.load(args.ledger)
    report = certify_mod.evaluate(
        ledger, skill_class=args.skill, current_tier=args.tier
    )
    print(report.summary())
    return 0 if report.promoted else 2


def cmd_audit(args) -> int:
    ledger = Ledger.load(args.ledger)
    policy = learn_mod.Policy.load(args.policy)
    report = certify_mod.evaluate(
        ledger, skill_class=args.skill, current_tier=args.tier
    )
    bundle = audit_mod.build(ledger, policy, report)

    if args.json:
        print(json.dumps(bundle, indent=2))
    else:
        print(audit_mod.render_text(bundle))

    if args.out:
        audit_mod.write(bundle, args.out)
        print(f"\nbundle written to {args.out}")

    return 0 if bundle["chain_integrity"]["intact"] else 1


def cmd_why(args) -> int:
    """Explain one learned value: the provenance query, from the terminal."""
    policy = learn_mod.Policy.load(args.policy)
    est = policy.provenance(args.state, args.action)
    if est is None:
        print(f"no fitted value for ({args.state!r}, {args.action!r})", file=sys.stderr)
        return 1

    print(f"state            {est.state}")
    print(f"action           {est.action}")
    print(f"mean return      {est.mean_return:+.6f}")
    print(f"mean cost        {est.mean_cost:.6f}")
    print(f"observations     {est.n}")
    print(f"caused by        {len(set(est.episode_ids))} episode(s):")
    for eid, seal in zip(est.episode_ids, est.episode_seals):
        print(f"  {eid}  seal {seal[:16]}…")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="custody", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="check ledger integrity")
    v.add_argument("ledger")
    v.set_defaults(fn=cmd_verify)

    f = sub.add_parser("fit", help="fit a policy from sealed episodes")
    f.add_argument("ledger")
    f.add_argument("--skill", required=True)
    f.add_argument("--version", default="v1.0")
    f.add_argument("--discount", type=float, default=0.95)
    f.add_argument("-o", "--out")
    f.set_defaults(fn=cmd_fit)

    c = sub.add_parser("certify", help="evaluate an autonomy promotion")
    c.add_argument("ledger")
    c.add_argument("--skill", required=True)
    c.add_argument("--tier", default="observe")
    c.set_defaults(fn=cmd_certify)

    a = sub.add_parser("audit", help="produce an evidence bundle")
    a.add_argument("ledger")
    a.add_argument("--policy", required=True)
    a.add_argument("--skill", required=True)
    a.add_argument("--tier", default="observe")
    a.add_argument("--json", action="store_true")
    a.add_argument("-o", "--out")
    a.set_defaults(fn=cmd_audit)

    w = sub.add_parser("why", help="show provenance for one learned value")
    w.add_argument("--policy", required=True)
    w.add_argument("--state", required=True)
    w.add_argument("--action", required=True)
    w.set_defaults(fn=cmd_why)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
