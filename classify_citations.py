#!/usr/bin/env python3
"""Separate cited artifacts from restated narrative, without calling a model.

The problem
-----------
`citation_coverage` counts steps whose evidence list is non-empty. That is a
test of whether a field was filled, not of whether anything was cited. Reading
the dump by hand makes the difference obvious:

    QRH                                             a document
    NOTAMs_section_of_release                       a document
    secondary_engine_display_on_MFD                 an instrument reading
    I_believe_this_is_what_caused_the_5NL_attitude  the reporter's speculation
    crew_also_did_not_feel_well                     the narrative restating itself

Only the first three are things a reviewer could go and check.

Two independent cuts, deliberately conservative
-----------------------------------------------
RECURRENCE  A shared artifact gets named the same way by different reporters.
            A paraphrase of one incident's sentence cannot repeat, because no
            other reporter wrote that sentence. Recurrence across *episodes* is
            therefore evidence of an artifact, not of a common phrasing.

VOCABULARY  A name-list of records, instruments, publications, and system
            outputs. Independent of recurrence, so a genuine artifact named
            once can still be caught.

Both cuts miss things. A real artifact mentioned once, in words the list does
not contain, is counted as a restatement. So the figure this prints is a LOWER
BOUND on citation quality and must be reported as one.

    python3 classify_citations.py data/asrs_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CITE = "cite:"
ADAPTER_TAILS = {"coded_result"}

#: Records, instruments, publications, and system outputs an auditor could pull.
#: Actor and action words (ATC, dispatch, clearance, procedure) are deliberately
#: absent: "notified_ATC" names an event, not a document.
ARTIFACT_TERMS = {
    # publications and records
    "qrh", "notam", "notams", "checklist", "manual", "logbook", "release",
    "placard", "placards", "mel", "cdl", "sop", "sops", "fom", "far", "afm",
    "poh", "bulletin", "chart", "plate", "worksheet", "workorder",
    "work_order", "paperwork", "form", "manifest", "guidelines", "logbooks",
    # instruments, displays, and system outputs
    "eicas", "ecam", "mfd", "pfd", "cdu", "fms", "fmc", "mcp", "acars",
    "atis", "tcas", "gpws", "egpws", "fdr", "cvr", "transponder",
    "altimeter", "annunciator", "display", "indication", "indicator",
    "gauge", "readout", "printout", "metar", "taf",
}

#: Markers that the string is the narrative talking about itself.
_FIRST_PERSON = re.compile(r"(^|_)(i|we|my|our|us|me)(_|$)", re.I)
_SPECULATION = re.compile(
    r"(^|_)(believe|believed|think|thought|feel|felt|seem|seemed|assume|assumed"
    r"|probably|likely|maybe|perhaps|might|unsure|guess|suspect|suspected)(_|$)",
    re.I,
)
MAX_CHARS = 45
MAX_WORDS = 6


def tail_of(ref: str) -> str:
    return ref.split(":", 2)[-1]


def load(path: Path):
    """Return {citation_tail: {episode_ids}} for model-contributed citations."""
    by_tail: dict[str, set[str]] = defaultdict(set)
    total = adapter = 0
    episodes = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            episodes += 1
            for step in ep.get("steps", []):
                for ref in step.get("evidence", []):
                    if not ref.startswith(CITE):
                        continue
                    total += 1
                    tail = tail_of(ref)
                    if tail in ADAPTER_TAILS:
                        adapter += 1
                        continue
                    by_tail[tail].add(ep["episode_id"])
    return by_tail, total, adapter, episodes


def looks_restated(tail: str) -> bool:
    """First-person and speculation disqualify outright; length only if unnamed.

    A document title can be long -- "Loss_of_Oil_Pressure_Right_Non-normal_
    Checklist" is an artifact -- so the length cap applies only to strings that
    name nothing from the vocabulary.
    """
    if _FIRST_PERSON.search(tail) or _SPECULATION.search(tail):
        return True
    if in_vocabulary(tail):
        return False
    words = tail.split("_")
    return len(tail) > MAX_CHARS or len(words) > MAX_WORDS


def in_vocabulary(tail: str) -> bool:
    return any(w.strip(".,()").lower() in ARTIFACT_TERMS for w in tail.split("_"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", default="data/asrs_candidates.jsonl")
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.dump)
    if not path.exists():
        raise SystemExit(f"no such dump: {path}")

    by_tail, total, adapter, episodes = load(path)
    model_total = sum(len(v) for v in by_tail.values())

    recurrent, vocab, restated, neither = set(), set(), set(), set()
    for tail, eps in by_tail.items():
        if len(eps) >= 2:
            recurrent.add(tail)
        if looks_restated(tail):
            restated.add(tail)
        elif in_vocabulary(tail):
            vocab.add(tail)
        else:
            neither.add(tail)

    # Recurrence is reported but not folded in: generic phrasing recurs too
    # ("landed_without_incident"), so it cannot carry the headline on its own.
    artifact = vocab
    distinct = len(by_tail)
    occ = {t: len(e) for t, e in by_tail.items()}
    art_occ = sum(occ[t] for t in artifact)

    print(f"episodes {episodes}   citations {total}"
          f"   (model {model_total}, adapter {adapter})")
    print(f"distinct model citations {distinct}\n")

    print("classification of distinct strings")
    print(f"  names an artifact        {len(vocab):5}  ({100*len(vocab)/distinct:.1f}%)")
    print(f"  restated narrative       {len(restated):5}  ({100*len(restated)/distinct:.1f}%)")
    print(f"  neither                  {len(neither):5}  ({100*len(neither)/distinct:.1f}%)")
    print(f"\n  recurs across episodes   {len(recurrent):5}  "
          f"(weak signal, contaminated by generic phrasing)")
    print(f"  recurrent and named      {len(recurrent & vocab):5}")

    print(f"\nby occurrence: {art_occ} of {model_total} model citations "
          f"({100*art_occ/model_total:.1f}%) name an artifact")

    rng = random.Random(args.seed)
    for label, group in (("names an artifact", vocab),
                         ("restated narrative", restated),
                         ("neither", neither)):
        if not group:
            continue
        picks = rng.sample(sorted(group), min(args.sample, len(group)))
        print(f"\n{label} — sample")
        for p in picks:
            print(f"  {p}")

    print("\nmost recurrent (distinct episodes)")
    for tail, n in Counter(occ).most_common(10):
        if tail in artifact:
            print(f"  {n:3}  {tail}")

    print(
        "\nReport this as a LOWER BOUND. Both cuts are conservative: a genuine\n"
        "artifact named once, in words the list does not carry, lands in\n"
        "'neither'. The number worth quoting is that citation_coverage tested\n"
        "whether a field was non-empty, and this tests whether a reviewer could\n"
        "go and pull the thing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
