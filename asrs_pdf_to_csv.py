#!/usr/bin/env python3
"""Convert an ASRS Database Report Set PDF into the CSV shape the adapter reads.

NASA publishes two things. The Database Online query builder exports CSV, which
`adapters/asrs.py` already consumes. The topical Report Sets
(https://asrs.arc.nasa.gov/publications/reportsets.html) are PDFs of the same
underlying records — 50 analyst-screened reports each, with the coded fields
intact.

    python3 asrs_pdf_to_csv.py acr_fatg.pdf flt_attendant.pdf -o data/asrs_real.csv

This writes the two-row header the adapter expects, so nothing downstream
changes: the model still reads only `Narrative`, and outcome and guardrail are
still lifted from the analyst-coded `Events / Result` and `Events / Detector`.

One honesty note, and it is the same argument this project makes about models.
Extracting text from a PDF is itself a reading, and this converter sits *outside*
the seal. It is a preprocessing step whose fidelity is asserted, not proven. The
coded fields survive it verbatim, which is what matters for the coded/prose
split — but if you present numbers from this path, say that a parser produced
the CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# `ACN: 2083422 (2 of 50)` opens every record block.
_ACN = re.compile(r"^ACN:\s*(\d+)\s*\((\d+)\s+of\s+(\d+)\)", re.M)

# Coded fields arrive as `Key : Value`, sometimes dotted: `Result.Flight Crew : Diverted`.
_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ./\-]*?)\s*:\s*(.+?)\s*$")

# A narrative runs until the next section header.
_NARR_START = re.compile(r"^\s*Narrative\s*:?\s*\d*\s*$", re.I)
_SECTION = re.compile(r"^\s*(Synopsis|Callback|Narrative|ACN)\b", re.I)


def _pages(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("pypdf is required:  python3 -m pip install --user pypdf")
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)


def _unwrap(lines: list[str]) -> str:
    """Join PDF hard-wraps back into flowing prose.

    The line breaks in a Report Set PDF are typography, not structure, so
    keeping them would hand the reader a segmentation signal the reporter never
    wrote.
    """
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _split_records(text: str) -> list[tuple[str, str]]:
    """Return (acn, block) for every record that actually carries a narrative.

    The front of each Report Set lists all 50 ACNs with synopses only, then the
    full records repeat those ACNs. Keeping the longest block per ACN keeps the
    full record and drops the synopsis stub.
    """
    marks = list(_ACN.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        block = text[m.end():end]
        acn = m.group(1)
        if "Narrative" not in block:
            continue
        if len(block) > len(best.get(acn, "")):
            best[acn] = block
    return sorted(best.items(), key=lambda kv: int(kv[0]))


def _parse_block(acn: str, block: str) -> dict[str, str]:
    lines = block.splitlines()
    coded: dict[str, list[str]] = {}
    narrative: list[str] = []
    collecting = False

    for line in lines:
        if _NARR_START.match(line):
            collecting = True
            continue
        if collecting:
            if _SECTION.match(line):
                collecting = False
            else:
                if line.strip():
                    narrative.append(line.strip())
                continue

        m = _FIELD.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        # `Anomaly.Aircraft Equipment Problem : Critical` carries meaning in both
        # halves, so the dotted suffix is folded into the value.
        if "." in key:
            head, _, tail = key.partition(".")
            key, val = head.strip(), f"{tail.strip()} {val}".strip()
        coded.setdefault(key, []).append(val)

    def get(name: str) -> str:
        return "; ".join(dict.fromkeys(coded.get(name, [])))

    return {
        "ACN": acn,
        "Date": get("Date"),
        "Locale Reference": get("Locale Reference") or "ZZZ",
        "Flight Phase": get("Flight Phase"),
        "Function": get("Function"),
        "Anomaly": get("Anomaly"),
        "Detector": get("Detector"),
        "Result": get("Result"),
        "Narrative": _unwrap(narrative),
    }


GROUPS = ["", "Time", "Place", "Aircraft 1", "Person 1", "Events", "", "", "Report 1"]
FIELDS = ["ACN", "Date", "Locale Reference", "Flight Phase", "Function",
          "Anomaly", "Detector", "Result", "Narrative"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", help="ASRS Report Set PDF(s)")
    ap.add_argument("-o", "--out", default="data/asrs_real.csv")
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    for pdf in args.pdfs:
        path = Path(pdf)
        if not path.exists():
            sys.exit(f"no such file: {path}")
        found = _split_records(_pages(path))
        for acn, block in found:
            rec = _parse_block(acn, block)
            if rec["Narrative"]:
                rows.append(rec)
        print(f"{path.name}: {len(found)} record(s)")

    seen: set[str] = set()
    unique = [r for r in rows if not (r["ACN"] in seen or seen.add(r["ACN"]))]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(GROUPS)
        w.writerow(FIELDS)
        for r in unique:
            w.writerow([r[f] for f in FIELDS])

    missing = sum(1 for r in unique if not r["Result"])
    print(f"\nwrote {len(unique)} record(s) to {out}")
    print(f"  without a coded Result  {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
