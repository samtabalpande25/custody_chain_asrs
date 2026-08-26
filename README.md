# Custody-Chain RL 

Slice of **Custody-Chain RL**: auditable offline policy learning from operational records. This folder is the NASA Aviation Safety Reporting System (ASRS) path — converting real pilot/controller prose into sealed custody episodes.

It is **not** the full `custody_chain` package. `run_asrs.py` and the tests import modules that live in the parent project (`ledger`, `learn`, `certify`, `audit`, package layout). Drop these files into that tree as:

```
custody_chain/
  extract.py
  schema.py
  adapters/asrs.py
examples/run_asrs.py
tests/test_asrs_adapter.py
tests/fixtures/asrs_sample.csv
```

## What it does

1. Read an ASRS Database Online CSV (two-row header).
2. A model (or the rule-based stub) reads the **narrative** and proposes a sequence of actions.
3. **Outcome** and **guardrail** come only from analyst-coded fields, never from the model.
4. **Reward is always 0.0** — reward design belongs to a deployment, not this adapter.
5. Extractions default to `unreviewed` and are **inadmissible** until a human reviews them.
6. If the rest of the package is present: append to a hash-chained ledger, then `fit` / `certify` / `audit`.

## Get data

Public, no account, no DUA: [ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html)

Export CSV and point the runner at it. A tiny fixture is in `asrs_sample.csv`.

## Run (full package)

```bash
python examples/run_asrs.py --csv data/asrs_export.csv --limit 200
```

First pass is expected to admit nothing (unreviewed extractions). After a human actually reads the extractions:

```bash
python examples/run_asrs.py --csv data/asrs_export.csv --reviewer YOUR_NAME
```

From this folder, with the sample CSV and the stub reader (still needs `custody_chain` on `PYTHONPATH`):

```bash
python run_asrs.py --csv asrs_sample.csv --model stub --out data/asrs_ledger.jsonl
```

Hosted model (needs `anthropic` and `ANTHROPIC_API_KEY`):

```bash
python run_asrs.py --csv asrs_sample.csv --model claude-sonnet-4-6
```

Local checkpoint requires `--weights-sha256` (a model name is not an identity).

## Tests

```bash
pip install pytest
pytest test_asrs_adapter.py
```

Requires the package layout above and the sample CSV at `tests/fixtures/asrs_sample.csv`.

## Files in this folder

| File | Role |
|------|------|
| [schema.py](schema.py) | `CustodyEpisode` / `Step`, hash seal, admissibility |
| [extract.py](extract.py) | LLM or stub reads prose; provenance sealed into the episode |
| [asrs.py](asrs.py) | ASRS CSV adapter; coded fields override the model |
| [run_asrs.py](run_asrs.py) | End-to-end: extract → ledger → fit / certify / audit |
| [test_asrs_adapter.py](test_asrs_adapter.py) | Boundaries: no invented reward, unreviewed = inadmissible |
| [asrs_sample.csv](asrs_sample.csv) | Tiny two-row-header fixture |
