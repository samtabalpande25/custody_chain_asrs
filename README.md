# Custody-Chain RL 

Auditable offline policy learning from operational records. This repo converts NASA Aviation Safety Reporting System (ASRS) narratives into sealed custody episodes.

The language model may propose **actions**. It may **not** invent reward, outcome, or “an intervention occurred.” Those come from NASA’s coded fields. A first pass is supposed to admit **nothing** until a human reviews the extractions.

## Install (macOS)

There is no `pip` command on a stock Mac. Use `python3 -m pip`:

```bash
python3 -m pip install --user -r requirements.txt
```

`anthropic` is **optional**. The default `--model stub` runs with no API key and no extra packages.

## Run the sample (no API key)

```bash
python3 run_asrs.py --csv asrs_sample.csv --model stub --out data/asrs_ledger.jsonl
```

Expected first result: **nothing admitted**. Unreviewed extractions are inadmissible on purpose.

After a human has actually read the extractions **and resolved** flagged uncertainty / unclassified actions:

```bash
python3 run_asrs.py --csv asrs_sample.csv --model stub --reviewer YOUR_NAME --out data/asrs_ledger.jsonl
```

`--reviewer` is necessary, not sufficient. It records an amendment; it does not clear `uncertain` items or rubber-stamp the ledger. Do not pass it just to make the pipeline succeed.

## Run with Claude (optional)

```bash
python3 -m pip install --user anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python3 run_asrs.py --csv asrs_sample.csv --model claude-sonnet-4-6
```

Local checkpoints require `--weights-sha256` (a model name is not an identity).

## Your own ASRS export

Public, no account: https://asrs.arc.nasa.gov/search/database.html

```bash
python3 run_asrs.py --csv path/to/asrs_export.csv --limit 200 --model stub
```

## Tests

```bash
python3 -m pip install --user pytest
python3 -m pytest test_asrs_adapter.py -q
```

## Layout

```
custody_chain/
  schema.py          episode + hash seal
  extract.py         stub or Claude reads prose
  ledger.py          append-only chain
  learn.py / certify.py / audit.py
  adapters/asrs.py   NASA CSV adapter
run_asrs.py          extract → ledger → fit / certify / audit
asrs_sample.csv      tiny two-row-header fixture
```

