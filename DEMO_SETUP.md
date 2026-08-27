# Live demo setup — slide 7

Drop these three files into `custody_chain_asrs/`, next to `run_asrs.py`:

```
custody_chain_asrs/
  custody_chain/            (already there)
  run_asrs.py               (already there)
  asrs_sample.csv           (already there)
  app.py                    <- new
  llm_preview.py            <- new
  .streamlit/config.toml    <- new
```

Then:

```bash
python3 -m pip install --user streamlit
streamlit run app.py
```

No API key needed. Nothing in the app touches the network.

## What the sidebar controls

**Who reads the prose** — three readers, all producing the same sealed record
format:

- `Keyword stub` — the deterministic matcher already in `extract.py`. Emits
  `observation = {"log_line": N}`, which is why every fitted value in your
  bundle has `n=1`.
- `Scripted model reading` — `llm_preview.py`. The JSON a competent model
  returns for the six sample narratives, in the exact schema
  `PROMPT_TEMPLATE` asks for. Use this on stage.
- `Claude (API)` — only appears when `ANTHROPIC_API_KEY` is set. Wrapped in a
  disk cache (see below).

**Reviewer** — leave blank and the ledger comes up empty, which is the honest
first-pass result. Fill it in and five episodes are admitted. That toggle is
worth showing before the tamper demo.

**Or skip extraction** — upload a `.jsonl` you already sealed. No narrative is
re-read and no model is called.

## The tamper demo

Tamper tab → pick an episode → Apply.

| Edit | Failure it produces |
|---|---|
| reward on step 0 | bad seal — *this* record was edited |
| swap the weights digest | bad seal — proves the checkpoint is inside the seal |
| remove this episode | bad **link** — a record was inserted or removed around it |

Then Verify (names the index), then Fit (refuses, exit 1). The chain strip at
the top turns that card rust in real time.

Tampering is in-memory only. `Rebuild ledger` resets it.

## The extraction cache

`data/extraction_cache.json`, keyed on `sha256(prompt)`. The prompt contains the
source text and its digest is already sealed into every episode as
`prompt_sha256` — so a cache hit is provably the same reading, not merely a
similar one.

Practically: pay for the model once, and the demo runs forever offline. If the
API later returns a 400, a warm cache means the demo still works. The app also
catches reader failures and prints a readable message rather than a traceback.

## Before the talk

- Rotate the API key that was pasted in plain text.
- Run once with `Claude (API)` to warm the cache, then unset the key so the
  machine on stage cannot make a network call at all.
- `python3 -m pytest test_asrs_adapter.py -q` — the adapter tests pin the
  boundaries (no invented reward, guardrails from coded fields only, review
  re-seals).
- Reconcile your local `extract.py` with the copy you shared: yours has the
  friendly 400 handler, the shared copy would throw a raw `BadRequestError`.

## Not built yet

The backwards gate from slide 10 — *how many more episodes at the observed rate
before this skill is promotable?* It is an inversion of `wilson_lower_bound` and
would sit naturally in the Certify tab. It is the idea on that slide you said
you had not seen framed elsewhere, so it is the one worth having live.
