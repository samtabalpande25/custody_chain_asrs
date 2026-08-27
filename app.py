"""Custody-Chain RL — live demo surface.

Slide 7 is a two-minute demo: tamper with one record, watch verify name it, watch
fit refuse. This app is that demo.

It renders nothing of its own. Every seal, every break, every refusal on screen is
produced by calling `custody_chain` — the same package the CLI runs. A dashboard
that reimplements the logic it displays is a mockup, and a mockup is worthless
when the thing being demonstrated is trustworthiness.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from custody_chain import audit as audit_mod
from custody_chain import certify as certify_mod
from custody_chain import learn as learn_mod
from custody_chain.adapters import asrs
from custody_chain.extract import (
    extraction_problems,
    mark_reviewed,
    rule_based_completion,
)
from custody_chain.ledger import Ledger
from custody_chain.schema import validate

ROOT = Path(__file__).resolve().parent

# Deck palette, sampled from the slides so the demo does not visually break away
# from the talk it sits inside.
BG, SURFACE, CODE_BG = "#0C2B23", "#0A241D", "#060F0C"
INK, MUTED = "#F4FFFF", "#8FAFA5"
TEAL, RUST, AMBER = "#37B98E", "#C4593C", "#D2A03C"

st.set_page_config(page_title="Custody-Chain RL", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""<style>
  .stApp {{ background:{BG}; color:{INK}; }}
  header[data-testid="stHeader"] {{ background:transparent; }}
  .block-container {{ padding-top:2.2rem; }}
  section[data-testid="stSidebar"] > div {{ background:{CODE_BG}; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p
    {{ color:{MUTED} !important; }}
  h1,h2,h3,h4 {{ color:{INK}; letter-spacing:-.01em; }}
  .eyebrow {{ color:{TEAL}; font:600 11px/1 ui-monospace,Menlo,monospace;
              letter-spacing:.18em; text-transform:uppercase; margin-bottom:.4rem; }}
  .lede {{ color:{MUTED}; font-size:.95rem; max-width:64ch; }}
  .card {{ background:{SURFACE}; border:1px solid #1C3B33; border-radius:10px;
           padding:.7rem .8rem; }}
  .card.bad {{ border-color:{RUST}; background:#2A1109; }}
  .card .eid {{ font:700 13px/1.4 ui-monospace,Menlo,monospace; color:{INK}; }}
  .card.bad .eid {{ color:{RUST}; }}
  .card .st {{ font:12px/1.5 ui-monospace,Menlo,monospace; color:{TEAL}; }}
  .card.bad .st {{ color:{RUST}; }}
  .card .sl {{ font:11px/1.5 ui-monospace,Menlo,monospace; color:{MUTED}; }}
  pre b {{ color:{TEAL}; }} pre i {{ color:{RUST}; font-style:normal; }}
  .kpi {{ font:700 26px/1.1 ui-monospace,Menlo,monospace; color:{INK}; }}
  .kpi-l {{ font:11px/1.4 ui-monospace,Menlo,monospace; color:{MUTED};
            letter-spacing:.1em; text-transform:uppercase; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; border-bottom:1px solid #1C3B33; }}
  .stTabs [data-baseweb="tab"] {{ color:{MUTED}; font-size:.9rem; }}
  .stTabs [aria-selected="true"] {{ color:{TEAL}; }}
</style>""", unsafe_allow_html=True)


def term(text: str) -> None:
    # st.html, not st.markdown: markdown strips <pre> and collapses the newlines
    # that make a terminal transcript readable. Styles are inline for the same
    # reason — class attributes do not reliably survive.
    st.html(
        f"<pre style='background:{CODE_BG};border-radius:10px;padding:1rem 1.1rem;"
        f"font:13px/1.65 ui-monospace,Menlo,monospace;color:#CFE7DE;"
        f"white-space:pre-wrap;margin:0'>{text}</pre>")


def kpis(pairs: list[tuple[str, str]]) -> None:
    for col, (label, val) in zip(st.columns(len(pairs)), pairs):
        col.markdown(f"<div class='kpi'>{val}</div><div class='kpi-l'>{label}</div>",
                     unsafe_allow_html=True)


# ------------------------------------------------------------------ readers
CACHE = ROOT / "data" / "extraction_cache.json"


def cached(fn, model: str):
    """Wrap a CompleteFn with an on-disk cache keyed by prompt digest.

    The prompt already contains the source text, and its SHA-256 is sealed into
    every episode — so a cache hit is provably the same reading, not merely a
    similar one. This is what makes it safe to pay for the model once and demo
    from the result: the seal does not care whether the bytes came from the wire
    or from disk, and it breaks either way if they differ.
    """
    import hashlib

    store = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def _complete(prompt: str) -> str:
        key = f"{model}:{hashlib.sha256(prompt.encode()).hexdigest()}"
        if key in store:
            return store[key]
        out = fn(prompt)
        store[key] = out
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(store, indent=2))
        return out

    return _complete


def get_reader(choice: str):
    """Return (complete_fn, model_name). The only place a model enters."""
    if choice == "Claude (API)":
        from custody_chain.extract import anthropic_completion
        return cached(anthropic_completion("claude-sonnet-4-6"),
                      "claude-sonnet-4-6"), "claude-sonnet-4-6"
    if choice == "Scripted model reading":
        from llm_preview import scripted_completion
        return scripted_completion, "claude-sonnet-4-6"
    return rule_based_completion, "rule-based-stub"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("<div class='eyebrow'>Source</div>", unsafe_allow_html=True)
    upload = st.file_uploader("ASRS CSV export", type="csv",
                              label_visibility="collapsed")
    csv_path = str(ROOT / "asrs_sample.csv")
    if upload:
        csv_path = str(ROOT / "_uploaded.csv")
        Path(csv_path).write_bytes(upload.getvalue())
    st.caption(Path(csv_path).name)

    st.markdown("<div class='eyebrow'>Who reads the prose</div>",
                unsafe_allow_html=True)
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    options = ["Keyword stub", "Scripted model reading"] + (
        ["Claude (API)"] if has_key else [])
    reader = st.radio("reader", options, label_visibility="collapsed")
    if not has_key:
        st.caption("Set ANTHROPIC_API_KEY in the shell to enable the live "
                   "Claude reader.")

    skill = st.selectbox("Skill class", list(asrs.ASRS_SKILL_CLASSES), index=2)
    reviewer = st.text_input("Reviewer", value="",
                             placeholder="leave empty for first pass")

    st.markdown("<div class='eyebrow'>Or skip extraction</div>",
                unsafe_allow_html=True)
    led_file = st.file_uploader("Sealed ledger (.jsonl)", type="jsonl",
                                label_visibility="collapsed")
    st.caption("Load a ledger you already sealed. Nothing is re-read, and no "
               "model is called.")

    st.markdown("<div class='eyebrow'>Demo</div>", unsafe_allow_html=True)
    st.caption("Model readings are cached to disk. Tampering acts on the "
               "sealed ledger, in memory only.")
    if st.button("Rebuild ledger", width='stretch'):
        st.session_state.pop("led", None)
        st.session_state.pop("_key", None)


# ------------------------------------------------------------------ pipeline
@st.cache_data(show_spinner="Reading narratives…")
def extract(csv_path: str, reader: str, reviewer: str):
    fn, name = get_reader(reader)
    eps, stats = asrs.convert(csv_path, complete_fn=fn, model_name=name)
    rows = [{
        "id": e.episode_id, "skill": e.skill_class, "outcome": e.outcome,
        "steps": len(e.steps),
        "guardrail": next((s.guardrail for s in e.steps if s.guardrail), "—"),
        "actions": ", ".join(s.action for s in e.steps),
        "uncertain": len(e.extraction.get("uncertain", [])),
        "schema_problems": validate(e, skill_classes=asrs.ASRS_SKILL_CLASSES),
        "admissibility": extraction_problems(e),
    } for e in eps]
    return eps, stats, rows


def build_ledger(eps, reviewer):
    led = Ledger(skill_classes=asrs.ASRS_SKILL_CLASSES)
    refused = []
    for ep in asrs.iter_admissible(eps, reviewer=reviewer or None):
        try:
            led.append(ep)
        except Exception as exc:
            refused.append(f"{ep.episode_id}: {exc}")
    return led, refused


if led_file is not None:
    # A sealed ledger needs no reader and no reviewer — it is already a record.
    lp = ROOT / "_uploaded_ledger.jsonl"
    lp.write_bytes(led_file.getvalue())
    key = ("ledger", led_file.name, len(led_file.getvalue()))
    if st.session_state.get("_key") != key:
        st.session_state.led = Ledger.load(lp,
                                           skill_classes=asrs.ASRS_SKILL_CLASSES)
        st.session_state._key = key
    episodes, stats, rows = [], None, []
else:
    try:
        episodes, stats, rows = extract(csv_path, reader, reviewer)
    except Exception as exc:
        st.error(f"The reader failed, so there is nothing to seal.\n\n`{exc}`")
        st.caption("A 400 about credit balance is a billing state, not a bug in "
                   "the adapter — switch the reader to Keyword stub or Scripted "
                   "model reading to keep going.")
        st.stop()

    if st.session_state.get("_key") != (csv_path, reader, reviewer):
        import copy
        led, refused = build_ledger(copy.deepcopy(episodes), reviewer)
        st.session_state.led, st.session_state.refused = led, refused
        st.session_state._key = (csv_path, reader, reviewer)

led: Ledger = st.session_state.led


# ------------------------------------------------------------------ header
st.markdown("<div class='eyebrow'>Custody-Chain RL · live demo</div>",
            unsafe_allow_html=True)
st.markdown("## Edit one record. Watch the system refuse.")
st.markdown("<p class='lede'>Every figure below is returned by the "
            "<code>custody_chain</code> package — the same code the CLI runs. "
            "This page computes nothing itself.</p>", unsafe_allow_html=True)

breaks = led.verify_chain()
bad_ix = {b.index for b in breaks}

if len(led):
    st.write("")
    for col, (i, ep) in zip(st.columns(min(len(led), 8)), enumerate(led.episodes)):
        cls = "card bad" if i in bad_ix else "card"
        mark = "seal ✗ mismatch" if i in bad_ix else "seal ✓"
        col.markdown(
            f"<div class='{cls}'><div class='eid'>{ep.episode_id}</div>"
            f"<div class='st'>{mark}</div>"
            f"<div class='sl'>{(ep.seal or '')[:4]}…{(ep.seal or '')[-4:]}</div></div>",
            unsafe_allow_html=True)
else:
    why = ("Review is necessary, not sufficient — a reviewer is set, but "
           "unresolved uncertainty, unclassified actions, missing evidence, or "
           "an undeclared skill class still refuse every candidate."
           if reviewer else
           "Nothing was admitted. That is the expected first pass — set a "
           "reviewer in the sidebar once a human has read the extractions.")
    st.markdown(f"<div class='card'><div class='eid' style='color:{AMBER}'>"
                f"Ledger empty</div><div class='sl'>{why}</div></div>",
                unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["Extract", "Tamper", "Verify", "Fit", "Certify",
                "Bundle", "Why"])

# ------------------------------------------------------------------ extract
with tabs[0]:
    if stats is None:
        st.markdown("<div class='eyebrow'>Loaded from disk</div>",
                    unsafe_allow_html=True)
        st.markdown("<p class='lede'>These episodes were sealed elsewhere. No "
                    "narrative was re-read and no model was called — the seals "
                    "on screen are the ones the file arrived with.</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='eyebrow'>What the reader proposed</div>",
                    unsafe_allow_html=True)
        kpis([("rows", str(stats.rows)), ("converted", str(stats.converted)),
              ("unclassified", str(stats.unclassified_actions)),
              ("flagged uncertain", str(stats.flagged_uncertain)),
              ("inadmissible", str(stats.inadmissible))])
        st.write("")
        st.dataframe(
            [{k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}
             for r in rows], width='stretch', hide_index=True)

        st.markdown("<div class='eyebrow'>Boundary</div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='lede'>The reader proposed every action string above. It "
            "proposed no reward, no outcome, and no guardrail — outcome and "
            "guardrail are lifted from NASA's analyst-coded fields, and reward is "
            "0.0 on every step by construction.</p>", unsafe_allow_html=True)

        pick = st.selectbox("Inspect an extraction", [r["id"] for r in rows])
        ep = next(e for e in episodes if e.episode_id == pick)
        a, b = st.columns(2)
        a.markdown("**Model-proposed**")
        a.json({"steps": [{"t": s.t, "observation": s.observation,
                           "action": s.action} for s in ep.steps]}, expanded=True)
        b.markdown("**Coded + sealed provenance**")
        b.json({"outcome": ep.outcome,
                "coded_fields": ep.extraction["coded_fields"],
                "collection_regime": ep.extraction["collection_regime"],
                "model": ep.extraction["model"],
                "prompt_sha256": ep.extraction["prompt_sha256"][:24] + "…",
                "source_sha256": ep.extraction["source_sha256"][:24] + "…",
                "review_status": ep.extraction["review_status"],
                "uncertain": ep.extraction["uncertain"]})

# ------------------------------------------------------------------ tamper
with tabs[1]:
    st.markdown("<div class='eyebrow'>The demo moment</div>",
                unsafe_allow_html=True)
    if not len(led):
        st.markdown("<p class='lede'>Admit something first — the ledger is "
                    "empty, and there is nothing sealed to tamper with.</p>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<p class='lede'>Change a sealed record after the fact. "
                    "Nothing else on this page is touched.</p>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("Episode",
                              [e.episode_id for e in led.episodes])
        what = c2.selectbox("Edit", ["reward on step 0", "swap the weights digest",
                                     "remove this episode"])
        c3.write("")
        if c3.button("Apply", type="primary", width='stretch'):
            i = next(i for i, e in enumerate(led.episodes)
                     if e.episode_id == target)
            if what == "reward on step 0":
                led.episodes[i].steps[0].reward = 9.99
            elif what == "swap the weights digest":
                led.episodes[i].extraction = dict(led.episodes[i].extraction)
                led.episodes[i].extraction["weights_sha256"] = "d00d" + "0" * 60
            else:
                del led.episodes[i]
            st.rerun()

        st.markdown("<p class='lede'>A bad seal means <i>this</i> record was "
                    "edited. A bad link means a record was inserted or removed "
                    "around it. They are reported apart because they mean "
                    "different things to an investigator.</p>",
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ verify
with tabs[2]:
    term(f"$ custody verify data/asrs_ledger.jsonl\n"
         f"  episodes  {len(led)}\n"
         f"  head      {led.head[:32]}…\n" +
         ("  chain     <b>INTACT</b>" if not breaks else
          f"  chain     <i>BROKEN — {len(breaks)} problem(s)</i>\n" +
          "\n".join(f"    <i>! {b}</i>" for b in breaks)))

# ------------------------------------------------------------------ fit
with tabs[3]:
    try:
        pol = learn_mod.fit(led, skill_class=skill)
        sup = pol.support_summary()
        term(f"$ custody fit data/asrs_ledger.jsonl --skill {skill}\n"
             f"  episodes used    {pol.n_episodes}\n"
             f"  values fitted    {len(pol.estimates)}\n"
             f"  fingerprint      {pol.fingerprint[:32]}…\n"
             f"  median obs/value {sup['median_n']}")
        st.session_state.pol = pol
        if sup["values"] and sup["thin_fraction"] > 0.5:
            st.warning(
                f"{sup['thin']} of {sup['values']} fitted values rest on fewer "
                f"than {sup['thin_below']} observations. The state abstraction "
                "is too fine for this much data. Most of these numbers are noise.")
        if pol.estimates:
            st.dataframe([{"state": e.state, "action": e.action,
                           "mean_return": e.mean_return, "n": e.n,
                           "episodes": len(e.episode_ids)}
                          for e in pol.estimates.values()],
                         width='stretch', hide_index=True)
        st.caption("Every reward here is 0.0. The adapter does not invent reward, "
                   "so this table is structurally meaningful and numerically empty "
                   "until a deployment supplies a reward specification.")
    except ValueError as exc:
        st.session_state.pop("pol", None)
        term(f"$ custody fit data/asrs_ledger.jsonl --skill {skill}\n"
             + "\n".join(f"  <i>{l}</i>" for l in str(exc).splitlines())
             + "\n  <i>(exit 1)</i>")
        st.markdown("<p class='lede'>The refusal is the feature. Training on "
                    "unverifiable records produces a policy whose provenance "
                    "claim is worthless — the one outcome this exists to "
                    "prevent.</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------ certify
with tabs[4]:
    tier = st.selectbox("Current tier", list(certify_mod.TIERS[:-1]))
    rep = certify_mod.evaluate(led, skill_class=skill, current_tier=tier)
    st.session_state.rep = rep
    term(f"$ custody certify data/asrs_ledger.jsonl --skill {skill} "
         f"--tier {tier}\n\n"
         + rep.summary().replace("PROMOTE", "<b>PROMOTE</b>")
                        .replace("HOLD", "<i>HOLD</i>"))
    st.caption("Wilson score lower bound, 95%. Four successes out of four is a "
               "raw 100% and a lower bound near 51% — the honest reading.")

# ------------------------------------------------------------------ bundle
with tabs[5]:
    if "pol" not in st.session_state:
        st.markdown("<p class='lede'>No fitted policy — fit refused, or nothing "
                    "was admitted.</p>", unsafe_allow_html=True)
    else:
        bundle = audit_mod.build(led, st.session_state.pol,
                                 st.session_state.rep)
        term(audit_mod.render_text(bundle))
        st.download_button("Download evidence bundle",
                           json.dumps(bundle, indent=2, default=str),
                           file_name="asrs_bundle.json", mime="application/json")

# ------------------------------------------------------------------ why
with tabs[6]:
    pol = st.session_state.get("pol")
    if not pol or not pol.estimates:
        st.markdown("<p class='lede'>Fit a policy first.</p>",
                    unsafe_allow_html=True)
    else:
        key = st.selectbox("Learned value", sorted(pol.estimates))
        est = pol.estimates[key]
        lines = [f"$ custody why --state {est.state!r} --action {est.action!r}",
                 "", f"  mean return      {est.mean_return:+.6f}",
                 f"  observations     {est.n}",
                 f"  caused by        {len(set(est.episode_ids))} episode(s):"]
        lines += [f"    {eid}  seal {seal[:16]}…"
                  for eid, seal in zip(est.episode_ids, est.episode_seals)]
        term("\n".join(lines))
        st.caption("Because the estimator is linear in its episodes, attribution "
                   "is exact rather than approximate — no influence-function "
                   "estimation required.")
