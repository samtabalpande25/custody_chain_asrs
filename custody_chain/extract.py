"""Extract structured custody episodes from unstructured operational prose.

Why this module exists
----------------------
The premise of this project is that regulated facilities already hold the data.
That is true, and it is also misleading: they hold it as *prose*. Nursing notes,
shift handovers, maintenance write-ups, switching orders, incident narratives.
Almost none of it arrives as (state, action, reward) tuples.

Every other input path in this repo sidesteps that. The synthetic generator emits
structured records by construction; the MIMIC adapter consumes a cohort someone
else already preprocessed. This module handles the case that actually blocks
deployment: a shift log written by a human for other humans.

The provenance problem this creates
-----------------------------------
If a language model reads the note and decides what the action was, then the
model is now part of the causal chain behind the policy. "Why did you learn
this?" can no longer be answered with a list of episodes. The honest answer is:
this model, reading this text, under this prompt, produced this reading -- and
here is whether a human ever checked it.

So extraction metadata is sealed *into* the episode. It covers the model id, a
hash of the prompt, a hash of the source text, and the review status. Change any
of them and the episode's seal breaks like any other edit. An extraction step
that sits outside the chain silently voids every integrity claim downstream of
it, which is worse than having no chain at all.

Extractions default to `review_status="unreviewed"`, and unreviewed episodes are
inadmissible for fitting unless explicitly allowed. A model's guess about what a
nurse meant is a hypothesis, not a record.

Provider independence
---------------------
`extract_episode` takes a `complete_fn(prompt) -> str`. Any provider works, and
the deterministic stub in `rule_based_completion` lets the pipeline and its tests
run with no network and no API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import CustodyEpisode, Step

CompleteFn = Callable[[str], str]

REVIEW_STATES = ("unreviewed", "human_reviewed", "human_corrected", "rejected")

PROMPT_TEMPLATE = """You are converting an operational log into a structured record.

Return ONLY a JSON object, no prose and no markdown fences, with this shape:

{{
  "steps": [
    {{
      "t": <number, minutes from start>,
      "observation": {{"<field>": "<short value>"}},
      "action": "<short verb phrase, lowercase, underscores>",
      "guardrail": null | "gate" | "halt",
      "evidence_mentioned": ["<thing the log cites as evidence>"]
    }}
  ],
  "outcome": "clean" | "gated" | "aborted",
  "uncertain": ["<anything you had to guess at>"]
}}

Rules:
- Record only what the log states. Do not infer actions that are not described.
- Set "guardrail" only if the log describes a stop or an approval being required.
- Put anything ambiguous in "uncertain". An empty list means you are claiming the
  log was unambiguous, so use it sparingly.

LOG:
{log}
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Extraction:
    """Provenance for a model-produced record. Sealed into the episode."""

    model: str
    prompt_sha256: str
    source_sha256: str
    source_chars: int
    review_status: str = "unreviewed"
    uncertain: list[str] = field(default_factory=list)
    reviewer: str | None = None

    # A model *name* proves nothing about which weights actually did the
    # reading. On-prem and airgapped deployments pull checkpoints off physical
    # media, so the identity that matters is the digest, not the label. Sealed
    # with the rest of the record: swap the checkpoint and the seal breaks.
    weights_sha256: str | None = None
    decode_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "weights_sha256": self.weights_sha256,
            "decode_params": self.decode_params,
            "prompt_sha256": self.prompt_sha256,
            "source_sha256": self.source_sha256,
            "source_chars": self.source_chars,
            "review_status": self.review_status,
            "uncertain": self.uncertain,
            "reviewer": self.reviewer,
        }


def rule_based_completion(prompt: str) -> str:
    """A deterministic stand-in for a model, so tests need no network.

    It is a crude keyword matcher and it is not pretending otherwise. Its purpose
    is to exercise the extraction and sealing path, not to do the reading well.
    Real extraction quality is an open question this repo does not answer.
    """
    log = prompt.split("LOG:\n", 1)[-1].strip()
    steps, uncertain = [], []

    for i, raw in enumerate(l for l in log.splitlines() if l.strip()):
        line = raw.strip()
        low = line.lower()

        if "held" in low or "waited" in low or "paused" in low:
            action = "hold"
        elif "rerouted" in low or "alternate" in low:
            action = "reroute"
        elif "stopped" in low or "halted" in low or "abort" in low:
            action = "halt"
        elif "proceeded" in low or "continued" in low or "delivered" in low:
            action = "proceed"
        else:
            action = "unclassified"
            uncertain.append(f"line {i}: no action verb matched")

        guardrail = None
        if "stopped" in low or "halted" in low:
            guardrail = "halt"
        elif "approval" in low or "supervisor" in low or "authorised" in low:
            guardrail = "gate"

        m = re.search(r"\b(\d{1,2}):(\d{2})\b", line)
        t = float(int(m.group(1)) * 60 + int(m.group(2))) if m else float(i)

        steps.append({
            "t": t,
            "observation": {"log_line": str(i)},
            "action": action,
            "guardrail": guardrail,
            "evidence_mentioned": [f"log_line:{i}"],
        })

    outcome = "clean"
    if any(s["guardrail"] == "halt" for s in steps):
        outcome = "aborted"
    elif any(s["guardrail"] == "gate" for s in steps):
        outcome = "gated"

    return json.dumps({"steps": steps, "outcome": outcome, "uncertain": uncertain})


def anthropic_completion(model: str = "claude-sonnet-4-6") -> CompleteFn:
    """Build a completion function backed by the Anthropic API.

    Requires the `anthropic` package and an API key in the environment. Kept out
    of the default path so the repo installs and tests with no credentials.
    """
    def _complete(prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return _complete


def _parse(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    return json.loads(text)


def extract_episode(
    log_text: str,
    episode_id: str,
    site: str,
    skill_class: str,
    complete_fn: CompleteFn = rule_based_completion,
    model_name: str = "rule-based-stub",
    evidence_prefix: str = "log",
) -> tuple[CustodyEpisode, Extraction]:
    """Turn one operational log into a candidate custody episode.

    Returns the episode and its extraction provenance. The episode is *not*
    sealed here -- sealing happens on append, so the ledger controls chain order.
    """
    prompt = PROMPT_TEMPLATE.format(log=log_text)
    parsed = _parse(complete_fn(prompt))

    steps = []
    for s in parsed.get("steps", []):
        mentions = s.get("evidence_mentioned") or []
        steps.append(
            Step(
                t=float(s.get("t", 0.0)),
                observation=s.get("observation", {}),
                action=s.get("action", "unclassified"),
                reward=0.0,      # extraction does not invent rewards
                cost=0.0,
                guardrail=s.get("guardrail"),
                evidence=[f"{evidence_prefix}:{episode_id}:{m}" for m in mentions],
            )
        )

    extraction = Extraction(
        model=model_name,
        prompt_sha256=_sha(prompt),
        source_sha256=_sha(log_text),
        source_chars=len(log_text),
        uncertain=parsed.get("uncertain", []),
    )

    episode = CustodyEpisode(
        episode_id=episode_id,
        site=site,
        skill_class=skill_class,
        steps=steps,
        outcome=parsed.get("outcome", "gated"),
        extraction=extraction.to_dict(),
    )
    return episode, extraction


def mark_reviewed(
    ep: CustodyEpisode,
    reviewer: str,
    status: str = "human_reviewed",
) -> CustodyEpisode:
    """Record that a human checked the extraction, and re-seal.

    Review changes the record, so the seal must be recomputed. If the episode is
    already in a ledger, everything after it must be re-sealed too -- which is
    the correct cost. Reviewing an extraction is an amendment, and amendments are
    supposed to be visible.
    """
    if status not in REVIEW_STATES:
        raise ValueError(f"unknown review status {status!r}; expected {REVIEW_STATES}")
    if not ep.extraction:
        raise ValueError(f"{ep.episode_id}: no extraction to review")

    ep.extraction = dict(ep.extraction)
    ep.extraction["review_status"] = status
    ep.extraction["reviewer"] = reviewer
    ep.seal_record(prev_seal=ep.prev_seal)
    return ep


#: Providers that identify a model by name because the weights are not local.
HOSTED_MODEL_PREFIXES = ("claude-", "gpt-", "gemini-", "rule-based-")


def _is_hosted(model: str) -> bool:
    return any(model.startswith(p) for p in HOSTED_MODEL_PREFIXES)


def extraction_problems(ep: CustodyEpisode, allow_unreviewed: bool = False) -> list[str]:
    """Admissibility checks specific to model-extracted records."""
    problems: list[str] = []
    ex = ep.extraction
    if not ex:
        return problems

    status = ex.get("review_status")
    if status == "rejected":
        problems.append(f"{ep.episode_id}: extraction was rejected by {ex.get('reviewer')}")
    elif status == "unreviewed" and not allow_unreviewed:
        problems.append(
            f"{ep.episode_id}: extraction is unreviewed — a model's reading of a "
            "log is a hypothesis, not a record"
        )

    if not _is_hosted(ex.get("model", "")) and not ex.get("weights_sha256"):
        problems.append(
            f"{ep.episode_id}: local model {ex.get('model')!r} recorded without a "
            "weights digest — the reading cannot be attributed to a checkpoint"
        )

    if ex.get("uncertain"):
        problems.append(
            f"{ep.episode_id}: extraction flagged {len(ex['uncertain'])} uncertain "
            "item(s); a human should resolve them before this trains anything"
        )

    for i, s in enumerate(ep.steps):
        if s.action == "unclassified":
            problems.append(f"{ep.episode_id}[{i}]: action could not be classified")

    return problems
