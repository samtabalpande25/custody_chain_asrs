"""What changes when a real model does the reading.

No API key here, so this file scripts the JSON a competent model returns for the
six sample narratives, in exactly the schema PROMPT_TEMPLATE asks for. It is a
stand-in for `--model claude-sonnet-4-6`, not a replacement: swap `SCRIPTED` for
`anthropic_completion("claude-sonnet-4-6")` and the rest of the pipeline is
byte-identical.

The point is the *shape* of the difference, which is bigger than "better labels":
the stub's observation is {"log_line": N}, so every state is unique and every
fitted value rests on one observation. A model proposes domain features, which
recur across episodes, which is the only way the estimator gets support.
"""

from __future__ import annotations

import json

# keyed by a distinctive substring of the narrative
SCRIPTED: dict[str, dict] = {
    "cleared for the visual approach": {
        "steps": [
            {"t": 860.0,
             "observation": {"phase": "initial_approach", "clearance": "visual",
                             "traffic_alert": "none"},
             "action": "accept_clearance", "guardrail": None,
             "evidence_mentioned": ["clearance RWY 28R at 14:20"]},
            {"t": 863.0,
             "observation": {"phase": "initial_approach", "clearance": "visual",
                             "traffic_alert": "tcas_ra"},
             "action": "arrest_descent", "guardrail": None,
             "evidence_mentioned": ["TCAS RA at 14:23"]},
            {"t": 865.0,
             "observation": {"phase": "initial_approach", "clearance": "go_around",
                             "traffic_alert": "tcas_ra"},
             "action": "execute_go_around", "guardrail": "halt",
             "evidence_mentioned": ["tower instruction at 14:25"]},
        ],
        "outcome": "aborted",
        "uncertain": [],
    },
    "ground gave us taxi": {
        "steps": [
            {"t": 545.0,
             "observation": {"phase": "taxi", "clearance": "taxi_alpha",
                             "hold_bars": "unlit"},
             "action": "taxi_per_clearance", "guardrail": None,
             "evidence_mentioned": ["ground clearance at 09:05"]},
            {"t": 547.0,
             "observation": {"phase": "taxi", "clearance": "taxi_alpha",
                             "hold_bars": "lit"},
             "action": "hold_short", "guardrail": None,
             "evidence_mentioned": ["hold bars lit at 09:07"]},
            {"t": 549.0,
             "observation": {"phase": "taxi", "clearance": "new_clearance",
                             "hold_bars": "lit"},
             "action": "cross_after_approval", "guardrail": "gate",
             "evidence_mentioned": ["supervisor approval relayed at 09:09"]},
        ],
        "outcome": "gated",
        "uncertain": ["whether the runway was actually entered or only approached"],
    },
    "autopilot captured the wrong altitude": {
        "steps": [
            {"t": 1870.0,
             "observation": {"phase": "cruise", "altitude_capture": "incorrect",
                             "atc_notified": "no"},
             "action": "detect_deviation", "guardrail": None,
             "evidence_mentioned": ["autopilot capture at 31:10"]},
            {"t": 1872.0,
             "observation": {"phase": "cruise", "altitude_capture": "corrected",
                             "atc_notified": "yes"},
             "action": "correct_profile", "guardrail": None,
             "evidence_mentioned": ["advised center at 31:12"]},
        ],
        "outcome": "clean",
        "uncertain": [],
    },
    "amber caution illuminated": {
        "steps": [
            {"t": 0.0,
             "observation": {"phase": "rollout", "caution": "amber",
                             "aircraft_moving": "yes"},
             "action": "assess_caution", "guardrail": None,
             "evidence_mentioned": ["amber caution during rollout"]},
            {"t": 1.0,
             "observation": {"phase": "rollout", "caution": "amber",
                             "aircraft_moving": "no"},
             "action": "stop_and_request_tow", "guardrail": "halt",
             "evidence_mentioned": ["halted turnoff, requested tow"]},
        ],
        "outcome": "aborted",
        "uncertain": [],
    },
    "cabin was secured": {
        "steps": [
            {"t": 0.0,
             "observation": {"phase": "climb", "cabin": "secured",
                             "service": "commenced"},
             "action": "commence_service", "guardrail": None,
             "evidence_mentioned": ["cabin secured, service commenced"]},
        ],
        "outcome": "clean",
        "uncertain": [],
    },
    "received a traffic alert": {
        "steps": [
            {"t": 1360.0,
             "observation": {"phase": "descent", "traffic_alert": "ta",
                             "manoeuvre": "none"},
             "action": "assess_traffic", "guardrail": None,
             "evidence_mentioned": ["traffic alert at 22:40"]},
            {"t": 1361.0,
             "observation": {"phase": "descent", "traffic_alert": "ta",
                             "manoeuvre": "evasive_turn"},
             "action": "evasive_turn", "guardrail": "halt",
             "evidence_mentioned": ["stopped descent, turned right at 22:41"]},
            {"t": 1364.0,
             "observation": {"phase": "descent", "traffic_alert": "clear",
                             "manoeuvre": "none"},
             "action": "resume_course", "guardrail": None,
             "evidence_mentioned": ["proceeded on course at 22:44"]},
        ],
        "outcome": "clean",
        "uncertain": [],
    },
}


def scripted_completion(prompt: str) -> str:
    """A CompleteFn. Same contract as rule_based_completion."""
    log = prompt.split("LOG:\n", 1)[-1]
    for needle, payload in SCRIPTED.items():
        if needle in log:
            return json.dumps(payload)
    return json.dumps({"steps": [], "outcome": "gated",
                       "uncertain": ["no scripted reading for this narrative"]})
