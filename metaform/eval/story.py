"""Make a sim run legible two ways: hard stats + a short human story.

The story is the humanist read — you should be able to feel whether the twin
acted like *you*. The stats are the comparable signal autoresearch optimises.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from ..llm import chat
from ..simulator.events import SimRun


def stats(run: SimRun) -> Dict[str, Any]:
    twin = [e for e in run.events if e.actor.startswith("subtwin:")]
    counts = Counter(e.action for e in twin)
    think_chars = sum(len(e.think or "") for e in twin)
    return {
        "steps": len(twin),
        "observable": sum(1 for e in twin if e.observable),
        "action_counts": dict(counts),
        "think_chars": think_chars,
        "think_budget": run.config.think_budget,
        "mode": run.config.mode,
    }


def narrate(run: SimRun) -> str:
    """Short story of how the twin behaved on this scenario."""
    lines = []
    for e in run.events:
        who = "they" if e.actor.startswith("party") else "the twin"
        body = (e.payload or {}).get("body", "")
        seg = f"t{e.t} {who} [{e.action}]"
        if e.think:
            seg += f" (thinks: {e.think})"
        if body:
            seg += f": {body}"
        lines.append(seg)
    log = "\n".join(lines)
    prompt = f"""Write a 2-3 sentence story, in plain past tense, of how this scoped agent
('the twin') handled the moment. Convey what it noticed, what it chose, and whether it
stayed in character. Be concrete, not generic.

EVENT LOG:
{log}
"""
    try:
        return chat(prompt, max_tokens=180, temperature=0.7).strip()
    except Exception:
        return log  # fall back to the raw trajectory
