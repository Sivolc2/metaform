"""Static world model: scenarios drawn from real, replayed history.

The honesty principle: scenarios are grounded in what actually happened,
perturbed only for red-team. Gold labels are what the human actually did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Exchange:
    t: int
    channel: str
    inbound: str  # what the party / world said
    gold_action: Optional[Dict[str, Any]] = None  # what the human actually did


@dataclass
class Scenario:
    """One test case for a subtwin: an inbound + the prior history, with the
    human's real response as the gold label."""

    id: str
    party: str  # generic label
    role_self: str
    channel: str
    inbound: str
    history: List[Exchange] = field(default_factory=list)
    gold_action: Optional[Dict[str, Any]] = None
    adversarial: Optional[str] = None  # red-team injection, in the party's voice

    def history_text(self, limit: int = 4) -> str:
        rows = []
        for e in self.history[-limit:]:
            rows.append(f"[{e.channel}] party: {e.inbound}")
            if e.gold_action:
                body = (e.gold_action.get("payload") or {}).get("body", "")
                rows.append(f"        you: ({e.gold_action.get('action')}) {body}")
        return "\n".join(rows) if rows else "(no prior history)"
