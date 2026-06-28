"""The simulator runs as an event-sourced action log — not a conversation.

A run is an ordered series of typed events. Speaking is just one action whose
payload carries the message body. The model gets a *bounded* space to think,
controlled by `think_budget` (a knob varied between sims).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Action vocabulary. Speaking == `message`; everything else is non-verbal action.
ACTIONS = (
    "read",
    "message",
    "tool_call",
    "schedule",
    "escalate",
    "decline",
    "wait",
    "note",
)


@dataclass
class SimEvent:
    t: int  # logical step index
    actor: str  # "subtwin:<id>" | "party:<label>" | "world"
    action: str  # one of ACTIONS
    payload: Dict[str, Any] = field(default_factory=dict)  # e.g. {"to","channel","body"}
    think: Optional[str] = None  # bounded private reasoning (capped by think_budget)
    observable: bool = False  # did it cross the boundary (sent / acted) vs internal?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimConfig:
    """Config for one simulation.

    `think_budget` is a two-part constraint, NOT a model-compute limit: the prompt
    *asks* the subtwin to keep each `think` under the budget, and any returned
    reasoning is then *truncated* to it (0 = thinking forbidden entirely). It bounds
    the reasoning the subtwin is asked to use and is allowed to retain/expose — the
    underlying model may still compute more internally. (Roadmap: map the budget to
    the model's `num_predict` for a genuine compute cap.)
    """

    scenario_id: str
    mode: str = "workflow"  # "workflow" (fidelity backtest) | "redteam" (boundary)
    think_budget: int = 240  # chars of private reasoning asked-for + retained; 0 = none
    max_steps: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimRun:
    subtwin_id: str
    config: SimConfig
    events: List[SimEvent] = field(default_factory=list)
    gold_action: Optional[Dict[str, Any]] = None  # what the human actually did

    def observable_events(self) -> List[SimEvent]:
        return [e for e in self.events if e.observable]

    def primary_action(self) -> Optional[SimEvent]:
        """The twin's first boundary-crossing action — what it actually 'did'."""
        for e in self.events:
            if e.observable and e.actor.startswith("subtwin:"):
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtwin_id": self.subtwin_id,
            "config": self.config.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "gold_action": self.gold_action,
        }
