"""The simulator: turn a (subtwin, scenario, config) into an event log."""

from __future__ import annotations

from ..model.schema import Scenario
from ..twin.subtwin import Subtwin
from .events import SimConfig, SimEvent, SimRun


def run_sim(subtwin: Subtwin, scenario: Scenario, config: SimConfig) -> SimRun:
    """Replay one scenario through a subtwin and record the event log.

    t=0 is the inbound (the world acting). The subtwin's events follow.
    """
    events = [
        SimEvent(
            t=0,
            actor=f"party:{scenario.party}",
            action="message",
            payload={"channel": scenario.channel, "body": scenario.inbound},
            observable=True,
        )
    ]
    twin_events = subtwin.act(scenario, config.think_budget, config.mode)
    # renumber after the inbound, respect max_steps
    for i, e in enumerate(twin_events[: config.max_steps], start=1):
        e.t = i
        events.append(e)
    return SimRun(
        subtwin_id=subtwin.id,
        config=config,
        events=events,
        gold_action=scenario.gold_action,
    )
