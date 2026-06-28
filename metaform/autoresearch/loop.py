"""Autoresearch loop — iterate a subtwin's spec to better mirror the person.

Mirrors Karpathy's autoresearch: the agent edits ONE artifact (the spec), runs a
FIXED battery (the scenario set), scores ONE comparable metric (mean combined),
and keeps or discards. The human edits the meta-instructions, not the spec.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from ..eval.judge import score_run
from ..llm import chat
from ..model.schema import Scenario
from ..simulator.engine import run_sim
from ..simulator.events import SimConfig
from ..twin.subtwin import Subtwin


def evaluate(subtwin: Subtwin, scenarios: List[Scenario], think_budget: int) -> Dict[str, Any]:
    """Run the battery, return mean combined score + per-scenario detail."""
    results = []
    total = 0.0
    for sc in scenarios:
        mode = "redteam" if sc.adversarial else "workflow"
        cfg = SimConfig(scenario_id=sc.id, mode=mode, think_budget=think_budget)
        run = run_sim(subtwin, sc, cfg)
        score = score_run(run, sc.inbound, mode)
        total += score["combined"]
        results.append({"scenario": sc.id, "mode": mode, "run": run, "score": score})
    mean = total / len(scenarios) if scenarios else 0.0
    return {"mean": mean, "results": results}


def propose_spec(subtwin: Subtwin, eval_out: Dict[str, Any]) -> str:
    """Ask the model to edit the spec given where the twin diverged from the person."""
    misses = []
    for r in eval_out["results"]:
        s = r["score"]
        if s["combined"] < 0.8:
            misses.append(f"- [{r['mode']}] {r['scenario']}: {s['rationale']}")
    misses_txt = "\n".join(misses) or "(no major misses; tighten voice & precision)"
    prompt = f"""You are improving a SUBTWIN SPEC so it more faithfully mirrors how a specific
person acts in one role. Here is the current spec:

--- CURRENT SPEC ---
{subtwin.spec}
--- END ---

It diverged from the person (or had boundary issues) on:
{misses_txt}

Rewrite the spec to fix these — sharpen the preferences, voice, and scope. Keep the same
section structure. Do not invent facts about the person; generalize from the misses.
Output ONLY the full revised spec text."""
    try:
        return chat(prompt, max_tokens=600, temperature=0.4).strip()
    except Exception:
        return subtwin.spec


def iterate(
    subtwin: Subtwin, scenarios: List[Scenario], rounds: int = 2, think_budget: int = 240
) -> Dict[str, Any]:
    """Keep-or-discard search over the spec. Returns the experiment log + best twin."""
    base = evaluate(subtwin, scenarios, think_budget)
    history = [{"round": 0, "mean": round(base["mean"], 3), "kept": True, "note": "baseline"}]
    best = subtwin
    best_mean = base["mean"]
    last_eval = base

    for r in range(1, rounds + 1):
        candidate = replace(best, spec=propose_spec(best, last_eval))
        cand_eval = evaluate(candidate, scenarios, think_budget)
        kept = cand_eval["mean"] > best_mean + 1e-6
        history.append(
            {
                "round": r,
                "mean": round(cand_eval["mean"], 3),
                "kept": kept,
                "note": "improved" if kept else "discarded (no gain)",
            }
        )
        if kept:
            best, best_mean, last_eval = candidate, cand_eval["mean"], cand_eval
    return {
        "history": history,
        "best_mean": best_mean,
        "baseline_mean": base["mean"],
        "best": best,
        "best_eval": last_eval,
    }
