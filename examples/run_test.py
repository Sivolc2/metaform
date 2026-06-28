"""End-to-end test: build two subtwins from the staged dyad slices, simulate,
red-team, run the autoresearch loop, and surface cross-dyad tensions.

Usage:
  python examples/run_test.py --smoke      # one scenario, fast sanity check
  python examples/run_test.py              # full run over both dyads
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metaform import llm  # noqa: E402
from metaform.autoresearch.loop import iterate  # noqa: E402
from metaform.data.loader import _INJECTIONS, list_dyads, load_scenarios, make_redteam  # noqa: E402
from metaform.eval.judge import score_run  # noqa: E402
from metaform.eval.story import narrate, stats  # noqa: E402
from metaform.simulator.engine import run_sim  # noqa: E402
from metaform.simulator.events import SimConfig  # noqa: E402
from metaform.twin.advisor import find_tensions  # noqa: E402
from metaform.twin.subtwin import Subtwin, default_spec  # noqa: E402

REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "report.json")


def build(slug):
    about, role, scen = load_scenarios(slug, limit=8)
    twin = Subtwin(id=slug, role_self=role, party=slug, spec=default_spec(role, slug, about))
    return about, role, scen, twin


def smoke():
    dyads = list_dyads()
    if not dyads:
        print("no staged dyads under data/slices/")
        return
    slug = dyads[0]
    print("warming up", llm.MODEL, "...")
    llm.warmup()
    about, role, scen, twin = build(slug)
    sc = scen[0]
    cfg = SimConfig(scenario_id=sc.id, mode="workflow", think_budget=240)
    t0 = time.time()
    run = run_sim(twin, sc, cfg)
    score = score_run(run, sc.inbound, "workflow")
    story = narrate(run)
    print(f"\n=== SMOKE: {slug} (role={role}) ===")
    print(f"model={llm.MODEL}  elapsed={time.time()-t0:.0f}s")
    print(f"fidelity={score['fidelity']:.2f}  combined={score['combined']:.2f}")
    print(f"rationale: {score['rationale']}")
    print(f"stats: {stats(run)}")
    print(f"story: {story}")


def run_dyad(slug, n_workflow, rounds, think_budget, n_injections):
    about, role, scen, twin = build(slug)
    scenarios = scen[:n_workflow]
    if scenarios:
        # exercise injection variants, cycling over available workflow scenarios
        k = max(0, min(n_injections, len(_INJECTIONS)))
        redteams = [
            make_redteam(scenarios[i % len(scenarios)], injection_index=i) for i in range(k)
        ]
        scenarios = scenarios + redteams
    print(f"\n=== DYAD: {slug} (role={role}) — {len(scenarios)} scenarios ===")
    res = iterate(twin, scenarios, rounds=rounds, think_budget=think_budget)
    print(f"  autoresearch: baseline={res['baseline_mean']:.2f} -> best={res['best_mean']:.2f}")
    for h in res["history"]:
        print(f"    round {h['round']}: mean={h['mean']:.2f}  {h['note']}")

    # per-scenario breakdown so the red-team boundary result is legible, not buried in a mean
    print("  per-scenario (best spec):")
    scenario_rows = []
    for r in res["best_eval"]["results"]:
        s = r["score"]
        tag = "BREACH" if s["breach"] else ("held" if r["mode"] == "redteam" else "-")
        print(
            f"    {r['mode']:<8} {r['scenario']:<28} "
            f"fidelity={s['fidelity']:.2f} safety={s['safety']:.2f} {tag}"
        )
        scenario_rows.append(
            {
                "scenario": r["scenario"],
                "mode": r["mode"],
                "fidelity": s["fidelity"],
                "safety": s["safety"],
                "breach": s["breach"],
            }
        )

    # think-budget knob: same scenario, with vs without room to think
    demo = scenarios[0]
    sweep = {}
    for tb in (0, think_budget):
        cfg = SimConfig(scenario_id=demo.id, mode="workflow", think_budget=tb)
        r = run_sim(res["best"], demo, cfg)
        s = score_run(r, demo.inbound, "workflow")
        sweep[tb] = {"combined": s["combined"], "stats": stats(r), "story": narrate(r)}
    print(f"  think-budget sweep on {demo.id}:")
    for tb, v in sweep.items():
        print(f"    budget={tb:>3}: combined={v['combined']:.2f}  think_chars={v['stats']['think_chars']}")
    print(f"  story (budget={think_budget}): {sweep[think_budget]['story']}")

    return {
        "slug": slug,
        "role": role,
        "autoresearch": res["history"],
        "scenarios": scenario_rows,
        "baseline_mean": res["baseline_mean"],
        "best_mean": res["best_mean"],
        "best_spec": res["best"].spec,
        "think_sweep": {str(k): {"combined": v["combined"], "think_chars": v["stats"]["think_chars"]} for k, v in sweep.items()},
        "demo_story": sweep[think_budget]["story"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workflow", type=int, default=2, help="workflow scenarios per dyad")
    ap.add_argument("--rounds", type=int, default=1, help="autoresearch rounds")
    ap.add_argument("--think", type=int, default=240)
    ap.add_argument("--injections", type=int, default=4, help="red-team variants per dyad (max 4)")
    args = ap.parse_args()

    if not llm.health():
        print("Ollama not reachable at", llm.OLLAMA_URL)
        return
    if args.smoke:
        smoke()
        return
    print("warming up", llm.MODEL, "...")
    llm.warmup()

    t0 = time.time()
    reports = []
    specs_for_advisor = []
    for slug in list_dyads():
        rep = run_dyad(slug, args.workflow, args.rounds, args.think, args.injections)
        reports.append(rep)
        specs_for_advisor.append({"id": slug, "role_self": rep["role"], "spec": rep["best_spec"]})

    print("\n=== ADVISOR: cross-dyad tensions (full sight, zero hands) ===")
    tensions = find_tensions(specs_for_advisor)
    for tn in tensions:
        print(f"  • between {tn.get('between')}: {tn.get('tension')}")
        print(f"    ↳ {tn.get('question')}")

    out = {"model": llm.MODEL, "elapsed_s": round(time.time() - t0), "dyads": reports, "tensions": tensions}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\ndone in {out['elapsed_s']}s — full report at data/report.json (git-ignored)")


if __name__ == "__main__":
    main()
