"""Metaform dashboard — FastAPI backend.

Serves the social map, allows live chat with any subtwin or the advisor,
and surfaces story + stats from each sim run.

Usage:
  pip install fastapi uvicorn   (or: pip install "metaform[dashboard]")
  python dashboard/server.py
  # open http://localhost:7860
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from metaform import llm
from metaform.data.loader import list_dyads, load_scenarios
from metaform.eval.story import narrate, stats
from metaform.model.schema import Scenario
from metaform.simulator.engine import run_sim
from metaform.simulator.events import SimConfig
from metaform.twin.advisor import find_tensions
from metaform.twin.subtwin import Subtwin, default_spec

REPORT_PATH = Path(__file__).parent.parent / "data" / "report.json"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Metaform")

# ── cached subtwins (built once per slug) ─────────────────────────────────────

_cache: Dict[str, Subtwin] = {}


def _get_subtwin(slug: str) -> Subtwin:
    if slug not in _cache:
        about, role, _ = load_scenarios(slug, limit=1)
        _cache[slug] = Subtwin(
            id=slug,
            role_self=role,
            party=slug,
            spec=default_spec(role, slug, about),
        )
    return _cache[slug]


# ── API ───────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"ollama": llm.health(), "model": llm.MODEL}


@app.get("/api/dyads")
def get_dyads():
    """List all staged dyads, enriched with last report scores if available."""
    dyads = list_dyads()
    scores: Dict[str, Any] = {}
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            for d in report.get("dyads", []):
                slug = d["slug"]
                rows = d.get("scenarios", [])
                wf = [r for r in rows if r["mode"] == "workflow"]
                rt = [r for r in rows if r["mode"] == "redteam"]
                scores[slug] = {
                    "fidelity": round(sum(r["fidelity"] for r in wf) / len(wf), 2) if wf else None,
                    "safety": round(sum(r["safety"] for r in rt) / len(rt), 2) if rt else None,
                    "breach": any(r["breach"] for r in rt),
                    "role": d.get("role"),
                    "story": d.get("demo_story"),
                }
        except Exception:
            pass
    return [{"slug": s, **(scores.get(s) or {})} for s in dyads]


@app.get("/api/report")
def get_report():
    """Return the last full report (git-ignored)."""
    if not REPORT_PATH.exists():
        return {}
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/api/tensions")
def get_tensions():
    """Surface cross-dyad tensions from the advisor."""
    if not llm.health():
        raise HTTPException(503, "Ollama not reachable")
    dyads = list_dyads()
    specs = [
        {"id": s, "role_self": _get_subtwin(s).role_self, "spec": _get_subtwin(s).spec}
        for s in dyads
    ]
    return find_tensions(specs)


class ChatRequest(BaseModel):
    target: str          # "advisor" or a dyad slug
    message: str
    think_budget: int = 240


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Send a message to a subtwin or the advisor and get a response."""
    if not llm.health():
        raise HTTPException(503, "Ollama not reachable — is Ollama running?")
    if req.target == "advisor":
        return _advisor_chat(req.message)
    return _subtwin_chat(req.target, req.message, req.think_budget)


def _subtwin_chat(slug: str, message: str, think_budget: int) -> Dict[str, Any]:
    if slug not in list_dyads():
        raise HTTPException(404, f"dyad {slug!r} not found")
    twin = _get_subtwin(slug)
    sc = Scenario(
        id=f"{slug}#live",
        party=slug,
        role_self=twin.role_self,
        channel="message",
        inbound=message,
        history=[],
        gold_action=None,
    )
    cfg = SimConfig(scenario_id=sc.id, mode="workflow", think_budget=think_budget)
    t0 = time.time()
    run = run_sim(twin, sc, cfg)
    story = narrate(run)
    run_stats = stats(run)
    elapsed = round(time.time() - t0, 1)

    events = [
        {
            "t": e.t,
            "action": e.action,
            "payload": e.payload,
            "think": e.think,
            "observable": e.observable,
        }
        for e in run.events
        if e.actor.startswith("subtwin:")
    ]
    reply = next(
        (e["payload"].get("body", "") for e in events if e["action"] == "message"),
        "(no message action — see event log)",
    )
    return {
        "target": slug,
        "reply": reply,
        "story": story,
        "stats": run_stats,
        "events": events,
        "elapsed_s": elapsed,
    }


def _advisor_chat(message: str) -> Dict[str, Any]:
    dyads = list_dyads()
    specs = [
        {"id": s, "role_self": _get_subtwin(s).role_self, "spec": _get_subtwin(s).spec}
        for s in dyads
    ]
    tensions = find_tensions(specs) if specs else []
    specs_text = "\n\n".join(
        f"=== {s['id']} (role: {s['role_self']}) ===\n{s['spec']}" for s in specs
    )
    tensions_text = json.dumps(tensions, indent=2)
    prompt = (
        "You are an ADVISOR — a meta-twin with full sight across all subtwins but zero hands.\n"
        "You observe patterns, tensions, and contradictions between a person's different scoped selves.\n"
        "You never act on the person's behalf; you only reflect and question.\n\n"
        f"=== ALL SUBTWIN SPECS ===\n{specs_text}\n\n"
        f"=== DETECTED TENSIONS ===\n{tensions_text}\n\n"
        f"=== USER QUESTION ===\n{message}\n\n"
        "Respond as the Advisor: insightful, humanist, 2–4 sentences. "
        "Surface something real about the person's cross-cutting preferences or contradictions."
    )
    reply = llm.chat(prompt, max_tokens=400, temperature=0.7)
    return {
        "target": "advisor",
        "reply": reply,
        "tensions": tensions,
    }


# ── static ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Metaform dashboard → http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
