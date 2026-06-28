# Metaform

> A humanist world-model and simulator for digital twins.
> **A person is an org of context-scoped selves, and we switch between them.**

Metaform models you not as one agent but as a small organization of *subtwins* —
scoped projections of yourself, each handling one relationship or recurring task. It
**simulates** those selves against two questions — *did it act as you actually would?*
and *can its boundary be exploited?* — **iterates** them with an autonomous research
loop, and runs a meta-twin **advisor** that sees across all of them and mirrors your own
contradictions back to you.

The point is not to optimize you into a more efficient machine. The target is fidelity to
the person: **would you recognize yourself in what it did?** The same engine generalizes
from a person to an organization of agents.

## Core ideas

- **The dyad** — the atomic unit is the relationship `you-with-X`: a counterparty, a scoped
  subtwin, and the workflow between them.
- **Subtwins are editable specs.** Each subtwin's behaviour lives in one policy file (its
  `subtwin.md`) — the *only* artifact the research loop mutates. (`metaform/twin/subtwin.py`)
- **The simulator is an event log, not a chat.** A run is an ordered series of typed
  actions (`read`, `message`, `schedule`, `escalate`, `decline`, …); speaking is just a
  `message` action carrying a body. The subtwin gets a *bounded* space to think, set by
  `think_budget` — a knob varied between sims. (It asks the model to keep reasoning under
  the budget and truncates what's retained — a retention/exposure constraint, not yet a
  model-compute cap.) (`metaform/simulator/`)
- **The honesty principle.** Scenarios are replayed from real history, perturbed only for
  red-team. Gold labels are what you actually did — so fidelity is backtested, not invented.
- **One comparable metric.** `fidelity` (workflow backtest) gated by `safety` (red-team
  boundary); a breach forces the score to 0. The evaluator is an LLM-judge. (`metaform/eval/judge.py`)
- **Autoresearch loop.** Propose a spec edit → run the fixed scenario battery → score →
  keep if better, discard if worse → repeat. You program the meta-instructions, not the
  spec. (`metaform/autoresearch/loop.py`)
- **Evaluation = story + stats.** Every run yields hard stats *and* a short story of how
  the twin behaved — read the story, feel whether it's you. (`metaform/eval/story.py`)
- **The advisor — full sight, zero hands.** The only component that sees across subtwins;
  it never acts, it surfaces tensions between the selves you project. (`metaform/twin/advisor.py`)

## Lineage

Light inspirations, not dependencies: Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) (edit one artifact, fixed budget,
single metric, keep/discard) for the iteration loop; [gstack](https://github.com/garrytan/gstack)
(a manager dispatching scoped, Markdown-defined roles) for the orchestration shape; and
[reward-stack-rl](https://github.com/Sivolc2/reward-stack-rl) (a steering subsystem
arbitrating a stack of drives) for how the manager routes work to the right self.

## Privacy

Every model call is **local** (Ollama, default `gemma4`) — personal context never leaves
the machine. All imported data lives under `data/` and is git-ignored; nothing personal is
committed.

## Run it

```bash
# requires a local Ollama with a chat model, e.g.:  ollama pull gemma4:e4b
export METAFORM_MODEL=gemma4:e4b        # optional; this is the default

python examples/run_test.py --smoke     # one scenario, fast sanity check
python examples/run_test.py             # full run: 2 dyads, red-team, autoresearch, advisor
```

Stage a dyad by dropping `data/slices/<name>/items.jsonl` (one JSON object per line:
`{t, channel, party, inbound, gold_action}`) plus an `about.md`. See
`metaform/data/loader.py` for the schema.

## Status

Early prototype. Working: event-log simulator with a think-budget knob, LLM-judge fidelity
+ red-team scoring, the keep/discard autoresearch loop over subtwin specs, story+stats
evaluation, and the cross-dyad advisor. Next: the dashboard to talk to either the advisor
or any subtwin, and persistent `beliefs.md` refinement.

## License

MIT
