"""The manager / advisor twin — full sight, zero hands.

It is the only component that sees across subtwins. It does not act; it holds and
reflects (a container, not an operator). Here it does the one job that needs
cross-dyad sight: surface tensions between the selves you project.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..llm import chat_json


def find_tensions(subtwin_specs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Given several subtwin specs (id, role_self, spec), name preference tensions
    a person might want to reconcile. Advisory only."""
    blocks = "\n\n".join(
        f"### {s['id']} (role: {s['role_self']})\n{s['spec']}" for s in subtwin_specs
    )
    prompt = f"""You are an ADVISOR that helps a person understand themselves by looking across the
different scoped selves ("subtwins") they project into different parts of life. Below are
several subtwin specs. Identify up to 3 genuine TENSIONS or inconsistencies between how the
person acts across these roles (e.g. a value protected in one role and spent in another, or
conflicting stances). Be specific and humane — you are mirroring them back to themselves, not
optimizing them.

{blocks}

Return JSON list, each: {{"between": ["<id>","<id>"], "tension": "<one sentence>",
"question": "<a gentle question that helps them reconcile it>"}}"""
    try:
        v = chat_json(prompt, max_tokens=500, temperature=0.5)
        return v if isinstance(v, list) else []
    except Exception:
        return []
