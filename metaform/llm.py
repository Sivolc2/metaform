"""Local model layer — talks to Ollama (gemma4) over HTTP. Stdlib only.

Personal context stays on the box: every model call is local. The installed
model emits reasoning-wrapped output, so JSON extraction is defensive.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("METAFORM_MODEL", "gemma4:e4b")


def chat(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.6,
    model: Optional[str] = None,
) -> str:
    """One non-streaming chat completion. Returns raw assistant text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {
            "model": model or MODEL,
            "messages": messages,
            "stream": False,
            "think": False,  # ask thinking-style models to skip the reasoning wrapper
            "keep_alive": "10m",  # keep the model resident between calls
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    ).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
            return data["message"]["content"]
        except Exception as exc:  # cold-start disconnects, transient errors
            last = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"ollama chat failed after retries: {last}")


def warmup(model: Optional[str] = None) -> bool:
    """Load the model into memory so the first real call doesn't cold-start out."""
    try:
        chat("ok", max_tokens=1, temperature=0.0, model=model)
        return True
    except Exception:
        return False


def chat_json(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 800,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> Any:
    """Chat and parse JSON out of the (possibly reasoning-wrapped) reply."""
    instr = "\n\nReturn ONLY a single JSON value. No prose, no markdown fences."
    raw = chat(prompt + instr, system, max_tokens, temperature, model)
    try:
        return extract_json(raw)
    except ValueError:
        # one stricter retry
        raw = chat(
            prompt + instr + " Output must start with { or [.",
            system,
            max_tokens,
            0.0,
            model,
        )
        return extract_json(raw)


def extract_json(text: str) -> Any:
    """Pull the first valid JSON object/array out of arbitrary model text."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip code fences
    if "```" in text:
        for part in text.split("```"):
            p = part.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p[:1] in "{[":
                try:
                    return json.loads(p)
                except Exception:
                    continue
    # balanced-delimiter scan (handles reasoning-wrapped output)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        if start == -1:
            continue
        depth = 0
        in_str = False
        i = start
        while i < len(text):
            c = text[i]
            if in_str:
                if c == "\\":
                    i += 2  # skip the escaped character; handles \\ and \"
                    continue
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_c:
                    depth += 1
                elif c == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break
            i += 1
    raise ValueError("no JSON found in model output")


def health() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False
