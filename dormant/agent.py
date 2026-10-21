"""The agent layer: a sentence in, validated tool calls out.

This is deliberately the least clever part of Dormant, and that is the point. The
model that decides whether a file is dead is a tabular foundation model doing
arithmetic over timestamps; language is only the handle you grab it with. So the
planner here is a keyword router by default, and the local-LLM path is opt-in via
DORMANT_LLM pointing at a model you run yourself. Nothing in the default configuration
reaches a network.

Two rules make the whole thing safe to wire to a UI: the planner can only name tools
that exist, and no tool that moves files runs without an explicit confirm flag.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

TOOLS = ("brief", "top", "archive", "undo", "scan", "forecast", "bench", "score")

_PATTERNS: list[tuple[str, re.Pattern, dict]] = [
    ("undo", re.compile(r"\b(undo|restore|put back|revert)\b", re.I), {}),
    ("bench", re.compile(r"\b(bench|compare|benchmark|scoreboard|versus|vs\.?)\b", re.I), {}),
    ("forecast", re.compile(r"\b(forecast|when.*(fill|full|run out)|disk.*(date|predict)|projection)\b", re.I), {}),
    ("scan", re.compile(r"\b(scan|index|rescan|refresh|look again)\b", re.I), {}),
    ("archive", re.compile(r"\b(archive|reclaim|free up|clear|tidy|move|stage|clean)\b", re.I), {}),
    ("top", re.compile(r"\b(which|what|list|show|dead|dormant|files)\b", re.I), {}),
    ("brief", re.compile(r"\b(brief|summary|how much|status|tell me)\b", re.I), {}),
]

_GB = re.compile(r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)", re.I)
_COUNT = re.compile(r"(\d+)\s+(?:file|top)")
_TOP_N = re.compile(r"top\s*(\d+)", re.I)


def parse(text: str) -> dict:
    """Map one sentence onto one or more tool calls."""
    calls: list[dict] = []
    matched = {name for name, pat, _ in _PATTERNS if pat.search(text)}
    budget = None
    if (m := _GB.search(text)):
        budget = float(m.group(1))
    n = None
    if (m := _TOP_N.search(text)):
        n = int(m.group(1))
    elif (m := _COUNT.search(text)):
        n = int(m.group(1))

    if "undo" in matched:
        calls.append({"tool": "undo", "args": {}})
    if "archive" in matched:
        calls.append({"tool": "archive", "args": {"top": n or 50, "max_gb": budget or 10.0, "confirm": False}})
        if budget or n:
            calls.append({"tool": "top", "args": {"limit": n or 50}})
    if "forecast" in matched:
        calls.append({"tool": "forecast", "args": {}})
    if "bench" in matched:
        calls.append({"tool": "bench", "args": {}})
    if "scan" in matched:
        calls.append({"tool": "scan", "args": {}})
    if "top" in matched and not any(c["tool"] == "top" for c in calls):
        calls.append({"tool": "top", "args": {"limit": n or 20}})
    if "brief" in matched or not calls:
        calls.append({"tool": "brief", "args": {}})
    return {"intent": "+".join(sorted(matched)) or "brief", "calls": calls}


def local_llm_plan(text: str) -> dict | None:
    """Optional: ask a model you host yourself. Off unless DORMANT_LLM is set."""
    model = os.environ.get("DORMANT_LLM")
    if not model:
        return None
    endpoint = os.environ.get("DORMANT_LLM_URL", "http://127.0.0.1:11434/api/generate")
    prompt = (
        "You route one sentence about a local disk-tidying tool to JSON with keys "
        f"intent and calls, where each call has tool in {list(TOOLS)} and args. "
        "Never invent a tool. Never set confirm to true. Reply with JSON only.\n"
        f"Sentence: {text}"
    )
    try:
        import urllib.request

        body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json"}).encode()
        with urllib.request.urlopen(endpoint, body, timeout=20) as res:
            raw = json.loads(res.read().decode()).get("response", "")
        parsed = json.loads(raw)
        calls = [c for c in parsed.get("calls", []) if c.get("tool") in TOOLS]
        for c in calls:
            c.setdefault("args", {})
            if c["tool"] == "archive":
                c["args"]["confirm"] = False
        return {"intent": parsed.get("intent", "llm"), "calls": calls, "planner": f"local:{model}"}
    except Exception as exc:
        return {"intent": "llm-failed", "calls": [], "planner": f"local:{model}", "error": str(exc)[:120]}


def handle(conn, text: str, *, confirm: bool = False) -> dict:
    """Plan, then execute read-only tools. Mutating tools stay planned unless confirmed."""
    plan = parse(text)
    llm = local_llm_plan(text)
    if llm and llm.get("calls"):
        plan = llm
    from . import action, engine
    from .store import connect as _connect  # noqa: F401  (kept for symmetry with callers)

    executed: list[dict] = []
    blocked: list[dict] = []
    for call in plan["calls"]:
        tool, args = call["tool"], dict(call.get("args") or {})
        try:
            if tool == "brief":
                st = engine.state(conn)
                executed.append({"tool": tool, "result": st["brief"]})
            elif tool == "top":
                executed.append({"tool": tool,
                                 "result": engine.top_flagged(conn, limit=args.get("limit", 20))})
            elif tool == "forecast":
                from .forecast import forecast

                executed.append({"tool": tool,
                                 "result": {k: v for k, v in forecast(conn).items() if k != "curve"}})
            elif tool == "scan":
                executed.append({"tool": tool, "result": {"queued": True}})
            elif tool == "bench":
                from .model import bench as run_bench

                executed.append({"tool": tool, "result": run_bench(conn)["results"]})
            elif tool == "archive":
                if not confirm:
                    blocked.append({"tool": tool, "args": args,
                                    "why": "moves files; pass confirm=true explicitly"})
                    continue
                rows = engine.top_flagged(conn, limit=args.get("top", 50))
                executed.append({"tool": tool, "result": action.archive(
                    conn, [r["id"] for r in rows], max_bytes=int(args.get("max_gb", 10) * 1e9),
                    reason="agent")})
            elif tool == "undo":
                if not confirm:
                    blocked.append({"tool": tool, "why": "moves files; pass confirm=true explicitly"})
                    continue
                executed.append({"tool": tool, "result": action.undo(conn)})
        except Exception as exc:
            executed.append({"tool": tool, "error": f"{type(exc).__name__}: {exc}"[:200]})
    return {"said": text, "planner": plan.get("planner", "keyword"), "intent": plan["intent"],
            "executed": executed, "blocked": blocked}
