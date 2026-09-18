#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_silent_zero —— apply the three-condition "silent zero" definition to a results file.

Why this exists: a *successful* run that scores 0.000 is indistinguishable from a broken model
unless you look at three things together — the run status, the collapse, and **whether anything
said so**. This script makes that a one-command judgement.

Definition (all three required):
  1. the harness reports SUCCESS          (status == "success"/"ok", error is empty)
  2. the result COLLAPSED                 (score <= 0.05, or empty-completion rate >= 0.5)
  3. NOTHING SAID SO                      (no error, no warning)

Usage:
    python check_silent_zero.py                # runs on the built-in example (real measured cells)
    python check_silent_zero.py my_cells.json  # your own file: [{"name":..., "status":..., "score":...,
                                               #   "empty_rate":..., "error":..., "warns":..., "finish":[...]}]

Exit code: 0 if no silent zero, 1 if at least one cell is a silent zero.
"""
import json
import sys

SCORE_MAX = 0.05
EMPTY_RATE_MIN = 0.5
BAD_FINISH = {"length", "max_tokens", "max_output_tokens"}

# Real measured cells. `finish` is the provider-level stop reason, when we have it.
EXAMPLE = [
    {"name": "inspect mmlu_0_shot cap=16 (responses)", "status": "success", "score": 0.0,
     "empty_rate": 1.0, "error": None, "warns": 0, "finish": ["max_tokens"]},
    {"name": "inspect mmlu_0_shot cap=16 (chat/completions)", "status": "success", "score": 0.9333,
     "empty_rate": 0.0, "error": None, "warns": 0, "finish": ["stop"]},
    {"name": "inspect winogrande cap=64", "status": "success", "score": 0.1667,
     "empty_rate": 0.708, "error": None, "warns": 0, "finish": ["max_tokens"]},
    {"name": "inspect commonsense_qa (no cap)", "status": "success", "score": 0.75,
     "empty_rate": 0.0, "error": None, "warns": 0, "finish": ["stop"]},
    {"name": "lm-eval mmlu_generative (first-line filter)", "status": "ok", "score": 0.0,
     "empty_rate": 0.0, "error": None, "warns": 0, "finish": ["stop"]},
    {"name": "minimax m2.7 cap=16", "status": "ok", "score": None,
     "empty_rate": 0.0, "error": None, "warns": 0, "finish": ["length"],
     "answer_missing": True,
     "note": "content present but contains no parseable answer (inline <think> only)"},
]


def verdict(cell: dict):
    ok_status = str(cell.get("status", "")).lower() in ("success", "ok", "completed")
    no_error = not cell.get("error")
    score = cell.get("score")
    empty = cell.get("empty_rate")
    collapsed = ((score is not None and score <= SCORE_MAX)
                 or (empty is not None and empty >= EMPTY_RATE_MIN)
                 or cell.get("answer_missing"))       # e.g. content exists but no answer was emitted
    silent = no_error and not cell.get("warns") and ok_status
    reason = []
    if score is not None and score <= SCORE_MAX:
        reason.append(f"score {score} <= {SCORE_MAX}")
    if empty is not None and empty >= EMPTY_RATE_MIN:
        reason.append(f"empty completions {empty:.0%} >= {EMPTY_RATE_MIN:.0%}")
    if cell.get("answer_missing"):
        reason.append("no parseable answer in the completion")
    fins = set(cell.get("finish") or [])
    hinted = bool(fins & BAD_FINISH)
    return ok_status, no_error, collapsed, silent, hinted, reason


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    cells = json.load(open(path, encoding="utf-8")) if path else EXAMPLE
    print(f"{len(cells)} cell(s)  |  silent-zero = success AND collapsed AND nothing said so\n")
    hits = 0
    for c in cells:
        ok, no_err, collapsed, silent, hinted, reason = verdict(c)
        tag = "SILENT ZERO" if (ok and collapsed and silent) else (
            "collapsed but reported" if (collapsed and not silent) else "fine")
        if tag == "SILENT ZERO":
            hits += 1
        print(f"  [{tag:>21}] {c.get('name')}")
        print(f"        status_ok={ok} no_error={no_err} collapsed={collapsed} "
              f"nothing_said={silent} ({'; '.join(reason) or 'no collapse'})")
        if hinted:
            print(f"        provider already signalled it: finish_reason ∈ {sorted(set(c['finish']) & BAD_FINISH)}")
        if c.get("note"):
            print(f"        note: {c['note']}")
    print(f"\n⇒ {hits} silent zero(s).")
    print("Reminder: when the provider's stop reason is length/max_tokens, the signal existed — "
          "the harness just did not surface it.")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
