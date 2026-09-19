#!/usr/bin/env python3
"""Evaluate MMLU-generative answer extraction patterns on recorded responses.

Why this exists: PR #4188 (fixing the extraction bug reported in #4187) was measured on a
hand-written table first, and one suggested pattern turned out to be **worse** on real data.
This script is the measurement that showed it, so anyone can re-run it.

    python pattern_eval.py evidence/mmlu_generative_responses.jsonl

The corpus is 600 real generations from two models (deepseek-flash, deepseek-v4-pro)
x 10 MMLU subjects x 30 items, recorded by the harness (`resps[0]`), with the gold letter.

Two yardsticks are printed when possible, and they disagree slightly by design:

  * `emulation`   - "take the last regex match, else no answer", which is what the task's
                    filter effectively does for a caller-supplied pattern. No dependencies.
  * `real filter` - if `lm_eval` is importable, the actual `MultiChoiceRegexFilter` with the
                    settings from the PR's yaml (`group_select: -1`, `ignore_case: true`).
                    It has an internal fallback, so counts can differ from the emulation by a
                    few rows; reporting both keeps the difference visible instead of hidden.

Scoring follows the task metric: `exact_match` with `ignore_case` and `ignore_punctuation`,
implemented here as normalization rather than an HF `evaluate` call.
"""

import json
import logging
import re
import sys

logging.disable(logging.CRITICAL)   # every non-matching row makes lm_eval log a warning;
                                    # thousands of them turn a 30-second run into a timeout.

_CACHE = {}

PATTERNS = [
    ("first-line chain (current release)", "chain", None),
    (r"PR draft: \b([A-D])\b", "last", r"\b([A-D])\b"),
    (r"suggested-then-retracted: (?i)\b([A-D])\b", "last", r"(?i)\b([A-D])\b"),
    (r"start-anchored: (?m)^[\s*_`>\-]*\(?([A-D])\)?[.):]", "last", r"(?m)^[\s*_`>\-]*\(?([A-D])\)?[.):]"),
    (r"bare-letter whole response: ^\s*\(?([A-Da-d])\)?[.)]?\s*$", "last", r"^\s*\(?([A-Da-d])\)?[.)]?\s*$"),
]


def norm(s):
    """case- and punctuation-insensitive comparison, same intent as the task's metric."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def chain_extract(raw):
    """the current release's filter: first line, whitespace-stripped."""
    first = str(raw).split("\n")[0]
    return first.strip()


def last_match(pattern, raw):
    hits = re.findall(pattern, str(raw))
    if not hits:
        return None
    v = hits[-1]
    return v if isinstance(v, str) else (v[0] if v else None)


def real_filter(pattern, raw):
    try:
        from lm_eval.filters.extraction import MultiChoiceRegexFilter
    except Exception:  # noqa: BLE001
        return None
    f = _CACHE.get(pattern)
    if f is None:
        f = MultiChoiceRegexFilter(regex_pattern=pattern, group_select=-1, ignore_case=True)
        _CACHE[pattern] = f
    return f.apply([[raw]], [{"choices": ["a", "b", "c", "d"]}])[0][0]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "evidence/mmlu_generative_responses.jsonl"
    fast = "--fast" in sys.argv
    if not fast:
        print("note: the real-filter column runs lm_eval per row and took ~7 minutes on our machine;")
        print("      pass --fast to print the dependency-free emulation only.\n")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"corpus: {len(rows)} responses from {path}")
    models = sorted({r['model'] for r in rows})
    print(f"models: {', '.join(models)}")

    try:
        import lm_eval  # noqa: F401
        have_lm = not fast
    except Exception:  # noqa: BLE001
        have_lm = False
    print(f"lm_eval importable: {have_lm}  (emulation always printed; real filter {'too' if have_lm else 'unavailable'})\n")

    print(f"{'extraction':<52}{'emulation':>12}{'real filter':>14}")
    print("-" * 78)
    results = {}
    preds = {}          # name -> list of predictions (through the headline yardstick)
    for name, kind, pattern in PATTERNS:
        ok_em = ok_rf = 0
        col = []
        for r in rows:
            if kind == "chain":
                pred = chain_extract(r["raw"])
            else:
                pred = last_match(pattern, r["raw"])
            if pred is not None and norm(pred) == norm(r["gold"]):
                ok_em += 1
            if have_lm and kind != "chain":
                rp = real_filter(pattern, r["raw"])
                col.append(rp)
                if rp is not None and norm(rp) == norm(r["gold"]):
                    ok_rf += 1
            else:
                col.append(pred)
        preds[name] = col
        results[name] = (ok_em, ok_rf)
        rf = f"{ok_rf}/{len(rows)}" if have_lm and kind != "chain" else "-"
        print(f"{name:<52}{f'{ok_em}/{len(rows)}':>12}{rf:>14}", flush=True)

    # the flip that motivated the retraction — read off the SAME yardstick as the counts
    # above (no re-running the filter; it is slow because every non-match logs a warning).
    yard = "real filter" if have_lm else "emulation"
    print(f"\nflips between the PR draft and the retracted suggestion (via {yard}):")
    a, b = r"PR draft: \b([A-D])\b", r"suggested-then-retracted: (?i)\b([A-D])\b"
    worse = better = 0
    for i, r in enumerate(rows):
        x, y = preds[a][i], preds[b][i]
        ox = x is not None and norm(x) == norm(r["gold"])
        oy = y is not None and norm(y) == norm(r["gold"])
        worse += bool(ox and not oy)
        better += bool(oy and not ox)
    print(f"  correct -> wrong: {worse}    wrong -> correct: {better}")

    n_standalone = sum(1 for r in rows if re.search(r"(?i)(?<![a-z])(?:a|b|c|d)(?![a-z])", r["raw"]))
    n_bare = sum(1 for r in rows if re.fullmatch(r"[a-dA-D][.)]?", r["raw"].strip()))
    print(f"\nshape counts: standalone single-letter word present: {n_standalone}/{len(rows)}"
          f"   bare-letter response: {n_bare}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
