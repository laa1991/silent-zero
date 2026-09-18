#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_harness_specifics —— two harness defects that need **no API calls** to demonstrate.

  1. lm-eval `mmlu_*_generative` (`get_response`): the filtered prediction is the *whole first line*,
     compared to the gold letter with `exact_match` ⇒ `B. 4` or `ANSWER: B` score 0.
  2. OpenCompass `gsm8k_postprocess`: splits numeric answers at thousands separators
     ⇒ `\\boxed{$9{,}500}` -> `500`, `\\boxed{8,000}` -> `000`.

Both need the respective package installed; each block is skipped if the import fails, so the script
still runs and tells you what it could and could not check.

Usage:
    python check_harness_specifics.py
"""
import sys

BAR = "-" * 78


def lm_eval_block() -> bool:
    print(BAR)
    print("1) lm-eval · mmlu_*_generative · first-line filter + exact_match")
    try:
        from lm_eval.api.metrics import exact_match_hf_evaluate
        from lm_eval.filters.extraction import RegexFilter, WhitespaceFilter
        from lm_eval.filters.selection import TakeFirstFilter
    except Exception as e:  # noqa: BLE001
        print(f"   SKIP — lm_eval not importable here ({type(e).__name__}: {e})")
        return False

    chain = [RegexFilter(r"^(.*?)(?=\n|$)"), WhitespaceFilter(), RegexFilter(r"^(.*?)\s*$"), TakeFirstFilter()]
    rows = [r"B", "B.", "B. 4", r"B. \(6x^2 + 4x + 6\)", "ANSWER: B"]
    print("   raw response                      -> filtered prediction                exact_match vs 'B'")
    for raw in rows:
        resps = [[raw]]
        for f in chain:
            resps = list(f.apply(resps, [{}]))
        pred = resps[0][0] if isinstance(resps[0], list) else resps[0]
        em = exact_match_hf_evaluate(predictions=[pred], references=["B"],
                                     ignore_case=True, ignore_punctuation=True)["exact_match"]
        flag = "  <-- a correct answer scored wrong" if em == 0.0 and raw not in ("B", "B.") else ""
        print(f"   {raw!r:<35} -> {str(pred)!r:<37} {em}{flag}")
    print("   ⇒ the gold letter is never extracted from a longer line; the filter requires a bare letter.")
    return True


def opencompass_block() -> bool:
    print(BAR)
    print("2) OpenCompass · gsm8k_postprocess · thousands separators")
    try:
        from opencompass.datasets.gsm8k import gsm8k_postprocess
    except Exception as e:  # noqa: BLE001
        print(f"   SKIP — opencompass not importable here ({type(e).__name__}: {e})")
        return False

    for raw, expect in [(r"\boxed{9500}", "9500"),
                        (r"\boxed{$9{,}500}", "9500"),
                        (r"\boxed{8,000}", "8000"),
                        (r"\boxed{10,000}", "10000"),
                        ("The answer is $9{,}500.", "9500")]:
        got = gsm8k_postprocess(raw)
        flag = "" if got == expect else f"  <-- expected {expect}"
        print(f"   {raw!r:<28} -> {got!r:<10}{flag}")
    print("   ⇒ any comma-separated number is read as a different number, silently.")
    return True


if __name__ == "__main__":
    a = lm_eval_block()
    b = opencompass_block()
    print(BAR)
    print(f"   ran: lm-eval={'yes' if a else 'no'}  opencompass={'yes' if b else 'no'}")
    sys.exit(0 if (a or b) else 2)
