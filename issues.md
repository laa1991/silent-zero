# Upstream reports

Four defects found while doing this work, each filed with a **self-contained minimal reproduction**
so that a maintainer can verify it without reading any of our other material.

| Repository | Issue | Defect |
|---|---|---|
| `UKGovernmentBEIS/inspect_evals` | [#2473](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2473) | `mmlu_0_shot` defaults to `max_non_cot_tokens = GPT_5_MIN_TOKENS = 16` — an **API minimum** for one model reused as a **cap**. With a reasoning model on the Responses API every item returns an empty completion and the run reports **accuracy 0.000, `status=success`, `error=None`**. Reproduction: the same task and the same 16 on the chat-completions path scores **0.933**. |
| `UKGovernmentBEIS/inspect_ai` | [#5467](https://github.com/UKGovernmentBEIS/inspect_ai/issues/5467) | Feature request: warn or count when the output budget was consumed by the reasoning channel (`reasoning_tokens == max_tokens`) and the completion is empty. Today the only trace is in `usage`; the scorer cannot distinguish "answered wrongly" from "never answered". |
| `EleutherAI/lm-evaluation-harness` | [#4187](https://github.com/EleutherAI/lm-evaluation-harness/issues/4187) | `mmlu_*_generative`'s `get_response` filter compares the **whole first line** with the gold letter via `exact_match` ⇒ `B. 4`, `B. \(...\)` and even `ANSWER: B` all score **0.0**. Observed at scale: **0.000 (0/298) for two models, with 0/298 empty responses**. |
| `open-compass/opencompass` | [#2647](https://github.com/open-compass/opencompass/issues/2647) | `gsm8k_postprocess` splits numeric answers at thousands separators: `\boxed{$9{,}500}` → `500`, `\boxed{8,000}` → `000`. Silent — indistinguishable from a wrong answer. |

## The framing we used in all four (reusable)

> **Silent zero**: the harness reports a *successful* run whose score is ~0, with no error, no warning,
> and no indication that the model's entire output budget was consumed by its reasoning span
> (`output_tokens == reasoning_tokens == max_tokens`, `completion == ""`).
> We hit this while comparing three independent harnesses on the same model and the same items;
> the difference between 0.000 and 0.933 came down to **which endpoint/field carries the output budget**.
> We then reproduced the same shape **on a second vendor** (chat-completions only): at `max_tokens` 16
> and 256 the response is `finish_reason='length'` and the content contains only the model's inline
> `<think>…</think>` text — **no `ANSWER`**; at 4096 it is `finish_reason='stop'` and contains `ANSWER: C`.
> ⇒ **In every zero we measured, the API already said so (`finish_reason ∈ {length, max_tokens}`)
> and the harness said nothing.**

## Status

All four are open and unanswered as of 2026-09-18 (the day they were filed).
Corrections, counter-examples and closed-as-wontfix verdicts are all welcome — please open an issue or
a discussion here, or comment on the upstream thread directly.
