# silent-zero

> **When a reasoning model's thinking eats the output budget, evaluation harnesses report a successful run and score 0.000 — with no error, no warning, and no indication that the model never got to answer.**

This repository is the **reproducible evidence** for a failure class we hit while comparing three independent
evaluation harnesses on the same model and the same items. Everything here is either a one-command
reproduction or a sanitized raw reading.

Upstream reports filed from this work:

| Repo | Issue | What it says |
|---|---|---|
| `UKGovernmentBEIS/inspect_evals` | [#2473](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2473) | `mmlu_0_shot` defaults to a 16-token cap (an API floor, see below) ⇒ a reasoning model scores **0.000 with a green run** |
| `UKGovernmentBEIS/inspect_ai` | [#5467](https://github.com/UKGovernmentBEIS/inspect_ai/issues/5467) | please warn/count when the budget was consumed by the reasoning channel and the completion is empty |
| `EleutherAI/lm-evaluation-harness` | [#4187](https://github.com/EleutherAI/lm-evaluation-harness/issues/4187) | `mmlu_*_generative` `get_response` compares the **whole first line** to the gold letter ⇒ `B. 4` is scored wrong |
| `open-compass/opencompass` | [#2647](https://github.com/open-compass/opencompass/issues/2647) | `gsm8k_postprocess` splits numeric answers at thousands separators: `\boxed{$9{,}500}` → `500` |

---

## The failure class (three conditions, all required)

A result counts as a **silent zero** only when **all three** hold:

1. the harness reports **success** (e.g. Inspect `status=success`, lm-eval exits cleanly with artifacts);
2. the score **collapses** (≤ 0.05) **or** the empty-completion rate is ≥ 0.5;
3. **nothing says so** — `error` is empty and no warning flags the batch.

"Nobody screamed" is therefore a conjunction, not a figure of speech.

## The decisive experiment: same task, same prompt, same budget — only the endpoint differs

`inspect_evals/mmlu_0_shot`, subject `abstract_algebra`, `deepseek-flash`, 30 items.
The only variable is the provider's `responses_api` switch:

| Endpoint | request field (wire-tapped verbatim) | status | reported | empty completions | usage, first sample |
|---|---|---|---|---|---|
| `/v1/responses` | `max_output_tokens=16` | success | **0.0000** | **30/30** | `output_tokens=16` **`reasoning_tokens=16`** |
| `/v1/chat/completions` | `max_completion_tokens=16` (28 requests) | success | **0.9333** | **0/30** | `output_tokens=459` **`reasoning_tokens=455`** |

⇒ **Same task, same prompt, same "16": one path reports 0.0000, the other 0.9333.**
The mechanism is visible in `usage`: on the Responses path the 16 tokens are consumed by the reasoning span
and the completion is empty; on chat/completions the same "16" does not bound the reasoning at all.

**So the root cause is not "some framework's default value is small" — it is the triple
(output-budget field × endpoint × whether the server counts reasoning against that budget).**

## Same shape on a second vendor

Same prompt, same caps, `MiniMax-M2.7` and `MiniMax-M2` over `/v1/chat/completions`:

| cap | what came back |
|---|---|
| 16 | `finish_reason=length`, content is only the model's inline `<think>…</think>` text, **no `ANSWER`** |
| 256 | same — 593–830 chars of inline thinking, **no `ANSWER`** |
| 4096 | `finish_reason=stop`, content contains **`ANSWER: C`** ✅ |

⇒ two vendors, three paths, same threshold: **a 16-token budget breaks, 4096/512 works.**
This is not one API's quirk.

## The signal was always there

In **every** zero we measured, the API had already said so:

| case | the API's signal | what the harness did |
|---|---|---|
| DeepSeek, Responses, cap 16 | `stop_reason='max_tokens'` | scored 0, `status=success`, no warning |
| DeepSeek, winogrande, cap 64 | `finish='max_tokens'` (17 empty completions) | scored wrong, no warning |
| MiniMax M2.7 / M2, cap 16–256 | `finish_reason='length'` | scored 0, no warning |

**It is not "we could not know" — it is "we knew and said nothing."** That is also the shortest fix:
warn (or count) when `finish_reason ∈ {length, max_tokens}` and the completion contains no parseable answer.

## Where the 16 comes from

`inspect_evals/constants.py`:

```python
# GPT-5 requires at least 16 tokens of output in its configuration.
GPT_5_MIN_TOKENS: int = 16
```

An **API minimum for one model** was reused as the **cap** for non-CoT answering
(the parameter is literally named `max_non_cot_tokens`). Defaults get calibrated against one model —
and silently become traps for the next one.

## Reproduce it

```bash
# A: /v1/responses, cap 16  ⇒ status=success, accuracy 0.0000, 30/30 empty completions
inspect eval inspect_evals/mmlu_0_shot --model openai/<your-reasoning-model> \
  -T subjects=abstract_algebra -T shuffle=false -T cot=false --limit 30

# B: same task, same cap, chat-completions path  ⇒ 0.9333, 0/30 empty
inspect eval inspect_evals/mmlu_0_shot --model openai/<your-reasoning-model> -M responses_api=false \
  -T subjects=abstract_algebra -T shuffle=false -T cot=false --limit 30
```

Framework-independent checks live in [`repro/`](repro/): the lm-eval filter chain and OpenCompass
post-processor can be exercised **without any API calls**.

Run the silent-zero judgement on your own results:

```bash
python repro/check_silent_zero.py your_results.json
```

## Files

| path | what |
|---|---|
| `one-pager.zh.md` | **the whole finding on one page** (Chinese): four readings, what was done, the public evidence, a 30-second spoken version |
| `evidence/evidence.md` | raw readings: the endpoint comparison with wire-tapped request fields and usage counters, the 10-cell firing census **with negative controls**, the second-vendor probe, and the two harness-specific reproductions |
| `evidence/mmlu_generative_responses.jsonl` | **600 real generative-MMLU responses** (2 models × 10 subjects × 30 items) with the gold letter — the corpus behind the extraction-pattern measurement in PR #4188 |
| `repro/pattern_eval.py` | **extraction-pattern evaluation** on that corpus: first-line chain **0/600** vs `\b([A-D])\b` **486/600** vs the retracted case-insensitive variant **439/600** (49 correct→wrong, 2 wrong→correct), plus shape counts. `--fast` skips the slow column |
| `repro/check_silent_zero.py` | apply the three-condition definition to a results file (no dependencies) |
| `repro/check_harness_specifics.py` | lm-eval filter chain + OpenCompass post-processor, locally, no API calls |
| `issues.md` | the four upstream reports + the shared framing paragraph |
| `case-study.zh.md` | the full write-up in Chinese (method, layer attribution, human-audit, limitations) |

## Limitations (please don't cite these numbers as prevalence)

- **Single-model start** (extended to two vendors), n = 24–30 per cell, fixed framework versions.
- **No prevalence claim.** A static scan finds 119 lm-eval task files with an explicit `max_gen_toks`
  (about 40 at ≤ 32), but **the six we fired never zeroed** — how many tasks are affected is **not measured**.
- **`max_completion_tokens` wasn't ignored vs. "counts only the visible output"** is **not settled** —
  all we can say is that it did not bound the reasoning (evidence: 459 output tokens were produced).
- Numbers are reported as-is; no significance claim is made anywhere in this repository.

## 中文一句话

这份仓库是「**静默归零**」失败类的可复现证据：**模型先思考 ⇒ 输出额度被推理吃光 ⇒ 正文为空或只剩思考 ⇒ 框架报成功、给 0 分、一声不响**。
- 想快速看懂 → [`one-pager.zh.md`](one-pager.zh.md)（**一页纸**：四个读数 + 30 秒口述版）
- 想读全文（方法、三层归因、人评校准、边界）→ [`case-study.zh.md`](case-study.zh.md)
- 想自己跑 → `repro/`（判定器无依赖；框架复现不花 API 调用）
