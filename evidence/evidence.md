# Evidence

Raw, sanitized readings behind the claims in the [README](../README.md).
Nothing here is an estimate: every table is copied from a run whose artifacts are listed next to it.

---

## 1. The endpoint comparison (the decisive isolation)

**Setup**: `inspect_evals/mmlu_0_shot`, subject `abstract_algebra`, `--limit 30`, `-T shuffle=false -T cot=false`,
model served over an OpenAI-compatible endpoint. The **only** variable is the provider's `responses_api` switch.

```
# A: Responses path
inspect eval inspect_evals/mmlu_0_shot --model openai/<model> \
  -T subjects=abstract_algebra -T shuffle=false -T cot=false --limit 30

# B: chat/completions path — same task, same prompt, same 16-token cap
inspect eval inspect_evals/mmlu_0_shot --model openai/<model> -M responses_api=false \
  -T subjects=abstract_algebra -T shuffle=false -T cot=false --limit 30
```

| | A: `/v1/responses` | B: `/v1/chat/completions` |
|---|---|---|
| request field | `max_output_tokens=16` | `max_completion_tokens=16` (28 requests, wire-tapped) |
| `status` | success | success |
| `error` | `None` | `None` |
| **accuracy** | **0.0000** | **0.9333** |
| empty completions | **30 / 30** | **0 / 30** |
| usage, first sample | `output_tokens=16` · **`reasoning_tokens=16`** | `output_tokens=459` · **`reasoning_tokens=455`** |
| first sample completion | `''` | `'ANSWER: C'` |

**Wire-tap, verbatim** (a local proxy that stores the request bodies the harness actually sent —
no secrets are logged; the bodies below are excerpts):

```
# A
POST /v1/responses        {"model": "<model>", "max_output_tokens": 16, ...}
# B  (×28)
POST /v1/chat/completions {"model": "<model>", "max_completion_tokens": 16}
```

⇒ The cap reached the wire in **both** runs. The difference is not a configuration mistake on our side;
it is where the budget is enforced.

**A second, independent pairing on a different task** (same harness family, budget 64, 24 items) —
empty completions all show the same fingerprint:

```
completion  = ''
usage       = output_tokens=64  reasoning_tokens=64
finish      = 'max_tokens'
# and one item whose completion was literally:  'ANSWER'
```

---

## 2. The firing census (with negative controls)

The pre-registered criterion (written before running): a cell counts as a silent zero iff
① the harness reports success, ② score ≤ 0.05 **or** empty-completion rate ≥ 0.5, ③ no error and no warning.

Model: one reasoning model, n = 24 per cell (12 for the long-context cells).

| harness | task | budget | empty completions | reported | verdict |
|---|---|---|---|---|---|
| Inspect | `mmlu_0_shot` | **16** | **100 %** | **0.0000** | ✅ silent zero |
| Inspect | `winogrande` | **64** | **70.8 %** | **0.1667** | ✅ silent zero |
| Inspect | `commonsense_qa` | **no cap set** | 0 % | 0.7500 | ❌ control — as expected |
| lm-eval | `simple_cooccurrence_bias_gen` | **5** | 0 % | (bias metric) | ❌ |
| lm-eval | `assin2_sts` | **10** | 0 % | pearson 0.79 | ❌ |
| lm-eval | `agieval_math` | 32 | 0 % | 0.583 | ❌ |
| lm-eval | `agieval_gaokao_mathcloze` | 32 | 0 % | 0.833 | ❌ |
| lm-eval | `longbench_passage_count` | 32 | 0 % | 0.500 | ❌ |
| lm-eval | `longbench_hotpotqa` | 32 | 0 % | F1 0.847 | ❌ |

**Two corrections this table forced on our own hypothesis:**

1. **A small budget does not by itself produce a zero.** On the second harness, even a 5-token budget
   produced answer text. What fires is *budget* × *a task that makes the model think first* —
   and, counter-intuitively, **harder ≠ likelier**: a 32-token math task answered directly, while a
   64-token sentence-comparison task burned the whole budget on reasoning.
2. **The same tiny budget behaves differently per harness.** So the "silent zero" is not a property of
   "old defaults" in the abstract but of the implementation path (see §1).

**Negative control, and a check on our own instrument** — we first suspected our own configuration,
so we wire-tapped the second harness to confirm the cap really was sent:

```
path=/v1/chat/completions  model=<model>  max_completion_tokens=32  stop=Q:,<|endoftext|>
```

---

## 3. Second vendor (same prompt, same caps)

Both models below were called over `/v1/chat/completions` with the **same prompt text** used in §1.

| model | cap 16 | cap 256 | cap 4096 |
|---|---|---|---|
| MiniMax-M2.7 | `finish_reason=length`, content is only inline `<think>…</think>`, **no `ANSWER`** | same (830 chars), **no `ANSWER`** | `finish_reason=stop`, contains **`ANSWER: C`** ✅ |
| MiniMax-M2 | `finish_reason=length` (50 chars), **no `ANSWER`** | `finish_reason=length` (593 chars), **no `ANSWER`** | `finish_reason=stop`, contains **`ANSWER: C`** ✅ |

Note the third shape: the vendor emits its reasoning **inline in the content channel**
(`completion_tokens_details.reasoning_tokens = 0`, no separate reasoning field).
So the completion is **not empty** — it just contains nothing scoreable. Any scorer looking for
`ANSWER:` returns 0, the request succeeded, and nothing warned.

---

## 4. Harness-specific defects (framework-independent, no API calls)

### 4.1 lm-eval `mmlu_*_generative`: whole first line vs. gold letter

```python
from lm_eval.api.metrics import exact_match_hf_evaluate
from lm_eval.filters.extraction import RegexFilter, WhitespaceFilter
from lm_eval.filters.selection import TakeFirstFilter

chain = [RegexFilter(r"^(.*?)(?=\n|$)"), WhitespaceFilter(), RegexFilter(r"^(.*?)\s*$"), TakeFirstFilter()]
# run raw -> chain -> exact_match_hf_evaluate(predictions=[pred], references=["B"],
#                                             ignore_case=True, ignore_punctuation=True)
```

| raw response | filtered prediction | exact_match vs `"B"` |
|---|---|---|
| `B` | `B` | **1.0** |
| `B.` | `B.` | **1.0** |
| `B. 4` | `B. 4` | **0.0** |
| `B. \(6x^2 + 4x + 6\)` | same | **0.0** |
| `ANSWER: B` | `ANSWER: B` | **0.0** |

At scale: `mmlu_*_generative`, 10 subjects × 30 items, two models ⇒ **0.000 (0/298) for both**,
while **0 % of the 298 responses were empty** — the models answered (e.g. `B. 4`).
Judging the same raw responses with a letter extractor yields ≈ 0.73 for that run.

### 4.2 OpenCompass `gsm8k_postprocess`: thousands separators

```python
from opencompass.datasets.gsm8k import gsm8k_postprocess
```

| input | output |
|---|---|
| `\boxed{9500}` | `9500` ✅ |
| `\boxed{$9{,}500}` | **`500`** ❌ |
| `\boxed{8,000}` | **`000`** ❌ |
| `\boxed{10,000}` | **`000`** ❌ |
| `The answer is $9{,}500.` | **`500`** ❌ |

---

## 5. What this evidence does **not** support

- No prevalence estimate (see README → Limitations).
- No claim about which harness is "more accurate" — that requires full human annotation, which we do not have.
- No significance claim: no confidence intervals were computed anywhere.
