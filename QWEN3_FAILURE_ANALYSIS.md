# Why qwen3:8b produced no usable data

**Run:** 2026-08-11, job 4873994/4874032, `results/` (temperature 1.0)
**Verdict:** harness bug on our side, **not** a capability result for the model.
**Status:** fixed in `scripts/run_ollama.py`; re-run submitted as job 4878035 → `results_qwen3_fixed/`

---

## 1. The symptom

In `results/aggregate_metrics.csv`, qwen3:8b scores `0.0000` on every metric. Read
naively that says "the model failed the task." It did not — it never answered at all.

| | count | `done_reason` | `eval_count` | `raw_response` |
|---|---|---|---|---|
| failed | 114 / 120 | `length` | **256** (= `num_predict` exactly) | `""` — empty string |
| parsed | 6 / 120 | `stop` | 8 | `{"inferred_rule":"UNKNOWN"}` |

Across all three repetitions: 114, 118 and 119 of 120 items failed.

## 2. The decisive evidence

Two facts identify the cause and rule out the obvious alternative.

**`raw_response` is the empty string, not truncated text.** If a long answer had
simply been cut off at the token cap, the field would hold partial prose. It holds
nothing, while `eval_count` reports that a full 256 tokens *were* generated. The
tokens went somewhere the runner never read.

**The runner only ever read one field.** `scripts/run_ollama.py` did:

```python
raw = (resp.get("message") or {}).get("content", "")
```

Ollama routes reasoning-model output into a **separate `message.thinking` field**,
leaving `message.content` empty until the thinking block closes. qwen3 is a thinking
model. Its reasoning consumed the entire 256-token budget, generation hit the cap,
`message.content` never started, and the runner recorded an empty string.

## 3. Corroborating detail

The six items that *did* succeed were all `control_no_history_diagnosis` — the
trivial "no evidence available, answer UNKNOWN" controls. They were answered in
**8 tokens** with `done_reason: "stop"`. Those are precisely the prompts where the
model had nothing to reason about, so no thinking block was emitted and `content`
was populated normally. The failure tracks reasoning length, exactly as the
explanation predicts.

Two further consequences confirm the mechanism:

- **`attempts: 3` on every failed item.** The runner retried three times and got the
  identical empty result each time — the failure is deterministic, which sampling
  noise would not be.
- **Timing.** qwen3 averaged 2.36 s/item against 0.30–0.70 s for the other four
  models. It was generating a full 256 tokens every time, then discarding them.

## 4. Why the existing safety net missed it

The runner already had a `<think>...</think>` stripper:

```python
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
```

It could not help here for two independent reasons. It expects reasoning to arrive
*inline within* `content` as literal tags, but Ollama delivered it in a separate
JSON field. And the regex is non-greedy with a required closing tag — even inline,
reasoning truncated mid-thought never emits `</think>` and so would not match.

## 5. Not a temperature problem

The truncation is deterministic and orthogonal to sampling. The temperature-0 re-run
(job 4877990) therefore still contains a broken qwen3; it was deliberately left in so
that run changes only one variable relative to the original. This fix is a separate
run.

## 6. The fix

Three changes in `scripts/run_ollama.py`, defence in depth so that no single point
has to hold:

1. **Disable the reasoning channel.** New `--think {auto,on,off}` flag sets `think`
   in the `/api/chat` payload. With `off`, the whole `num_predict` budget goes to the
   answer. `auto` omits the field and keeps the previous server-default behaviour.
2. **Capture `message.thinking`.** It is now stored on every row, and if `content`
   yields no JSON the extractor falls back to the thinking text, tagging those rows
   `json_extraction: "from_thinking"`. An answer that lands in the reasoning channel
   can no longer be silently lost — this rescues the run even if `think: off` is not
   honoured by the server build.
3. **Raise the ceiling.** The re-run uses `--max-tokens 1024`. This does not affect
   cross-model comparability: the other four models finished in 12–17 eval tokens, so
   the 256 cap was never binding for them.

The manifest now records the `think` setting alongside temperature and seeds, so a
run's decoding configuration is always recoverable from its own output.

### Re-run command

```bash
MODELS="qwen3:8b" CONDITIONS=all120 REPS=3 \
TEMPERATURE=0 MAX_TOKENS=1024 EXTRA_ARGS="--think off" \
OUTDIR=$PROJ/results_qwen3_fixed \
sbatch run_benchmark_ollama.sh
```

## 7. How to read the old numbers

Until the re-run lands, qwen3's zeros in `results/aggregate_metrics.csv` are
**missing data, not scores**. They should be excluded from any model comparison
rather than reported as a floor. The same applies to qwen3 in `results_temp0/`.

## 8. The general lesson

The runner treated "no parseable JSON" as a model failure when it was a transport
failure. Any harness that reads a single response field should assert that the field
it read is non-empty whenever tokens were actually generated — `eval_count > 0` with
empty content is a contradiction that should raise, not score as zero. A silent zero
is far more dangerous than a crash, because it survives all the way into a results
table looking like a finding.
