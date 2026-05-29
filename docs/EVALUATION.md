# Quality Evaluation

HippocampAI includes a retrieval-**quality** evaluation harness in `bench/eval/`. It is
separate from `bench/runner.py`, which measures only latency/throughput.

## Why this matters

Speed numbers say nothing about whether the *right* memories come back. A memory engine
that is fast but returns irrelevant context is worse than useless — it quietly poisons
the agent's answers. The whole memory-engine market (Mem0, Zep, Letta, and others)
competes on accuracy benchmarks, primarily **LOCOMO** and **LongMemEval**, reported as:

- **Retrieval metrics** — did the relevant evidence get retrieved? (recall@k, etc.)
- **End-to-end QA accuracy** — given the retrieved context, is the final answer correct?

This harness produces both, using the same LLM-as-judge methodology those benchmarks
use, so HippocampAI's results are **directly comparable** to published numbers instead
of being unverifiable marketing claims.

## What it measures

| Metric | Question it answers |
|--------|---------------------|
| **recall@k** | Of the memories that contain the answer, how many appear in the top-k? |
| **precision@k** | Of the top-k retrieved, how many are actually relevant? |
| **MRR** | How high up is the first relevant memory? (1/rank) |
| **nDCG@k** | Ranking quality, discounting relevant hits that appear lower down. |
| **QA accuracy** | After answering from the retrieved context, is the answer correct (LLM-judged)? |

Retrieval metrics require the dataset to supply *evidence* (which messages hold the
answer). QA accuracy does not — it only needs the gold answer.

## Architecture

```
datasets.py   EvalSample model + loaders (LOCOMO, LongMemEval, bundled synthetic)
metrics.py    recall@k, precision@k, MRR, nDCG
judge.py      LLM-as-judge (answer-from-context, then grade) via provider adapters
harness.py    ingest -> retrieve -> (answer -> judge) -> aggregate metrics + report
run_eval.py   CLI entry point
```

For each conversation the harness ingests every message under a fresh synthetic user,
recording the memory id created for each source message. Each question then:

1. retrieves the top-k memories for the question text,
2. scores retrieval against the question's gold evidence (mapped to those memory ids),
3. optionally answers from the retrieved context and grades the answer with the judge.

Results are aggregated overall and per question category.

## Running it

### Offline smoke test (no setup, no LLM key)

```bash
python -m bench.eval.run_eval --dataset synthetic --no-qa
```

Uses a tiny bundled dataset and computes retrieval metrics only — useful to confirm the
harness and your Qdrant connection work.

### LOCOMO with end-to-end QA accuracy

```bash
python -m bench.eval.run_eval --dataset locomo --path ./locomo.json --k 10 \
    --output reports/locomo
```

Computes retrieval metrics **and** LLM-judged QA accuracy. Requires an LLM provider
configured (see [Providers](PROVIDERS.md)) — the judge uses it to answer and grade.

### LongMemEval, retrieval metrics only

```bash
python -m bench.eval.run_eval --dataset longmemeval --path ./longmemeval.json \
    --no-qa --max-samples 50
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `synthetic` | `synthetic`, `locomo`, or `longmemeval` |
| `--path` | — | Path to the benchmark JSON (required for locomo/longmemeval) |
| `--k` | `10` | Top-k for retrieval |
| `--no-qa` | off | Skip LLM-judge QA; retrieval metrics only |
| `--max-samples` | all | Limit number of conversations evaluated |
| `--max-questions-per-sample` | all | Limit questions per conversation |
| `--ingest-type` | `fact` | Memory type for ingested messages; an explicit type skips per-message LLM classification (faster, avoids rate limits) |
| `--output` | — | Path prefix; writes `<prefix>.json` and `<prefix>.md` reports |

## Datasets

Loaders read the **official on-disk JSON** for each benchmark — download the data
yourself and pass its path with `--path` (there is no auto-download). The bundled
`SyntheticDataset` needs no files and runs offline.

- **`LocomoDataset`** — reads LOCOMO's `conversation.session_N` turns and `qa` array.
- **`LongMemEvalDataset`** — reads `haystack_sessions` plus `answer_session_ids` /
  per-turn `has_answer` flags for evidence.
- **`SyntheticDataset`** — a small worked example; a plumbing test, not a benchmark score.

### Caveat: LOCOMO category encoding

LOCOMO encodes its five reasoning categories as integers, and that integer→label
encoding is **not stable across releases**. The default follows the widely-cited Mem0
eval (`1=single_hop, 2=multi_hop, 3=temporal, 4=open_domain, 5=adversarial`). If your
file differs, override it — otherwise the per-category report rows may be mislabeled
(overall metrics are unaffected):

```python
from bench.eval.datasets import LocomoDataset
ds = LocomoDataset("locomo.json", category_map={1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain", 5: "adversarial"})
```

Pass `category_map={}` to keep the raw integer as the label instead of guessing.

## Interpreting the report

The Markdown/JSON report contains an **Overall** block and a **By category** breakdown.
`retrieval_scored_questions` tells you how many questions had usable evidence — if this
is `0`, your loader isn't linking evidence and the retrieval metrics are meaningless, so
check the dataset format before trusting scores.

## Programmatic use

```python
from hippocampai import MemoryClient
from bench.eval.datasets import load_dataset
from bench.eval.harness import EvalHarness, EvalConfig
from bench.eval.judge import build_default_judge
from hippocampai.client import _initialize_llm
from hippocampai.config import get_config

dataset = load_dataset("locomo", "./locomo.json")
client = MemoryClient()
judge = build_default_judge(_initialize_llm(get_config()))
report = EvalHarness(client, judge=judge).run(dataset, EvalConfig(k=10))
print(report.to_markdown())
```

## See also

- [Benchmarks](benchmarks.md) — latency/throughput benchmarking (`bench/runner.py`)
- [Providers](PROVIDERS.md) — configuring the LLM used by the judge
