"""Retrieval-quality evaluation harness for HippocampAI.

Unlike ``bench/runner.py`` (which measures latency/throughput only), this package
measures *answer quality*: retrieval metrics (recall@k, precision@k, MRR, nDCG)
and end-to-end QA accuracy via LLM-as-judge, on LOCOMO / LongMemEval-style data.

Entry point: ``python -m bench.eval.run_eval --help``.
"""

from bench.eval.datasets import (
    EvalDataset,
    EvalMessage,
    EvalQuestion,
    EvalSample,
    EvalSession,
    LocomoDataset,
    LongMemEvalDataset,
    SyntheticDataset,
    load_dataset,
)
from bench.eval.harness import EvalConfig, EvalHarness, EvalReport
from bench.eval.judge import LLMJudge
from bench.eval import metrics

__all__ = [
    "EvalDataset",
    "EvalMessage",
    "EvalQuestion",
    "EvalSample",
    "EvalSession",
    "LocomoDataset",
    "LongMemEvalDataset",
    "SyntheticDataset",
    "load_dataset",
    "EvalConfig",
    "EvalHarness",
    "EvalReport",
    "LLMJudge",
    "metrics",
]
