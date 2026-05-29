"""Evaluation orchestrator: ingest -> retrieve -> (answer -> judge) -> metrics.

For each :class:`~bench.eval.datasets.EvalSample` the harness ingests every
message under a fresh synthetic user, tracking the memory id returned for each
source message. Each question is then answered by retrieving the top-k memories
and scoring both retrieval quality (against gold evidence) and, optionally, QA
accuracy via :class:`~bench.eval.judge.LLMJudge`.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from bench.eval.datasets import EvalDataset, EvalSample
from bench.eval.judge import LLMJudge
from bench.eval.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class EvalConfig:
    """Knobs for an evaluation run."""

    k: int = 10
    do_qa: bool = True
    max_samples: Optional[int] = None
    max_questions_per_sample: Optional[int] = None
    ingest_type: str = "fact"


@dataclass
class QuestionResult:
    question_id: str
    category: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    has_evidence: bool
    qa_correct: Optional[bool] = None
    predicted: Optional[str] = None


@dataclass
class EvalReport:
    dataset: str
    k: int
    num_samples: int
    num_questions: int
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    qa_evaluated: int
    duration_sec: float
    results: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def to_markdown(self) -> str:
        lines = [
            f"# HippocampAI Quality Eval - {self.dataset}",
            "",
            f"- **Samples:** {self.num_samples}",
            f"- **Questions:** {self.num_questions}",
            f"- **k:** {self.k}",
            f"- **QA-judged questions:** {self.qa_evaluated}",
            f"- **Duration:** {self.duration_sec:.1f}s",
            "",
            "## Overall",
            "",
            "| Metric | Score |",
            "| --- | --- |",
        ]
        for key, val in self.overall.items():
            lines.append(f"| {key} | {val:.4f} |")
        lines += ["", "## By category", "", "| Category | Recall@k | MRR | nDCG@k | QA acc | n |", "| --- | --- | --- | --- | --- | --- |"]
        for cat, vals in sorted(self.by_category.items()):
            qa = vals.get("qa_accuracy")
            qa_str = f"{qa:.3f}" if qa is not None and not _is_nan(qa) else "-"
            lines.append(
                f"| {cat} | {vals.get('recall_at_k', 0):.3f} | {vals.get('mrr', 0):.3f} | "
                f"{vals.get('ndcg_at_k', 0):.3f} | {qa_str} | {int(vals.get('count', 0))} |"
            )
        return "\n".join(lines)


def _is_nan(x: float) -> bool:
    return x != x


class EvalHarness:
    """Runs a dataset through a HippocampAI client and scores the results."""

    def __init__(self, client: Any, judge: Optional[LLMJudge] = None) -> None:
        self.client = client
        self.judge = judge

    def _ingest(self, sample: EvalSample, user_id: str, ingest_type: str) -> dict[str, str]:
        """Ingest all messages; return a map of source msg_id -> memory id."""
        msg_to_mem: dict[str, str] = {}
        for session in sample.sessions:
            for msg in session.messages:
                text = f"{msg.speaker}: {msg.text}" if msg.speaker else msg.text
                memory = self.client.remember(
                    text=text,
                    user_id=user_id,
                    session_id=session.session_id,
                    type=ingest_type,
                )
                mem_id = getattr(memory, "id", None)
                if mem_id is not None:
                    msg_to_mem[msg.msg_id] = str(mem_id)
        return msg_to_mem

    def run(self, dataset: EvalDataset, config: EvalConfig) -> EvalReport:
        start = time.perf_counter()
        results: list[QuestionResult] = []
        num_samples = 0

        for sample in dataset.load():
            if config.max_samples is not None and num_samples >= config.max_samples:
                break
            num_samples += 1
            user_id = f"eval_{dataset.name}_{uuid.uuid4().hex[:8]}"
            msg_to_mem = self._ingest(sample, user_id, config.ingest_type)

            questions = sample.questions
            if config.max_questions_per_sample is not None:
                questions = questions[: config.max_questions_per_sample]

            for q in questions:
                retrieved = self.client.recall(query=q.question, user_id=user_id, k=config.k)
                retrieved_ids = [str(r.memory.id) for r in retrieved]
                expected_ids = [msg_to_mem[e] for e in q.evidence if e in msg_to_mem]
                has_evidence = bool(expected_ids)

                qr = QuestionResult(
                    question_id=q.question_id,
                    category=q.category,
                    recall_at_k=recall_at_k(retrieved_ids, expected_ids, config.k),
                    precision_at_k=precision_at_k(retrieved_ids, expected_ids, config.k),
                    mrr=reciprocal_rank(retrieved_ids, expected_ids),
                    ndcg_at_k=ndcg_at_k(retrieved_ids, expected_ids, config.k),
                    has_evidence=has_evidence,
                )

                if config.do_qa and self.judge is not None:
                    context = "\n".join(f"- {r.memory.text}" for r in retrieved)
                    predicted, correct = self.judge.answer_and_grade(
                        q.question, q.answer, context
                    )
                    qr.predicted = predicted
                    qr.qa_correct = correct

                results.append(qr)

        duration = time.perf_counter() - start
        return self._build_report(dataset.name, config, num_samples, results, duration)

    @staticmethod
    def _build_report(
        dataset_name: str,
        config: EvalConfig,
        num_samples: int,
        results: list[QuestionResult],
        duration: float,
    ) -> EvalReport:
        with_ev = [r for r in results if r.has_evidence]
        qa_results = [r for r in results if r.qa_correct is not None]

        overall = {
            "recall_at_k": aggregate([r.recall_at_k for r in with_ev]),
            "precision_at_k": aggregate([r.precision_at_k for r in with_ev]),
            "mrr": aggregate([r.mrr for r in with_ev]),
            "ndcg_at_k": aggregate([r.ndcg_at_k for r in with_ev]),
            "qa_accuracy": aggregate([1.0 if r.qa_correct else 0.0 for r in qa_results]),
            "retrieval_scored_questions": float(len(with_ev)),
        }

        buckets: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in results:
            buckets[r.category].append(r)

        by_category: dict[str, dict[str, float]] = {}
        for cat, items in buckets.items():
            ev_items = [r for r in items if r.has_evidence]
            qa_items = [r for r in items if r.qa_correct is not None]
            by_category[cat] = {
                "recall_at_k": aggregate([r.recall_at_k for r in ev_items]),
                "mrr": aggregate([r.mrr for r in ev_items]),
                "ndcg_at_k": aggregate([r.ndcg_at_k for r in ev_items]),
                "qa_accuracy": (
                    aggregate([1.0 if r.qa_correct else 0.0 for r in qa_items])
                    if qa_items
                    else float("nan")
                ),
                "count": float(len(items)),
            }

        return EvalReport(
            dataset=dataset_name,
            k=config.k,
            num_samples=num_samples,
            num_questions=len(results),
            overall=overall,
            by_category=by_category,
            qa_evaluated=len(qa_results),
            duration_sec=duration,
            results=results,
        )
