"""CLI entry point for the HippocampAI quality evaluation harness.

Examples:
    # Offline smoke test (no LLM key, retrieval metrics + bundled data):
    python -m bench.eval.run_eval --dataset synthetic --no-qa

    # LOCOMO with end-to-end QA accuracy (needs an LLM provider configured):
    python -m bench.eval.run_eval --dataset locomo --path ./locomo10.json --k 10

    # LongMemEval, retrieval metrics only, first 50 samples:
    python -m bench.eval.run_eval --dataset longmemeval --path ./lme_s.json \\
        --no-qa --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from bench.eval.datasets import load_dataset
from bench.eval.harness import EvalConfig, EvalHarness
from bench.eval.judge import build_default_judge


def _build_client() -> Any:
    from hippocampai import MemoryClient

    return MemoryClient()


def _build_judge() -> Optional[Any]:
    """Build an LLM judge from the configured provider, or None if unavailable."""
    try:
        from hippocampai.client import _initialize_llm
        from hippocampai.config import get_config

        llm = _initialize_llm(get_config())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] could not initialize LLM for judge: {exc}", file=sys.stderr)
        return None
    return build_default_judge(llm)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HippocampAI retrieval-quality evaluation")
    parser.add_argument(
        "--dataset", default="synthetic", choices=["synthetic", "locomo", "longmemeval"]
    )
    parser.add_argument("--path", default=None, help="Path to the benchmark JSON file")
    parser.add_argument("--k", type=int, default=10, help="Top-k for retrieval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-questions-per-sample", type=int, default=None)
    parser.add_argument(
        "--ingest-type",
        default="fact",
        help="Memory type for ingested messages. An explicit type skips the LLM "
        "classification call per message (faster, avoids rate limits). Default: fact",
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip LLM-as-judge QA accuracy; compute retrieval metrics only",
    )
    parser.add_argument(
        "--output", default=None, help="Write JSON + Markdown report to this path prefix"
    )
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset, args.path)
    client = _build_client()
    judge = None if args.no_qa else _build_judge()
    if not args.no_qa and judge is None:
        print(
            "[warn] no LLM configured; falling back to retrieval-only metrics.",
            file=sys.stderr,
        )

    config = EvalConfig(
        k=args.k,
        do_qa=not args.no_qa and judge is not None,
        max_samples=args.max_samples,
        max_questions_per_sample=args.max_questions_per_sample,
        ingest_type=args.ingest_type,
    )

    harness = EvalHarness(client, judge=judge)
    report = harness.run(dataset, config)

    print(report.to_markdown())

    if args.output:
        prefix = Path(args.output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        prefix.with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")
        print(f"\nWrote report to {prefix.with_suffix('.json')} and {prefix.with_suffix('.md')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
