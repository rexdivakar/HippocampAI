"""Dataset abstractions and loaders for memory-quality evaluation.

A dataset yields :class:`EvalSample` objects. Each sample is one conversation
(mapped to one synthetic user) consisting of multiple sessions of messages, plus
a set of question/answer pairs to probe recall over that history.

Loaders read the *official* on-disk JSON for each benchmark from a local path you
provide (no network access). A small bundled :class:`SyntheticDataset` lets the
harness run offline out of the box.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class EvalMessage:
    """A single conversational turn to be ingested as memory."""

    speaker: str
    text: str
    msg_id: str
    timestamp: Optional[str] = None


@dataclass
class EvalSession:
    """An ordered group of messages (a conversation session)."""

    session_id: str
    messages: list[EvalMessage]


@dataclass
class EvalQuestion:
    """A probe over the ingested history.

    ``evidence`` holds the message ids (``EvalMessage.msg_id``) that contain the
    answer, when the benchmark provides them. They drive retrieval metrics; QA
    accuracy via the judge does not require them.
    """

    question_id: str
    question: str
    answer: str
    category: str = "single_hop"
    evidence: list[str] = field(default_factory=list)


@dataclass
class EvalSample:
    """One conversation (one user) plus its questions."""

    sample_id: str
    sessions: list[EvalSession]
    questions: list[EvalQuestion]

    def iter_messages(self) -> Iterator[EvalMessage]:
        for session in self.sessions:
            yield from session.messages


class EvalDataset(ABC):
    """Base class for evaluation datasets."""

    name: str = "dataset"

    @abstractmethod
    def load(self) -> Iterator[EvalSample]:
        """Yield evaluation samples."""


def _read_json(path: str | Path) -> Any:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {p}. Download the official benchmark JSON and "
            f"pass its path via --path."
        )
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# LOCOMO numeric category codes -> human-readable buckets used in reports.
#
# WARNING: the integer->label encoding is NOT stable across LOCOMO releases and
# differs between published evaluation scripts. This default follows the encoding
# used by the widely-cited Mem0 eval. If your file uses a different schema, pass a
# ``category_map`` to ``LocomoDataset`` (or {} to keep the raw integer as a label).
# Per-category report rows are only meaningful if this matches your source file;
# overall metrics are unaffected by the mapping.
_LOCOMO_CATEGORY = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


class LocomoDataset(EvalDataset):
    """Loader for the LOCOMO benchmark JSON.

    Expected shape (per the public LOCOMO release): a list of items, each with a
    ``conversation`` object holding ``session_1``, ``session_2``, ... arrays of
    ``{"speaker", "text", "dia_id"}`` turns, and a ``qa`` array of
    ``{"question", "answer", "evidence", "category"}``.

    Args:
        path: local JSON path.
        category_map: optional override for the numeric-category encoding. Defaults
            to :data:`_LOCOMO_CATEGORY`; pass ``{}`` to label questions by their raw
            integer code instead of guessing.
    """

    name = "locomo"

    def __init__(
        self, path: str | Path, category_map: Optional[dict[int, str]] = None
    ) -> None:
        self.path = path
        self.category_map = _LOCOMO_CATEGORY if category_map is None else category_map

    def load(self) -> Iterator[EvalSample]:
        raw = _read_json(self.path)
        items = raw if isinstance(raw, list) else raw.get("data", [])
        for idx, item in enumerate(items):
            conv = item.get("conversation", {})
            sessions: list[EvalSession] = []
            for key in sorted(
                (k for k in conv if k.startswith("session_") and isinstance(conv[k], list)),
                key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
            ):
                turns = conv[key]
                messages: list[EvalMessage] = []
                for t_idx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    text = turn.get("text") or turn.get("clean_text") or ""
                    if not text:
                        continue
                    messages.append(
                        EvalMessage(
                            speaker=turn.get("speaker", "unknown"),
                            text=text,
                            msg_id=str(turn.get("dia_id") or f"{key}:{t_idx}"),
                            timestamp=turn.get("timestamp") or conv.get(f"{key}_date_time"),
                        )
                    )
                if messages:
                    sessions.append(EvalSession(session_id=key, messages=messages))

            questions: list[EvalQuestion] = []
            for q_idx, qa in enumerate(item.get("qa", [])):
                if not isinstance(qa, dict) or "question" not in qa:
                    continue
                cat = qa.get("category")
                category = self.category_map.get(cat, str(cat)) if cat is not None else "unknown"
                evidence = qa.get("evidence") or []
                if isinstance(evidence, str):
                    evidence = [evidence]
                questions.append(
                    EvalQuestion(
                        question_id=f"{idx}:{q_idx}",
                        question=str(qa["question"]),
                        answer=str(qa.get("answer", "")),
                        category=category,
                        evidence=[str(e) for e in evidence],
                    )
                )

            if sessions and questions:
                yield EvalSample(
                    sample_id=str(item.get("sample_id", idx)),
                    sessions=sessions,
                    questions=questions,
                )


class LongMemEvalDataset(EvalDataset):
    """Loader for the LongMemEval benchmark JSON.

    Expected shape: a list of items, each with ``question_id``, ``question``,
    ``answer``, ``question_type``, and ``haystack_sessions`` (a list of sessions,
    each a list of ``{"role", "content"}`` turns). ``answer_session_ids`` /
    per-turn ``has_answer`` flags, when present, populate evidence.
    """

    name = "longmemeval"

    def __init__(self, path: str | Path) -> None:
        self.path = path

    def load(self) -> Iterator[EvalSample]:
        raw = _read_json(self.path)
        items = raw if isinstance(raw, list) else raw.get("data", [])
        for idx, item in enumerate(items):
            qid = str(item.get("question_id", idx))
            haystack = item.get("haystack_sessions", [])
            session_ids = item.get("haystack_session_ids") or [
                f"sess_{i}" for i in range(len(haystack))
            ]
            answer_sessions = set(str(s) for s in (item.get("answer_session_ids") or []))

            sessions: list[EvalSession] = []
            evidence: list[str] = []
            for s_idx, turns in enumerate(haystack):
                sid = str(session_ids[s_idx]) if s_idx < len(session_ids) else f"sess_{s_idx}"
                messages: list[EvalMessage] = []
                for t_idx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    content = turn.get("content") or turn.get("text") or ""
                    if not content:
                        continue
                    msg_id = f"{sid}:{t_idx}"
                    messages.append(
                        EvalMessage(
                            speaker=turn.get("role", "user"),
                            text=content,
                            msg_id=msg_id,
                            timestamp=turn.get("timestamp") or item.get("timestamp"),
                        )
                    )
                    if turn.get("has_answer") or sid in answer_sessions:
                        evidence.append(msg_id)
                if messages:
                    sessions.append(EvalSession(session_id=sid, messages=messages))

            question = EvalQuestion(
                question_id=qid,
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                category=str(item.get("question_type", "unknown")),
                evidence=evidence,
            )
            if sessions and question.question:
                yield EvalSample(sample_id=qid, sessions=sessions, questions=[question])


class SyntheticDataset(EvalDataset):
    """Tiny, fully offline dataset so the harness runs with zero setup.

    Not a benchmark score -- a smoke test and a worked example of the data model.
    """

    name = "synthetic"

    def load(self) -> Iterator[EvalSample]:
        sessions = [
            EvalSession(
                session_id="s1",
                messages=[
                    EvalMessage("alice", "I just moved to Berlin for a new job at a fintech.", "s1:0"),
                    EvalMessage("alice", "I'm allergic to peanuts, so I avoid Thai food.", "s1:1"),
                    EvalMessage("alice", "My sister Maria is visiting me next month.", "s1:2"),
                ],
            ),
            EvalSession(
                session_id="s2",
                messages=[
                    EvalMessage("alice", "The fintech job is going well; I lead the payments team.", "s2:0"),
                    EvalMessage("alice", "I adopted a dog named Pixel last weekend.", "s2:1"),
                    EvalMessage("alice", "Maria arrived and we visited the Brandenburg Gate.", "s2:2"),
                ],
            ),
        ]
        questions = [
            EvalQuestion("q1", "Which city does Alice live in?", "Berlin", "single_hop", ["s1:0"]),
            EvalQuestion("q2", "What is Alice allergic to?", "Peanuts", "single_hop", ["s1:1"]),
            EvalQuestion("q3", "What is the name of Alice's dog?", "Pixel", "single_hop", ["s2:1"]),
            EvalQuestion(
                "q4",
                "Who visited Alice and what landmark did they see together?",
                "Her sister Maria; they visited the Brandenburg Gate.",
                "multi_hop",
                ["s1:2", "s2:2"],
            ),
        ]
        yield EvalSample(sample_id="alice", sessions=sessions, questions=questions)


def load_dataset(name: str, path: Optional[str] = None) -> EvalDataset:
    """Factory: build a dataset by name.

    Args:
        name: one of ``locomo``, ``longmemeval``, ``synthetic``.
        path: local JSON path (required for ``locomo`` and ``longmemeval``).
    """
    key = name.lower()
    if key == "synthetic":
        return SyntheticDataset()
    if path is None:
        raise ValueError(f"Dataset '{name}' requires --path to its JSON file.")
    if key == "locomo":
        return LocomoDataset(path)
    if key == "longmemeval":
        return LongMemEvalDataset(path)
    raise ValueError(f"Unknown dataset '{name}'. Choose: locomo, longmemeval, synthetic.")
