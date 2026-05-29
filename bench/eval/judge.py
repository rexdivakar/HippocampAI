"""LLM-as-judge for end-to-end QA accuracy.

This is the methodology LOCOMO / LongMemEval report on: given the retrieved
context, answer the question, then have an LLM grade the answer against the gold
reference as correct/incorrect. The judge reuses HippocampAI's own provider
adapters (:class:`hippocampai.adapters.llm_base.BaseLLM`).
"""

from __future__ import annotations

from typing import Optional

from hippocampai.adapters.llm_base import BaseLLM

_ANSWER_SYSTEM = (
    "You are answering a question using ONLY the provided memory context. "
    "If the context does not contain the answer, reply exactly with 'NO ANSWER'. "
    "Be concise -- answer in a single short sentence."
)

_JUDGE_SYSTEM = (
    "You are a strict grader. Decide whether the PREDICTED answer is correct given "
    "the GOLD reference answer. Accept paraphrases, extra detail, and different "
    "wording as long as the core facts match the gold answer. Reply with exactly "
    "one word: CORRECT or INCORRECT."
)


class LLMJudge:
    """Answer questions from context and grade them against gold answers."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def answer(self, question: str, context: str) -> str:
        """Produce an answer to ``question`` using only ``context``."""
        prompt = f"Memory context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm.generate(
            prompt=prompt, system=_ANSWER_SYSTEM, max_tokens=256, temperature=0.0
        ).strip()

    def grade(self, question: str, gold: str, predicted: str) -> bool:
        """Return True if ``predicted`` is judged correct against ``gold``."""
        if not predicted or predicted.strip().upper() == "NO ANSWER":
            return False
        prompt = (
            f"Question: {question}\n"
            f"GOLD answer: {gold}\n"
            f"PREDICTED answer: {predicted}\n\n"
            "Is the predicted answer correct?"
        )
        verdict = self.llm.generate(
            prompt=prompt, system=_JUDGE_SYSTEM, max_tokens=8, temperature=0.0
        )
        return "CORRECT" in verdict.strip().upper().split()

    def answer_and_grade(
        self, question: str, gold: str, context: str
    ) -> tuple[str, bool]:
        """Convenience: answer then grade, returning ``(predicted, is_correct)``."""
        predicted = self.answer(question, context)
        return predicted, self.grade(question, gold, predicted)


def build_default_judge(llm: Optional[BaseLLM]) -> Optional[LLMJudge]:
    """Wrap an LLM in a judge, or return None when no LLM is configured."""
    return LLMJudge(llm) if llm is not None else None
