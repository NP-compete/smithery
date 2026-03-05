"""Multi-step plan evaluation using LLM-as-judge."""

from __future__ import annotations

from typing import Any


def plan_success_rate(
    plans: list[list[dict[str, Any]]],
    ground_truth: list[list[dict[str, Any]]],
    judge_model: str = "gpt-4o-mini",
) -> float:
    """Evaluate multi-step tool-call plans against ground truth.

    Uses an LLM judge to assess whether the plan achieves the stated goal,
    accounting for valid alternative orderings.

    Args:
        plans: Model-generated multi-step plans.
        ground_truth: Reference plans.
        judge_model: Model to use as the judge.

    Returns:
        Success rate as a float in [0, 1].
    """
    raise NotImplementedError
