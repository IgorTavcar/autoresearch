"""
Optional deterministic keep/discard policy engine for autoresearch experiments.

Evaluates experiment results based on val_bpb improvement and complexity score.
This is a standalone utility — it does NOT modify the main training loop.
The agent can use it as a decision aid, or ignore it entirely.

From upstream PR #276 (Nick Mandal).

Usage:
    from contrib.policy_engine import decide, Result
    candidate = Result(val_bpb=0.990, complexity=50, status='ok')
    baseline = Result(val_bpb=0.995, complexity=48, status='ok')
    decision = decide(candidate, baseline)
    print(decision)  # Decision(action='keep', reason='significant val_bpb improvement')
"""

from typing import NamedTuple


class Result(NamedTuple):
    val_bpb: float
    complexity: float  # e.g. num_params_M, lines of code, or any proxy
    status: str = "ok"  # 'ok', 'crash', 'timeout'


class Decision(NamedTuple):
    action: str  # 'keep' or 'discard'
    reason: str


def decide(candidate: Result, baseline: Result, improvement_threshold: float = 0.001) -> Decision:
    """Deterministic keep/discard decision based on val_bpb and complexity."""
    # Immediate discard on crash or timeout
    if candidate.status in ("crash", "timeout"):
        return Decision("discard", f"candidate status: {candidate.status}")

    improvement = baseline.val_bpb - candidate.val_bpb

    # Rule 1: significant improvement → keep regardless of complexity
    if improvement > improvement_threshold:
        return Decision("keep", "significant val_bpb improvement")

    # Rule 2: marginal or no improvement with higher complexity → discard
    if improvement <= improvement_threshold and candidate.complexity > baseline.complexity:
        return Decision("discard", "marginal improvement with higher complexity")

    # Rule 3: comparable val_bpb — use simplicity as tiebreaker
    if abs(improvement) <= improvement_threshold:
        if candidate.complexity < baseline.complexity:
            return Decision("keep", "comparable val_bpb with lower complexity (simplification)")
        elif candidate.complexity == baseline.complexity:
            return Decision("keep", "comparable val_bpb with same complexity")
        else:
            return Decision("discard", "comparable val_bpb with higher complexity")

    # Rule 4: worse val_bpb
    if improvement < 0:
        return Decision("discard", "worse val_bpb")

    return Decision("keep", "default")
