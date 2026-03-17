"""Tests for the optional policy engine. Run: python -m pytest contrib/test_policy_engine.py"""

from contrib.policy_engine import decide, Result


def test_crash_discarded():
    c = Result(val_bpb=0.0, complexity=0, status="crash")
    b = Result(val_bpb=1.0, complexity=50)
    assert decide(c, b).action == "discard"


def test_timeout_discarded():
    c = Result(val_bpb=0.0, complexity=0, status="timeout")
    b = Result(val_bpb=1.0, complexity=50)
    assert decide(c, b).action == "discard"


def test_significant_improvement_kept():
    c = Result(val_bpb=0.990, complexity=60)
    b = Result(val_bpb=1.000, complexity=50)
    assert decide(c, b).action == "keep"


def test_worse_val_bpb_discarded():
    c = Result(val_bpb=1.010, complexity=50)
    b = Result(val_bpb=1.000, complexity=50)
    assert decide(c, b).action == "discard"


def test_marginal_improvement_higher_complexity_discarded():
    c = Result(val_bpb=0.9995, complexity=60)
    b = Result(val_bpb=1.000, complexity=50)
    assert decide(c, b).action == "discard"


def test_comparable_lower_complexity_kept():
    c = Result(val_bpb=1.000, complexity=40)
    b = Result(val_bpb=1.000, complexity=50)
    assert decide(c, b).action == "keep"


def test_comparable_same_complexity_kept():
    c = Result(val_bpb=1.000, complexity=50)
    b = Result(val_bpb=1.000, complexity=50)
    assert decide(c, b).action == "keep"


def test_comparable_higher_complexity_discarded():
    c = Result(val_bpb=1.000, complexity=60)
    b = Result(val_bpb=1.000, complexity=50)
    assert decide(c, b).action == "discard"


def test_identical_results_kept():
    r = Result(val_bpb=1.000, complexity=50)
    assert decide(r, r).action == "keep"
