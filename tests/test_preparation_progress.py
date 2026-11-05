"""Progress must leave data order and random sampling untouched."""
import importlib
import random

import pytest

module = importlib.import_module("diffjev.progress")


def test_progress_counts_completed_work_and_preserves_rng(monkeypatch, capsys):
    ticks = iter([0., 1., 16., 17., 18.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    rows = [{"id": i} for i in range(3)]
    before = random.getstate()
    actual = list(module.progress(rows, "fixture"))
    assert all(a is b for a, b in zip(actual, rows))
    assert random.getstate() == before
    output = capsys.readouterr().out
    assert "fixture: 0/3" in output and "fixture: 2/3" in output
    assert "fixture: 3/3; elapsed=18.0s" in output
    assert output.rstrip().endswith("complete")


def test_progress_does_not_hide_failure_or_claim_completion(capsys):
    def failed_rows():
        yield 1
        raise ValueError("fixture failure")
    with pytest.raises(ValueError, match="fixture failure"):
        list(module.progress(failed_rows(), "fixture"))
    assert "complete" not in capsys.readouterr().out
