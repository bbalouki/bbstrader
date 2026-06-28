"""Tests for the experiment/results store."""

import pandas as pd
import pytest

from bbstrader.btengine.experiment import ExperimentRecord, ExperimentStore


def test_save_and_load_roundtrip(tmp_path):
    store = ExperimentStore(root=tmp_path)
    rid = store.save(
        "trend",
        params={"fast": 10, "slow": 30},
        metrics={"sharpe": 1.2},
        run_id="trend-001",
        created_at="2020-01-01T00:00:00+00:00",
    )
    assert rid == "trend-001"
    rec = store.load("trend-001")
    assert isinstance(rec, ExperimentRecord)
    assert rec.params == {"fast": 10, "slow": 30}
    assert rec.metrics["sharpe"] == 1.2
    assert "python" in rec.environment


def test_equity_curve_persisted(tmp_path):
    store = ExperimentStore(root=tmp_path)
    curve = pd.DataFrame({"Total": [100.0, 101.0, 102.0]})
    store.save("x", {}, {"ret": 0.02}, equity_curve=curve, run_id="x-1")
    loaded = store.load_equity("x-1")
    assert loaded is not None
    assert list(loaded["Total"]) == [100.0, 101.0, 102.0]


def test_load_missing_raises(tmp_path):
    store = ExperimentStore(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")


def test_list_sorted_by_created_at(tmp_path):
    store = ExperimentStore(root=tmp_path)
    store.save("a", {}, {"s": 1}, run_id="a", created_at="2021-01-01T00:00:00+00:00")
    store.save("b", {}, {"s": 2}, run_id="b", created_at="2020-01-01T00:00:00+00:00")
    ids = [r.id for r in store.list()]
    assert ids == ["b", "a"]  # earliest first


def test_compare_leaderboard_sorted(tmp_path):
    store = ExperimentStore(root=tmp_path)
    store.save(
        "a",
        {"p": 1},
        {"sharpe": 0.5},
        run_id="a",
        created_at="2020-01-01T00:00:00+00:00",
    )
    store.save(
        "b",
        {"p": 2},
        {"sharpe": 1.5},
        run_id="b",
        created_at="2020-01-02T00:00:00+00:00",
    )
    store.save(
        "c",
        {"p": 3},
        {"sharpe": 1.0},
        run_id="c",
        created_at="2020-01-03T00:00:00+00:00",
    )
    board = store.compare(metric="sharpe")
    assert list(board["id"]) == ["b", "c", "a"]  # descending by sharpe
    assert "sharpe" in board.columns


def test_delete(tmp_path):
    store = ExperimentStore(root=tmp_path)
    store.save("a", {}, {"s": 1}, run_id="a")
    store.delete("a")
    assert store.list() == []
