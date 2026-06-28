"""Tests for the cached data catalog.

The CSV-fallback path is tested unconditionally; the Parquet path is gated on
``pyarrow`` being installed. Cache-hit behavior is verified by asserting the
loader is not called a second time.
"""

from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.catalog import DataCatalog


def _sample_frame() -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=5, freq="D", name="Datetime")
    return pd.DataFrame(
        {"open": [1, 2, 3, 4, 5], "close": [1.5, 2.5, 3.5, 4.5, 5.5]}, index=idx
    )


def test_put_then_get_roundtrip_csv(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="csv")
    df = _sample_frame()
    cat.put(df, source="test", symbol="AAA", timeframe="D1")
    loaded = cat.get(source="test", symbol="AAA", timeframe="D1")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_has_and_metadata(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="csv")
    assert not cat.has("test", "AAA", "D1")
    cat.put(_sample_frame(), source="test", symbol="AAA", timeframe="D1")
    assert cat.has("test", "AAA", "D1")
    meta = cat.metadata("test", "AAA", "D1")
    assert meta["rows"] == 5
    assert meta["source"] == "test"
    assert "fetched_at" in meta


def test_fetch_hits_cache_and_skips_loader(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="csv")
    calls = {"n": 0}

    def loader() -> pd.DataFrame:
        calls["n"] += 1
        return _sample_frame()

    first = cat.fetch(loader, source="yf", symbol="AAA", timeframe="D1")
    second = cat.fetch(loader, source="yf", symbol="AAA", timeframe="D1")
    # The loader must run only once; the second call is served from cache.
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_fetch_force_bypasses_cache(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="csv")
    calls = {"n": 0}

    def loader() -> pd.DataFrame:
        calls["n"] += 1
        return _sample_frame()

    cat.fetch(loader, source="yf", symbol="AAA", timeframe="D1")
    cat.fetch(loader, source="yf", symbol="AAA", timeframe="D1", force=True)
    assert calls["n"] == 2


def test_fetch_respects_max_age(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="csv")
    calls = {"n": 0}

    def loader() -> pd.DataFrame:
        calls["n"] += 1
        return _sample_frame()

    cat.fetch(loader, source="yf", symbol="AAA", timeframe="D1")
    # max_age_days=0 means any cached copy is already stale -> reload.
    cat.fetch(loader, source="yf", symbol="AAA", timeframe="D1", max_age_days=0)
    assert calls["n"] == 2


def test_list_datasets(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="csv")
    cat.put(_sample_frame(), source="test", symbol="AAA", timeframe="D1")
    cat.put(_sample_frame(), source="test", symbol="BBB", timeframe="D1")
    datasets = cat.list_datasets()
    symbols = {d["symbol"] for d in datasets}
    assert symbols == {"AAA", "BBB"}


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    cat = DataCatalog(base_dir=str(tmp_path), fmt="parquet")
    assert cat.fmt == "parquet"
    df = _sample_frame()
    cat.put(df, source="test", symbol="AAA", timeframe="D1")
    loaded = cat.get(source="test", symbol="AAA", timeframe="D1")
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_auto_format_falls_back_to_csv_without_pyarrow(tmp_path):
    cat = DataCatalog(base_dir=str(tmp_path), fmt="auto")
    # fmt resolves to parquet only when pyarrow is importable.
    from bbstrader.btengine.catalog import has_pyarrow

    expected = "parquet" if has_pyarrow() else "csv"
    assert cat.fmt == expected
