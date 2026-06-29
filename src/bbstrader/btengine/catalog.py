"""A unified, cached data catalog over the existing data handlers.

Today the download handlers re-fetch on every run. The catalog adds a real
local store with cache-hit semantics and point-in-time metadata, so re-runs are
instant and offline-capable. Data is persisted as **Parquet** when ``pyarrow``
is available and transparently falls back to **CSV** otherwise, so a lean
install (without the ``[catalog]`` extra) keeps working.

The store is intentionally decoupled from the handler plumbing: ``fetch`` takes
any zero-argument loader that returns a normalized OHLCV ``DataFrame``, which
makes it trivial to back with ``YFDataHandler``, a broker API, or a test stub.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from bbstrader.config import BBSTRADER_DIR

__all__ = ["DataCatalog", "has_pyarrow"]


def has_pyarrow() -> bool:
    """Return True if a Parquet engine (pyarrow) is importable."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class DataCatalog:
    """A local OHLCV cache keyed by ``(source, symbol, timeframe)``.

    Args:
        base_dir: Root directory for the store. Defaults to
            ``~/.bbstrader/data/catalog``.
        fmt: ``"parquet"``, ``"csv"``, or ``"auto"`` (Parquet when pyarrow is
            installed, else CSV).
    """

    def __init__(self, base_dir: Optional[str] = None, fmt: str = "auto") -> None:
        """Initialise the catalog and ensure its base directory exists.

        Args:
            base_dir (Optional[str]): Root directory for the store. Defaults to
                ``~/.bbstrader/data/catalog``.
            fmt (str): One of ``"parquet"``, ``"csv"`` or ``"auto"`` (Parquet
                when pyarrow is installed, else CSV).

        Raises:
            ValueError: If ``fmt`` is not one of the accepted values.
        """
        if fmt not in ("auto", "parquet", "csv"):
            raise ValueError(f"fmt must be 'auto', 'parquet' or 'csv', got {fmt!r}.")
        self.base_dir = Path(base_dir or BBSTRADER_DIR / "data" / "catalog")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._fmt = fmt

    @property
    def fmt(self) -> str:
        """The effective storage format after resolving ``auto``."""
        if self._fmt == "parquet":
            return "parquet"
        if self._fmt == "csv":
            return "csv"
        return "parquet" if has_pyarrow() else "csv"

    def _key(self, source: str, symbol: str, timeframe: str) -> str:
        """Return the filesystem-safe cache key for a dataset.

        Args:
            source (str): The data source identifier (for example ``"yf"``).
            symbol (str): The instrument symbol; path separators are escaped.
            timeframe (str): The bar timeframe (for example ``"D1"``).

        Returns:
            str: A unique key combining the three fields.
        """
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        return f"{source}__{safe_symbol}__{timeframe}"

    def _data_path(self, source: str, symbol: str, timeframe: str) -> Path:
        """Return the on-disk path of a dataset's data file.

        Args:
            source (str): The data source identifier.
            symbol (str): The instrument symbol.
            timeframe (str): The bar timeframe.

        Returns:
            Path: The Parquet or CSV path for the dataset.
        """
        ext = "parquet" if self.fmt == "parquet" else "csv"
        return self.base_dir / f"{self._key(source, symbol, timeframe)}.{ext}"

    def _meta_path(self, source: str, symbol: str, timeframe: str) -> Path:
        """Return the on-disk path of a dataset's metadata file.

        Args:
            source (str): The data source identifier.
            symbol (str): The instrument symbol.
            timeframe (str): The bar timeframe.

        Returns:
            Path: The ``.meta.json`` path for the dataset.
        """
        return self.base_dir / f"{self._key(source, symbol, timeframe)}.meta.json"

    def has(self, source: str, symbol: str, timeframe: str) -> bool:
        """Return True if a cached dataset exists for the key."""
        return self._data_path(source, symbol, timeframe).exists()

    def metadata(
        self, source: str, symbol: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Return the stored metadata for the key, or None if absent."""
        meta_path = self._meta_path(source, symbol, timeframe)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text())

    def is_fresh(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        max_age_days: Optional[float] = None,
    ) -> bool:
        """Return True if the cached dataset exists and is within ``max_age_days``.

        A ``max_age_days`` of None means "never expires" (any cached copy is
        fresh); a value of 0 (or negative) means the cache is always stale,
        independent of clock resolution.
        """
        if not self.has(source, symbol, timeframe):
            return False
        if max_age_days is None:
            return True
        # A zero/negative budget means "always reload". Handle it explicitly so
        # the result does not hinge on sub-millisecond timestamp resolution
        # (Windows clocks can read an age of exactly 0 for a just-written file).
        if max_age_days <= 0:
            return False
        meta = self.metadata(source, symbol, timeframe)
        if not meta or "fetched_at" not in meta:
            return False
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        age = datetime.now(timezone.utc) - fetched_at
        return age.total_seconds() <= max_age_days * 86400.0

    def get(self, source: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load a cached dataset, or None if it is not present."""
        path = self._data_path(source, symbol, timeframe)
        if not path.exists():
            return None
        if self.fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def put(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        timeframe: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist ``df`` for the key and write a metadata sidecar."""
        path = self._data_path(source, symbol, timeframe)
        if self.fmt == "parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path)
        index = df.index
        meta: Dict[str, Any] = {
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": int(len(df)),
            "format": self.fmt,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "start": str(index.min()) if len(index) else None,
            "end": str(index.max()) if len(index) else None,
        }
        if extra_meta:
            meta.update(extra_meta)
        self._meta_path(source, symbol, timeframe).write_text(
            json.dumps(meta, indent=2)
        )
        return path

    def fetch(
        self,
        loader: Callable[[], pd.DataFrame],
        source: str,
        symbol: str,
        timeframe: str = "D1",
        max_age_days: Optional[float] = None,
        force: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Return cached data if fresh, otherwise call ``loader`` and cache it.

        Args:
            loader: Zero-argument callable returning a normalized OHLCV DataFrame
                (only called on a cache miss or when ``force`` is set).
            source: Logical source name (e.g. ``"yfinance"``).
            symbol: Instrument symbol.
            timeframe: Bar timeframe/period label used in the cache key.
            max_age_days: Maximum acceptable cache age; None means never expires.
            force: Bypass the cache and always reload.
        """
        if not force and self.is_fresh(source, symbol, timeframe, max_age_days):
            cached = self.get(source, symbol, timeframe)
            if cached is not None:
                return cached
        df = loader()
        self.put(df, source, symbol, timeframe, extra_meta=extra_meta)
        return df

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return metadata for every dataset currently in the store."""
        out: List[Dict[str, Any]] = []
        for meta_file in sorted(self.base_dir.glob("*.meta.json")):
            try:
                out.append(json.loads(meta_file.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out
