"""A lightweight experiment/results store for reproducible research.

Persists each backtest/optimization run -- its parameters, metrics, equity
curve and environment -- to disk so runs can be reloaded, compared
leaderboard-style, and reproduced later. Metadata is JSON; the equity curve is
CSV. Defaults to ``~/.bbstrader/experiments`` but any root works.
"""

from __future__ import annotations

import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from bbstrader.config import BBSTRADER_DIR

__all__ = ["ExperimentRecord", "ExperimentStore"]


@dataclass
class ExperimentRecord:
    id: str
    name: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: str
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _environment() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class ExperimentStore:
    """Save, load, list and compare persisted experiment runs."""

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        self.root = Path(root) if root else BBSTRADER_DIR / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def save(
        self,
        name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        equity_curve: Optional[pd.DataFrame] = None,
        run_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Persist a run and return its id.

        ``run_id``/``created_at`` may be supplied for deterministic, idempotent
        writes (e.g. in tests); otherwise a uuid and the current UTC time are
        used.
        """
        run_id = run_id or f"{name}-{uuid.uuid4().hex[:8]}"
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        record = ExperimentRecord(
            id=run_id,
            name=name,
            params=params,
            metrics=metrics,
            created_at=created_at,
            environment=_environment(),
        )
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(
            json.dumps(record.to_dict(), indent=2, default=str)
        )
        if equity_curve is not None:
            equity_curve.to_csv(run_dir / "equity.csv")
        return run_id

    def load(self, run_id: str) -> ExperimentRecord:
        meta = self._run_dir(run_id) / "meta.json"
        if not meta.exists():
            raise FileNotFoundError(f"No experiment with id {run_id!r}.")
        data = json.loads(meta.read_text())
        return ExperimentRecord(**data)

    def load_equity(self, run_id: str) -> Optional[pd.DataFrame]:
        path = self._run_dir(run_id) / "equity.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0)

    def list(self) -> List[ExperimentRecord]:
        records = []
        for meta in self.root.glob("*/meta.json"):
            records.append(ExperimentRecord(**json.loads(meta.read_text())))
        return sorted(records, key=lambda r: r.created_at)

    def compare(
        self, metric: Optional[str] = None, ascending: bool = False
    ) -> pd.DataFrame:
        """Return a leaderboard DataFrame of all runs' metrics.

        Sorted by ``metric`` (descending by default) when provided.
        """
        rows = []
        for rec in self.list():
            row = {"id": rec.id, "name": rec.name, "created_at": rec.created_at}
            row.update(rec.metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        if metric and metric in df.columns:
            df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        return df

    def delete(self, run_id: str) -> None:
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            for child in run_dir.iterdir():
                child.unlink()
            run_dir.rmdir()
