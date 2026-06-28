# Benchmarks

Reproducible timings for the `bbstrader` event-driven backtest engine. These
back the performance positioning in the main README with numbers anyone can
regenerate, and make engine regressions visible.

## Run

```bash
python benchmarks/run_benchmarks.py --bars 1000 10000 100000
python benchmarks/run_benchmarks.py --bars 10000 --out results.json
```

The script generates synthetic OHLCV data (no network, no extra dependency),
then reports for each dataset size:

- **backtest (s)** — wall time for one full event-driven `run_backtest`.
- **bars/sec** — event-loop throughput (mark-to-market + portfolio accounting).
- **2x replay (s)** — time to replay the data feed twice via `DataHandler.reset()`,
  demonstrating the replayable columnar feed that powers parameter sweeps.

The engine is event-driven by design (fidelity first). A fully vectorized
research fast-path for screening millions of bars per second is on the
[roadmap](../FEATURES.md#2-ultra-fast-hybrid-backtesting-engine-flagship); when
it lands, its numbers will be added here alongside the event-driven baseline.
