"""Walk-forward backtest harness for cross-sectional ranking signals.

Daily, long-only-advisory, rank-and-hold: a signal is a date × symbol score frame on
rebalance dates; the engine forms the top bucket, holds to the next rebalance, and
grades ranking quality (rank IC, decile spread) and the portfolio (net of costs)
against the equal-weight universe, SMH and SPY — the same benchmarks as
src/performance.py. Fill convention: signal at close t, fill at the NEXT close
(one-day lag, conservative, no look-ahead).
"""
