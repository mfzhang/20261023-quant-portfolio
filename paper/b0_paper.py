"""
B0 paper trading: top 50 KOSPI + KOSDAQ common stocks by market cap, buffer 40 / 60,
cap-weighted, quarterly rebalance, exposure 1.0 (no spike filter, no vol targeting).
Rules and costs are B0 of run_kr_largecap_experiment.py (same code).

Run after each session's close. It fetches the KRX snapshots up to the latest session,
replays B0 from 2018 (so the buffer state matches the backtest) and writes

  paper/b0/snapshot_YYYYMMDD.csv  portfolio after that session's close: names, weights at
                                  the last rebalance and drifted weights
  paper/b0/track.csv              realized daily returns after the start snapshot
                                  (2026-10-15 close): B0 price-only and dividend-accrued
                                  (both net of costs) and KODEX 200 close-to-close

A quarter-end signal (last session of Mar / Jun / Sep / Dec) is traded at the next
session's close, so its snapshot appears from that session on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_kr_largecap_experiment as lc
import run_kr_value_experiment as base
from kr.data import krx_daily

OUT = ROOT / "paper" / "b0"
START = pd.Timestamp("2026-10-15")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    s = lc.load(None)
    days, panel, exec_day = s["days"], s["panel"], s["exec_day"]
    latest = days[-1]
    if latest < START:
        raise SystemExit(f"latest session {latest.date()} is before the paper start {START.date()}")
    nets = {var: base.run_book(s["b0"], exec_day, panel["ret"], dd)["net"][s["live"]]
            for var, dd in (("price", None), ("total", s["div_daily"]))}

    sig = max(d for d in s["sigs"] if exec_day[d] <= latest)
    sizes = pd.Series(s["b0"][sig])
    target = sizes / sizes.sum()
    held = panel["ret"].loc[exec_day[sig]:latest, target.index].iloc[1:]
    drifted = target * (1.0 + held.fillna(0.0)).prod()
    drifted = drifted / drifted.sum()
    snap = pd.DataFrame({"date": latest.date(), "signal_date": sig.date(), "trade_date": exec_day[sig].date(),
                         "ticker": target.index, "name": panel["name"].loc[sig, target.index].to_numpy(),
                         "target_weight": target.to_numpy(), "current_weight": drifted.to_numpy()})
    snap = snap.sort_values("current_weight", ascending=False)
    path = OUT / f"snapshot_{latest:%Y%m%d}.csv"
    snap.to_csv(path, index=False, encoding="utf-8-sig")

    etf = krx_daily.benchmark_prices(base.PRICE_START)["종가"].reindex(days).ffill().pct_change()
    m = days[s["live"]] > START
    track = pd.DataFrame({"b0_price_return": nets["price"][m], "b0_total_return": nets["total"][m],
                          "kodex200_return": etf.reindex(nets["price"].index)[m]})
    for c in ("b0_price", "b0_total", "kodex200"):
        track[f"{c}_cumulative"] = (1.0 + track[f"{c}_return"]).cumprod() - 1.0
    track.index.name = "date"
    track.to_csv(OUT / "track.csv", encoding="utf-8-sig")

    print(f"B0 paper snapshot {latest.date()}: {len(target)} names, rebalance signal {sig.date()} "
          f"traded {exec_day[sig].date()}")
    print(snap.head(10)[["ticker", "name", "current_weight"]].to_string(index=False))
    if len(track):
        print(f"Since the {START.date()} close: B0 {track['b0_price_cumulative'].iloc[-1]:+.2%} (price), "
              f"KODEX 200 {track['kodex200_cumulative'].iloc[-1]:+.2%} over {len(track)} sessions")
    print(f"Saved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
