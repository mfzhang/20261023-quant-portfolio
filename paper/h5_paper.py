"""
H5 paper trading (rules fixed by the pre-registration in README, "H5. 대형주 시총가중 +
급등주 제외 + 변동성 타기팅"; code shared with run_kr_largecap_experiment.py).

Run after each session's close. It fetches the KRX snapshots up to the latest session,
replays the H5 rules from 2018 (so the buffer state and the volatility estimates match
the backtest) and writes

  paper/h5/snapshot_YYYYMMDD.csv  target portfolio for that session: names, weights at
                                  the last rebalance, drifted weights, exposure, and the
                                  resulting share of equity per name (CASH row = 1 - exposure,
                                  negative when borrowing)
  paper/h5/track.csv              realized daily returns since PAPER_START (H5, B1, KODEX 200
                                  price), rebuilt from the same data on every run

A quarter-end signal is used once the next session exists (trade at that close), and
a month-end exposure change likewise applies from the next session's close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_kr_largecap_experiment as h5
import run_kr_value_experiment as base
from kr.data import krx_daily

OUT = ROOT / "paper" / "h5"
PAPER_START = "2026-10-15"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = h5.load(None)
    days, panel, exec_day = s["days"], s["panel"], s["exec_day"]
    latest = days[-1]
    b1 = base.run_book(s["b1"], exec_day, panel["ret"], s["div_daily"])["net"][s["live"]]
    v = h5.vol_target(b1, s["ann"])

    sig = s["sigs"][-1]
    sizes = pd.Series(s["b1"][sig])
    target = sizes / sizes.sum()
    held = panel["ret"].loc[exec_day[sig]:latest, target.index].iloc[1:]
    drifted = target * (1.0 + held.fillna(0.0)).prod()
    drifted = drifted / drifted.sum()
    exposure = float(v["exposure_end"].iloc[-1])
    snap = pd.DataFrame({"date": latest.date(), "signal_date": sig.date(), "ticker": target.index,
                         "name": panel["name"].loc[sig, target.index].to_numpy(),
                         "target_weight": target.to_numpy(), "current_weight": drifted.to_numpy(),
                         "exposure": exposure, "equity_share": (exposure * drifted).to_numpy()})
    snap = snap.sort_values("current_weight", ascending=False)
    cash = {"date": latest.date(), "signal_date": sig.date(), "ticker": "CASH", "name": "현금(음수 = 차입)",
            "target_weight": 0.0, "current_weight": 0.0, "exposure": exposure, "equity_share": 1.0 - exposure}
    snap = pd.concat([snap, pd.DataFrame([cash])], ignore_index=True)
    path = OUT / f"snapshot_{latest:%Y%m%d}.csv"
    snap.to_csv(path, index=False, encoding="utf-8-sig")

    etf = krx_daily.benchmark_prices(base.PRICE_START)["종가"].reindex(days).ffill().pct_change().reindex(v.index)
    m = v.index >= pd.Timestamp(PAPER_START)
    track = pd.DataFrame({"h5_return": v["net"][m], "exposure": v["exposure"][m], "b1_return": b1[m],
                          "kodex200_price_return": etf[m]})
    for c in ("h5", "b1", "kodex200_price"):
        track[f"{c}_cumulative"] = (1.0 + track[f"{c}_return"]).cumprod() - 1.0
    track.index.name = "date"
    track.to_csv(OUT / "track.csv", encoding="utf-8-sig")

    print(f"H5 paper snapshot {latest.date()} (rebalance signal {sig.date()}): {len(target)} names, "
          f"exposure {exposure:.2f} (target set at the last month end {float(v['target'].iloc[-1]):.2f})")
    print(snap.head(10)[["ticker", "name", "current_weight", "equity_share"]].to_string(index=False))
    if len(track):
        print(f"Since {PAPER_START}: H5 {track['h5_cumulative'].iloc[-1]:+.2%}, "
              f"KODEX 200 (price) {track['kodex200_price_cumulative'].iloc[-1]:+.2%} over {len(track)} sessions")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
