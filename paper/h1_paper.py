"""
H1 paper trading (pre-registered in README, "H1 모의 운용", 2026-10-23): the Korean H1
rule of run_kr_value_experiment.py unchanged. Composite of E/P, B/M, CF/P, S/P, EBIT/EV
percentile ranks (geometric mean, >= 4 of 5) inside the eligible universe, top 20%
equal weight, quarterly. Benchmark: the eligible universe equal weight on the same
signal dates. Judged only on sessions after the 2026-10-23 close.

Run after each session's close (after paper/b0_paper.py). It writes

  paper/h1/snapshot_YYYYMMDD.csv   holdings after that session's close
  paper/h1/track.csv               realized daily returns after the start snapshot:
                                   H1 and eligible EW, price-only and dividend-accrued
                                   (net of costs), KODEX 200 close-to-close
  paper/h1/signals/metrics_YYYYMMDD.csv
                                   eligible universe and value metrics of each new
                                   quarter-end signal, built once with the filings
                                   received by that day

The 2026-09-30 signal is the backtest's own (output/kr_value/metrics_quarterly.csv).
A new signal is built on the first run after its quarter-end session: the DART bulk
files are refreshed, periods after 2026 2Q are parsed and appended to the backtest
statements, and receipt dates are re-read. A quarter whose bulk file is not yet
published on the signal date is used only through the receipt-date rule, as in the
backtest.
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
from kr.data import dart_bulk, dart_pit, krx_daily
from strategies.composite_value import composite_scores, quantile_sets

OUT = ROOT / "paper" / "h1"
SIG_DIR = OUT / "signals"
LIVE = base.OUT / "live"
START = pd.Timestamp("2026-10-23")
BASE_SIGNAL = pd.Timestamp("2026-09-30")
LAST_BULK = ("2026", "2Q")
KEYS = ["ticker", "consolidated", "item", "period_end", "report"]


def live_reports() -> pd.DataFrame:
    dart_bulk.download_all()
    long = pd.read_pickle(dart_pit.CACHE / "bulk_long.pkl")
    periods = sorted({tuple(p.name.split("_")[:2]) for p in dart_bulk.CACHE.glob("*.zip")})
    new = [p for p in periods if p > LAST_BULK]
    if new:
        parts = [dart_pit.parse_period(y, q) for y, q in new]
        long = pd.concat([long, *parts], ignore_index=True).drop_duplicates(KEYS, keep="last")
    print(f"  statements: backtest periods + {', '.join('_'.join(p) for p in new) or 'none'}", flush=True)
    rep = dart_pit.report_table(long, dart_pit.filing_dates())
    LIVE.mkdir(parents=True, exist_ok=True)
    rep.to_pickle(LIVE / "pit_reports_live.pkl")
    return rep


def build_signal(d: pd.Timestamp, panel: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    print(f"Building the {d.date()} signal", flush=True)
    rep = live_reports()
    _, counts, metrics = base.build_signals(panel, rep, days, sigs=[d])
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SIG_DIR / f"metrics_{d:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
    path = SIG_DIR / "counts.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, counts], ignore_index=True).to_csv(path, index=False)
    return metrics


def _read(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    m["ticker"] = m["ticker"].str.zfill(6)
    return m


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    s = lc.load(None)
    days, panel, exec_day = s["days"], s["panel"], s["exec_day"]
    latest = days[-1]
    if latest < START:
        raise SystemExit(f"latest session {latest.date()} is before the paper start {START.date()}")

    hist = _read(base.OUT / "metrics_quarterly.csv")
    metrics = {BASE_SIGNAL: hist[hist["date"] == BASE_SIGNAL]}
    for d in (d for d in s["sigs"] if d > BASE_SIGNAL):
        path = SIG_DIR / f"metrics_{d:%Y%m%d}.csv"
        metrics[d] = _read(path) if path.exists() else build_signal(d, panel, days)

    h1, ew, score = {}, {}, {}
    for d, m in metrics.items():
        g = m.set_index("ticker")
        score[d] = composite_scores(g, base.VALUE_METRICS, method="geometric", min_metrics=4)
        h1[d] = quantile_sets(score[d])[-1]
        ew[d] = set(g.index)

    ret = panel["ret"]
    nets = {(book, var): base.run_book(sets, exec_day, ret, dd)["net"]
            for book, sets in (("h1", h1), ("ew", ew))
            for var, dd in (("price", None), ("total", s["div_daily"]))}

    sig = max(d for d in metrics if exec_day[d] <= latest)
    names = [t for t in h1[sig] if t in ret.columns and pd.notna(ret.at[exec_day[sig], t])]
    held = ret.loc[exec_day[sig]:latest, names].iloc[1:]
    drifted = (1.0 + held.fillna(0.0)).prod()
    drifted = drifted / drifted.sum()
    g = metrics[sig].set_index("ticker")
    snap = pd.DataFrame({"date": latest.date(), "signal_date": sig.date(), "trade_date": exec_day[sig].date(),
                         "ticker": names, "name": panel["name"].loc[sig, names].to_numpy(),
                         "composite": score[sig].reindex(names).to_numpy(),
                         "report": g["report"].reindex(names).to_numpy(),
                         "target_weight": 1.0 / len(names), "current_weight": drifted.reindex(names).to_numpy()})
    snap = snap.sort_values("current_weight", ascending=False)
    path = OUT / f"snapshot_{latest:%Y%m%d}.csv"
    snap.to_csv(path, index=False, encoding="utf-8-sig")

    etf = krx_daily.benchmark_prices(base.PRICE_START)["종가"].reindex(days).ffill().pct_change()
    m = days > START
    track = pd.DataFrame({f"{book}_{var}_return": r[m] for (book, var), r in nets.items()})
    track["kodex200_return"] = etf[m]
    for c in [c[: -len("_return")] for c in track.columns]:
        track[f"{c}_cumulative"] = (1.0 + track[f"{c}_return"]).cumprod() - 1.0
    track["excess_total_cumulative"] = track["h1_total_cumulative"] - track["ew_total_cumulative"]
    track.index.name = "date"
    track.to_csv(OUT / "track.csv", encoding="utf-8-sig")

    print(f"H1 paper snapshot {latest.date()}: {len(names)} names (equal weight), "
          f"signal {sig.date()} traded {exec_day[sig].date()}")
    print(snap.head(10)[["ticker", "name", "composite", "current_weight"]].to_string(index=False))
    if len(track):
        last = track.iloc[-1]
        print(f"Since the {START.date()} close ({len(track)} sessions, dividend-accrued): "
              f"H1 {last['h1_total_cumulative']:+.2%}, eligible EW {last['ew_total_cumulative']:+.2%}, "
              f"KODEX 200 {last['kodex200_cumulative']:+.2%} (price)")
    print(f"Saved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
