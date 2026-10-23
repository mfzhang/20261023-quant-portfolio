"""Semi-annual large-cap screen: liabilities/equity < 1 and current assets/assets >= 40%.

Universe is the point-in-time top 100 common stocks (same exclusions as B0).
Banks, insurers and brokers are dropped because the ratios are not comparable.
Weights are common-share market cap. Signals are the last session of June and
December, traded at the next close. Recorded in README section 24.
Run after seeing section 23. Not a pre-registered hypothesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_kr_largecap_experiment as lc
import run_kr_value_experiment as base

OUT = base.OUT
TOP_N = 100
CA_MIN = 0.40
SPLIT = base.SPLIT


def semi_dates(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = days.to_series()
    ends = s.groupby(days.to_period("M")).max()
    ends = ends[ends.index.month.isin([6, 12]) & (ends >= pd.Timestamp("2017-12-01"))]
    return [d for d in ends if d < days[-1]]


def main() -> None:
    s = lc.load(base.cfg.END)
    panel, days, ann = s["panel"], s["days"], s["ann"]
    sigs = semi_dates(days)
    exec_day = {d: days[days.get_loc(d) + 1] for d in sigs}
    rep = pd.read_pickle(OUT / "pit_reports.pkl")
    items = pd.read_pickle(OUT / "quality_items.pkl")
    rep = rep.copy()
    rep["ticker"] = rep["ticker"].astype(str).str.zfill(6)
    rep = rep.merge(items, on=["ticker", "consolidated", "period_end", "report"], how="left")

    screened, base_book, rows = {}, {}, []
    for d in sigs:
        mc = lc.universe_mcap(panel, d).head(TOP_N)
        known = rep[rep["avail"] <= d]
        latest = known.sort_values(["ticker", "period_end", "avail"]).groupby("ticker").tail(1)
        latest = latest[(d - latest["period_end"]).dt.days <= 200].drop_duplicates("ticker").set_index("ticker")
        names = latest.reindex(mc.index)
        fin = [t for t in mc.index if t in names.index and pd.notna(names.at[t, "industry"])
               and base.is_financial(names.at[t, "industry"], names.at[t, "name"], "")]
        pool = mc.drop(labels=[t for t in fin if t in mc.index], errors="ignore")
        cur = names.reindex(pool.index)
        ca = cur["current_assets"] / cur["assets"]
        debt = cur["liabilities"] / cur["equity_parent"]
        ok = (cur["assets"] > 0) & (cur["equity_parent"] > 0) & (ca >= CA_MIN) & (debt < 1.0)
        picked = pool[ok.fillna(False)]
        screened[d] = picked.to_dict()
        base_book[d] = pool.to_dict()
        w = picked / picked.sum() if len(picked) else picked
        top = w.idxmax() if len(w) else ""
        rows.append({
            "date": d, "n_top": len(mc), "n_financial": len(fin), "n_scored": int(ca.notna().sum()),
            "n_pass": len(picked), "top": panel["name"].loc[d].get(top, ""),
            "top_weight": float(w.max()) if len(w) else 0.0,
            "median_ca": float(ca[ok.fillna(False)].median()) if ok.fillna(False).any() else np.nan,
            "median_debt": float(debt[ok.fillna(False)].median()) if ok.fillna(False).any() else np.nan,
        })
    counts = pd.DataFrame(rows)

    etf = lc.krx_daily.benchmark_prices(base.PRICE_START, base.cfg.END)["종가"].reindex(days).ffill()
    kodex = {
        "price": etf.pct_change().fillna(0.0),
        "total": pd.read_csv(OUT / "daily_returns.csv", index_col=0, parse_dates=True)["KODEX 200"].reindex(days),
    }
    live = days > exec_day[sigs[0]]
    table_rows, series = [], {}
    for var, dd in (("total", s["div_daily"]), ("price", None)):
        ran = {}
        for label, sets in (("screen", screened), ("top 100", base_book)):
            b = base.run_book(sets, exec_day, panel["ret"], dd)
            ran[label] = (b["net"][live], float(b["turnover"][live].mean() * ann))
        ran["KODEX 200"] = (kodex[var][live], 0.0)
        for label, (r, turn) in ran.items():
            series[(var, label)] = r
            table_rows.append(lc.stats(var, label, r, ann, turn, None))
    table = pd.DataFrame(table_rows)
    spreads = []
    for var in ("total", "price"):
        for bench in ("top 100", "KODEX 200"):
            d = series[(var, "screen")] - series[(var, bench)]
            rec = {"variant": var, "versus": bench}
            for period, w in lc.windows(d).items():
                rec[f"{period}_excess"] = float(w.mean() * ann)
                rec[f"{period}_t"] = base._tstat(w)
            spreads.append(rec)
    table.to_csv(OUT / "balance_screen_comparison.csv", index=False)
    pd.DataFrame(spreads).to_csv(OUT / "balance_screen_spreads.csv", index=False)
    counts.to_csv(OUT / "balance_screen_semiannual.csv", index=False)
    fmt = lambda x: f"{x:.4f}"
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== excess ===")
    print(pd.DataFrame(spreads).to_string(index=False, float_format=fmt))
    print("\n=== semi-annual ===")
    print(counts.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
