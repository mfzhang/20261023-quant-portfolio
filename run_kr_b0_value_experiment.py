"""
H6: US composite value screen inside the Korean large-cap basket B0.

Pre-registered in README ("H6. 기본 바스켓 안의 복합 가치", 2026-10-20) before this run.
B0 names (top 50, buffer 40/60), financials and names with fewer than 4 yields dropped,
geometric mean of percentile ranks computed inside that set, top quintile kept,
weighted by the same common-share market cap as B0. One run, sample ends cfg.END.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_kr_largecap_experiment as lc
import run_kr_value_experiment as base
from strategies.composite_value import composite_scores, quantile_sets

OUT = base.OUT
SPLIT = base.SPLIT
T_MIN = 2.6
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def screen(b0: dict, panel: dict) -> tuple[dict, pd.DataFrame]:
    m = pd.read_csv(OUT / "metrics_quarterly.csv", dtype={"ticker": str})
    m["ticker"] = m["ticker"].str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    books, rows = {}, []
    for d, sizes in b0.items():
        sub = m.loc[m["date"] == d].drop_duplicates("ticker").set_index("ticker")
        scored = sub.reindex([t for t in sizes if t in sub.index])
        geo = composite_scores(scored, base.VALUE_METRICS, method="geometric", min_metrics=4)
        top = quantile_sets(geo)[-1]
        books[d] = {t: sizes[t] for t in top}
        names = panel["name"].loc[d]
        w = pd.Series(books[d], dtype=float)
        w = w / w.sum() if len(w) else w
        for t in sizes:
            rows.append({
                "date": d, "ticker": t, "name": names.get(t),
                "in_metrics": t in scored.index, "scored": t in geo.index, "selected": t in top,
                "score": float(geo[t]) if t in geo.index else np.nan,
                "b0_mcap": sizes[t], "weight": float(w[t]) if t in w.index else 0.0,
            })
    return books, pd.DataFrame(rows)


def main() -> None:
    s = lc.load(base.cfg.END)
    days, panel, ann, exec_day, live = s["days"], s["panel"], s["ann"], s["exec_day"], s["live"]
    h6, detail = screen(s["b0"], panel)
    etf = lc.krx_daily.benchmark_prices(base.PRICE_START, base.cfg.END)["종가"].reindex(days).ffill()
    kodex = {
        "price": etf.pct_change().fillna(0.0),
        "total": pd.read_csv(OUT / "daily_returns.csv", index_col=0, parse_dates=True)["KODEX 200"].reindex(days),
    }
    saved_b0 = pd.read_csv(OUT / "largecap_daily_returns.csv", index_col=0, parse_dates=True)["B0 top 50"]

    rows, series = [], {}
    for var, dd in (("total", s["div_daily"]), ("price", None)):
        books = {}
        for label, sets in (("H6 value in B0", h6), ("B0 top 50", s["b0"])):
            b = base.run_book(sets, exec_day, panel["ret"], dd)
            books[label] = (b["net"][live], float(b["turnover"][live].mean() * ann))
        if var == "total":
            gap = (books["B0 top 50"][0] - saved_b0.reindex(books["B0 top 50"][0].index)).abs().max()
            if not np.isfinite(gap) or gap > 1e-8:
                raise SystemExit(f"rebuilt B0 disagrees with the saved backtest (max {gap})")
        books["KODEX 200"] = (kodex[var][live], 0.0)
        for label, (r, turn) in books.items():
            series[(var, label)] = r
            rows.append(lc.stats(var, label, r, ann, turn, None))
    table = pd.DataFrame(rows)

    spreads = []
    for var in ("total", "price"):
        d = series[(var, "H6 value in B0")] - series[(var, "B0 top 50")]
        k = series[(var, "H6 value in B0")] - series[(var, "KODEX 200")]
        rec = {"variant": var}
        for name, spread in (("b0", d), ("kodex", k)):
            for period, w in lc.windows(spread).items():
                rec[f"{name}_{period}_excess"] = float(w.mean() * ann)
                rec[f"{name}_{period}_t"] = base._tstat(w)
        rec["pass"] = (rec["b0_full_t"] >= T_MIN and rec["b0_train_excess"] > 0 and rec["b0_test_excess"] > 0)
        spreads.append(rec)
    spreads = pd.DataFrame(spreads)

    detail.to_csv(OUT / "b0_value_holdings.csv", index=False)
    table.to_csv(OUT / "b0_value_comparison.csv", index=False)
    spreads.to_csv(OUT / "b0_value_decision.csv", index=False)
    daily = pd.DataFrame({k: v for (var, k), v in series.items() if var == "total"})
    daily.to_csv(OUT / "b0_value_daily_returns.csv")

    q = detail.groupby("date").apply(lambda g: pd.Series({
        "n_b0": len(g), "n_scored": int(g["scored"].sum()), "n_selected": int(g["selected"].sum()),
        "top1": g.loc[g["selected"]].sort_values("weight").iloc[-1]["name"] if g["selected"].any() else "",
        "top1_weight": g.loc[g["selected"], "weight"].max() if g["selected"].any() else 0.0,
        "top2_weight": g.loc[g["selected"], "weight"].nlargest(2).sum() if g["selected"].any() else 0.0,
    }), include_groups=False)
    q.to_csv(OUT / "b0_value_quarterly.csv")

    fmt = lambda x: f"{x:.4f}"
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== H6 - B0 (decision: full t >= 2.6 and train, test excess > 0) ===")
    print(spreads.to_string(index=False, float_format=fmt))
    print("\n=== quarterly selected ===")
    print(q.to_string(float_format=fmt))
    plot(daily)


def plot(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, label, color, lw in (
        ("H6 value in B0", "H6 상위 50 안 복합가치", "#c33", 2.2),
        ("B0 top 50", "B0 상위 50", "black", 1.4),
        ("KODEX 200", "KODEX 200", "#36c", 1.4),
    ):
        ax.plot(daily.index, (1 + daily[col]).cumprod(), label=label, color=color, lw=lw)
    ax.axvline(SPLIT, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("H6: 기본 바스켓 안의 복합 가치 (비용·배당 근사)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "b0_value.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
