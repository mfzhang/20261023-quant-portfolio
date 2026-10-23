"""
Two large-cap screens, run once after H6. Recorded in README section 22.
Not a pre-registered hypothesis and not an adopted rule.

Universe for both: B0 at the signal date (KOSPI+KOSDAQ common stocks, top 50 by
market cap, buffer 40/60). Banks, insurers and brokers are dropped because the
ratios are not comparable. Weights are common-share market cap. Quarterly
rebalance, costs and sample end match the other Korean books.

  GP   gross margin of the latest regular report (cumulative revenue and COGS
       on that report) = (매출 − 매출원가) / 매출. Top 25 by this ratio.
  BS   same report versus the same report one year earlier:
       current assets / assets rose, equity / assets rose, and that equity-ratio
       change is larger than the change in liabilities / equity.
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
from kr.data import dart_bulk, dart_pit

OUT = base.OUT
ITEM_CACHE = OUT / "quality_items.pkl"
TOP_N = 25
REPORT = {"1Q": "q1", "2Q": "half", "3Q": "q3", "4Q": "annual"}


def _extract() -> pd.DataFrame:
    if ITEM_CACHE.exists():
        return pd.read_pickle(ITEM_CACHE)
    specs = {
        "BS": {"assets": (["Assets"], r"자산\s*총계")},
        "PL": {"cogs": (["CostOfSales", "CostOfMerchandiseSold"], r"매출원가")},
    }
    rows = []
    for stmt, items in specs.items():
        files: dict[str, Path] = {}
        for path in sorted(dart_bulk.CACHE.glob(f"*_{stmt}_*.zip")):
            files["_".join(path.name.split("_")[:2])] = path
        for key, path in files.items():
            year, q = key.split("_")
            df = dart_bulk.read_zip(path)
            if df.empty:
                continue
            df["tag"] = df["항목코드"].map(dart_pit._tag)
            df["label"] = df["항목명"].str.strip()
            key_ix = pd.MultiIndex.from_frame(df[["종목코드", "consolidated"]])
            for name, (tags, pattern) in items.items():
                matched = df["tag"].isin(tags)
                has = pd.MultiIndex.from_frame(df.loc[matched, ["종목코드", "consolidated"]].drop_duplicates())
                matched = matched | (df["label"].str.contains(pattern, regex=True, na=False) & ~key_ix.isin(has))
                sub = df.loc[matched, ["종목코드", "consolidated", "결산기준일", "tag", "v_main"]].copy()
                if sub.empty:
                    continue
                sub["item"] = name
                sub["tag_rank"] = sub["tag"].map({t: i for i, t in enumerate(tags)}).fillna(len(tags))
                sub["report"] = REPORT[q]
                sub["year"] = year
                rows.append(sub)
            print(f"  extracted {key} {stmt}", flush=True)
    raw = pd.concat(rows, ignore_index=True)
    raw["ticker"] = raw["종목코드"].astype(str).str.strip("[] ").str.zfill(6)
    raw["period_end"] = pd.to_datetime(raw["결산기준일"], errors="coerce")
    keys = ["ticker", "consolidated", "period_end", "report", "item"]
    raw = raw.sort_values("tag_rank", kind="stable").drop_duplicates(keys, keep="first")
    wide = raw.pivot_table(index=keys[:-1], columns="item", values="v_main", aggfunc="first").reset_index()
    wide.to_pickle(ITEM_CACHE)
    return wide


def _latest(known: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    latest = known.sort_values(["ticker", "period_end", "avail"]).groupby("ticker").tail(1)
    latest = latest[(d - latest["period_end"]).dt.days <= 200]
    return latest.drop_duplicates("ticker", keep="last").set_index("ticker")


def _screens(b0: dict, rep: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    rep = rep.copy()
    rep["ticker"] = rep["ticker"].astype(str).str.zfill(6)
    items = _extract()
    rep = rep.merge(items, on=["ticker", "consolidated", "period_end", "report"], how="left")
    gp_book, bs_book, rows = {}, {}, []
    for d, sizes in b0.items():
        known = rep[rep["avail"] <= d]
        latest = _latest(known, d)
        names = panel_names(latest)
        fin = {t for t in sizes if t in names.index and base.is_financial(
            names.at[t, "industry"], names.at[t, "name"], "")}
        pool = [t for t in sizes if t not in fin and t in latest.index]
        cur = latest.loc[pool]
        # A few quarterly filings store COGS as a negative credit. Those rows are
        # dropped rather than flipped, so a sign error cannot rank a name first.
        margin = ((cur["revenue"] - cur["cogs"]) / cur["revenue"]).where((cur["revenue"] > 0) & (cur["cogs"] > 0))
        ranked = margin.dropna().sort_values(ascending=False)
        picked = list(ranked.index[:TOP_N])
        gp_book[d] = {t: sizes[t] for t in picked}

        prev = known[["ticker", "period_end", "report", "current_assets", "assets", "liabilities", "equity_parent"]].copy()
        prev["period_end"] = prev["period_end"] + pd.DateOffset(years=1)
        prev = prev.rename(columns={c: f"{c}_yoy" for c in ("current_assets", "assets", "liabilities", "equity_parent")})
        m = cur.reset_index().merge(prev, on=["ticker", "period_end", "report"], how="left").set_index("ticker")
        ca = m["current_assets"] / m["assets"]
        ca_p = m["current_assets_yoy"] / m["assets_yoy"]
        eq = m["equity_parent"] / m["assets"]
        eq_p = m["equity_parent_yoy"] / m["assets_yoy"]
        dr = m["liabilities"] / m["equity_parent"]
        dr_p = m["liabilities_yoy"] / m["equity_parent_yoy"]
        ok = (m["assets"] > 0) & (m["assets_yoy"] > 0) & (m["equity_parent"] > 0) & (m["equity_parent_yoy"] > 0)
        ok = ok & (ca > ca_p) & ((eq - eq_p) > 0) & ((eq - eq_p) > (dr - dr_p))
        chosen = list(m.index[ok.fillna(False)])
        bs_book[d] = {t: sizes[t] for t in chosen}
        w_gp = pd.Series(gp_book[d], dtype=float)
        w_bs = pd.Series(bs_book[d], dtype=float)
        rows.append({
            "date": d, "n_b0": len(sizes), "n_financial": len(fin), "n_margin": int(margin.notna().sum()),
            "n_gp": len(picked), "gp_top": names.at[picked[0], "name"] if picked else "",
            "gp_top_margin": float(ranked.iloc[0]) if len(ranked) else np.nan,
            "gp_top_weight": float(w_gp.max() / w_gp.sum()) if len(w_gp) else 0.0,
            "n_bs": len(chosen), "bs_top_weight": float(w_bs.max() / w_bs.sum()) if len(w_bs) else 0.0,
        })
    return gp_book, bs_book, pd.DataFrame(rows)


def panel_names(latest: pd.DataFrame) -> pd.DataFrame:
    return latest[["name", "industry"]]


def main() -> None:
    s = lc.load(base.cfg.END)
    days, panel, ann, exec_day, live = s["days"], s["panel"], s["ann"], s["exec_day"], s["live"]
    rep = pd.read_pickle(OUT / "pit_reports.pkl")
    gp, bs, counts = _screens(s["b0"], rep)
    etf = lc.krx_daily.benchmark_prices(base.PRICE_START, base.cfg.END)["종가"].reindex(days).ffill()
    kodex = {
        "price": etf.pct_change().fillna(0.0),
        "total": pd.read_csv(OUT / "daily_returns.csv", index_col=0, parse_dates=True)["KODEX 200"].reindex(days),
    }
    saved = pd.read_csv(OUT / "largecap_daily_returns.csv", index_col=0, parse_dates=True)["B0 top 50"]
    rows, series = [], {}
    for var, dd in (("total", s["div_daily"]), ("price", None)):
        ran = {}
        for label, sets in (("GP gross margin", gp), ("BS balance sheet", bs), ("B0 top 50", s["b0"])):
            b = base.run_book(sets, exec_day, panel["ret"], dd)
            ran[label] = (b["net"][live], float(b["turnover"][live].mean() * ann))
        if var == "total":
            gap = (ran["B0 top 50"][0] - saved.reindex(ran["B0 top 50"][0].index)).abs().max()
            if not np.isfinite(gap) or gap > 1e-8:
                raise SystemExit(f"rebuilt B0 disagrees with the saved backtest (max {gap})")
        ran["KODEX 200"] = (kodex[var][live], 0.0)
        for label, (r, turn) in ran.items():
            series[(var, label)] = r
            rows.append(lc.stats(var, label, r, ann, turn, None))
    table = pd.DataFrame(rows)
    spreads = []
    for var in ("total", "price"):
        for label, key in (("GP gross margin", "gp"), ("BS balance sheet", "bs")):
            d = series[(var, label)] - series[(var, "B0 top 50")]
            rec = {"variant": var, "book": label}
            for period, w in lc.windows(d).items():
                rec[f"{period}_excess"] = float(w.mean() * ann)
                rec[f"{period}_t"] = base._tstat(w)
            spreads.append(rec)
    spreads = pd.DataFrame(spreads)
    table.to_csv(OUT / "quality_comparison.csv", index=False)
    spreads.to_csv(OUT / "quality_spreads.csv", index=False)
    counts.to_csv(OUT / "quality_quarterly.csv", index=False)
    fmt = lambda x: f"{x:.4f}"
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== excess versus B0 ===")
    print(spreads.to_string(index=False, float_format=fmt))
    print("\n=== quarterly ===")
    print(counts.to_string(index=False, float_format=fmt))
    print(f"\nBS quarters with no names: {int((counts['n_bs'] == 0).sum())} / {len(counts)}")


if __name__ == "__main__":
    main()
