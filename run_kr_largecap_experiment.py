"""
Phase 4, H5: large-cap cap-weighted basket, short-term spike filter, volatility
targeting (pre-registered in README, "H5. 대형주 시총가중 + 급등주 제외 + 변동성 타기팅
(사후 설계, 사전 등록 2026-10-15)"). Fifth attempt, designed after seeing the large-cap
rally, so the backtest is descriptive; paper trading from 2026-10-15 is the test.

  universe  KOSPI + KOSDAQ common stocks on the signal date (no preferred, SPAC, REIT,
            infrastructure funds, KOSDAQ 관리종목, names not trading); financials kept
  B0        top 50 by common-share market cap, buffer: holdings stay while ranked <= 60,
            names ranked <= 40 enter, then fill / trim to 50 by rank; cap-weighted
  B1        B0 without the round(10%) selected names with the highest 21-session return
  H5        B1 with vol targeting: month-end exposure min(1.5, sigma_long / sigma_short)
            (sample std of B1 daily returns, 21 sessions vs expanding, 1 until 252
            sessions), traded next session, cash at 0%, 5% a year on borrowing
  timing    quarterly signals and costs as in run_kr_value_experiment.py
Decision: H5 has smaller MDD and Sharpe not lower than B0 in train and held-out.
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

import run_kr_value_experiment as base
from kr.data import krx_daily

OUT = base.OUT
SPLIT = base.SPLIT
TOP_N, ENTER_RANK, EXIT_RANK = 50, 40, 60
SPIKE_FRAC, SPIKE_DAYS = 0.10, 21
SHORT, LONG_MIN = 21, 252
MAX_EXPOSURE, BORROW_RATE = 1.5, 0.05
REBOUND = ("2020-03-20", "2020-12-31")


def universe_mcap(panel: dict, d: pd.Timestamp) -> pd.Series:
    close, name = panel["close"].loc[d], panel["name"].loc[d]
    common = [t for t in close.dropna().index if base.is_common(t, name[t])]
    nm = name[common].astype(str)
    sect = panel["section"].loc[d, common].fillna("").astype(str)
    drop = (nm.str.contains("스팩") | sect.str.contains("SPAC") | nm.str.contains("리츠")
            | nm.str.rstrip().str.endswith("인프라") | sect.str.contains("관리종목")
            | (panel["trdval"].loc[d, common].fillna(0) <= 0) | (panel["open"].loc[d, common].fillna(0) <= 0))
    keep = pd.Index(common)[~drop.to_numpy()]
    return panel["mcap"].loc[d, keep].dropna().sort_values(ascending=False)


def select(mcap: pd.Series, prev: list[str] | None) -> list[str]:
    order = list(mcap.index)
    if prev is None:
        return order[:TOP_N]
    rank = {t: i + 1 for i, t in enumerate(order)}
    sel = {t for t in prev if rank.get(t, np.inf) <= EXIT_RANK} | set(order[:ENTER_RANK])
    sel = sorted(sel, key=lambda t: rank[t])
    if len(sel) < TOP_N:
        sel += [t for t in order if t not in set(sel)][: TOP_N - len(sel)]
    return sel[:TOP_N]


def spike_names(sel: list[str], ret: pd.DataFrame, d: pd.Timestamp) -> list[str]:
    window = ret.loc[:d, sel].iloc[-SPIKE_DAYS:]
    r21 = ((1.0 + window).prod() - 1.0).where(window.notna().sum() == SPIKE_DAYS).dropna()
    n = int(round(SPIKE_FRAC * len(sel)))
    return list(r21.sort_values(ascending=False).index[:n])


def signals(panel: dict, sigs: list[pd.Timestamp]) -> tuple[dict, dict, pd.DataFrame]:
    b0, b1, rows, prev = {}, {}, [], None
    for d in sigs:
        mcap = universe_mcap(panel, d)
        sel = select(mcap, prev)
        spikes = spike_names(sel, panel["ret"], d)
        b0[d] = mcap[sel].to_dict()
        b1[d] = {t: v for t, v in b0[d].items() if t not in spikes}
        w = mcap[sel] / mcap[sel].sum()
        for t in sel:
            rows.append({"date": d, "ticker": t, "name": panel["name"].at[d, t], "rank": mcap.index.get_loc(t) + 1,
                         "mcap": mcap[t], "b0_weight": w[t], "spike_excluded": t in spikes,
                         "new": prev is not None and t not in prev})
        prev = sel
    return b0, b1, pd.DataFrame(rows)


def vol_target(r: pd.Series, ann: float) -> pd.DataFrame:
    """Scaled version of a book's daily net returns; `exposure` is the start-of-day share,
    `exposure_end` the share after that day's close (and any adjustment)."""
    idx, x = r.index, r.to_numpy()
    n = len(x)
    month_end = np.zeros(n, dtype=bool)
    month_end[:-1] = idx.month[:-1] != idx.month[1:]
    out = np.zeros((n, 6))  # net, exposure, trade, financing, exposure_end, target
    f, pending, target = 1.0, None, 1.0
    for t in range(n):
        start = f
        fin = max(f - 1.0, 0.0) * BORROW_RATE / ann
        g = f * x[t] - fin
        f = f * (1.0 + x[t]) / (1.0 + g) if 1.0 + g > 0 else 0.0
        cost = trade = 0.0
        if pending == t:
            d = target - f
            trade = abs(d)
            cost = trade * base.TRADE_BPS + max(-d, 0.0) * base.sell_tax(idx[t])
            f, pending = target, None
        if month_end[t]:
            if t + 1 >= LONG_MIN:
                s_short = np.std(x[t - SHORT + 1: t + 1], ddof=1)
                s_long = np.std(x[: t + 1], ddof=1)
                target = min(MAX_EXPOSURE, s_long / s_short) if s_short > 0 else 1.0
            else:
                target = 1.0
            pending = t + 1
        out[t] = (g - cost, start, trade, fin, f, target)
    return pd.DataFrame(out, index=idx, columns=["net", "exposure", "trade", "financing", "exposure_end", "target"])


def windows(r: pd.Series) -> dict[str, pd.Series]:
    return {"full": r, "train": r[r.index <= SPLIT], "test": r[r.index > SPLIT]}


def stats(var: str, label: str, r: pd.Series, ann: float, turnover: float, exposure: pd.Series | None) -> dict:
    row = {"variant": var, "book": label}
    for k, w in windows(r).items():
        row[f"cagr_{k}"] = base._cagr(w, ann)
        row[f"sharpe_{k}"] = base._sharpe(w, ann)
        row[f"mdd_{k}"] = base._mdd(w)
    row["vol"] = float(r.std() * np.sqrt(ann))
    m = (r.index >= REBOUND[0]) & (r.index <= REBOUND[1])
    row["rebound_2020"] = float((1 + r[m]).prod() - 1)
    row["turnover"] = turnover
    row["avg_exposure"] = float(exposure.mean()) if exposure is not None else 1.0
    row["share_exposure_above_1"] = float((exposure > 1.0 + 1e-9).mean()) if exposure is not None else 0.0
    return row


def load(end: str | None) -> dict:
    """Panel, signals, dividends and live sessions for the H5 books up to `end`."""
    days = krx_daily.trading_days(base.PRICE_START, end)
    if end is None:
        krx_daily.fetch_prices(days, verbose=False)
    panel = krx_daily.load_panel(days)
    per_year = pd.Series(days.year).value_counts().sort_index()
    ann = float(per_year.loc[days[0].year + 1: days[-1].year - 1].mean())
    sigs = base.signal_dates(days)
    exec_day = {d: days[days.get_loc(d) + 1] for d in sigs}
    b0, b1, holdings = signals(panel, sigs)
    divs = {exec_day[d]: krx_daily.fundamental_snapshot(d).set_index("ticker")["DIV"] / 100.0 / ann for d in sigs}
    div_daily = pd.DataFrame(divs).T.reindex(days).ffill()
    live = days > exec_day[sigs[0]]
    return {"days": days, "panel": panel, "ann": ann, "sigs": sigs, "exec_day": exec_day, "b0": b0, "b1": b1,
            "holdings": holdings, "div_daily": div_daily, "live": live}


def main() -> None:
    s = load(base.cfg.END)
    days, panel, ann, exec_day, live = s["days"], s["panel"], s["ann"], s["exec_day"], s["live"]
    reported = pd.read_csv(OUT / "daily_returns.csv", index_col=0, parse_dates=True)
    etf = krx_daily.benchmark_prices(base.PRICE_START, base.cfg.END)["종가"].reindex(days).ffill()
    k200 = {"total": reported["KODEX 200"], "price": etf.pct_change().fillna(0.0)[live]}

    rows, series, exposure = [], {}, {}
    for var, dd in (("total", s["div_daily"]), ("price", None)):
        books = {}
        for label, sets in (("B0 top 50", s["b0"]), ("B1 spike filter", s["b1"])):
            b = base.run_book(sets, exec_day, panel["ret"], dd)
            books[label] = (b["net"][live], float(b["turnover"][live].mean() * ann))
        v = vol_target(books["B1 spike filter"][0], ann)
        books["H5 vol target"] = (v["net"], books["B1 spike filter"][1] + float(v["trade"].sum() / len(v) * ann))
        exposure[(var, "H5 vol target")] = v["exposure"]
        books["KODEX 200"] = (k200[var], 0.0)
        kv = vol_target(k200[var], ann)
        books["KODEX 200 + vol target"] = (kv["net"], float(kv["trade"].sum() / len(kv) * ann))
        exposure[(var, "KODEX 200 + vol target")] = kv["exposure"]
        for label, (r, turn) in books.items():
            series[(var, label)] = r
            rows.append(stats(var, label, r, ann, turn, exposure.get((var, label))))
        if var == "total":
            fin = float(v["financing"].sum() / len(v) * ann)
    table = pd.DataFrame(rows)

    decision, filt = [], []
    for var in ("total", "price"):
        t = table[table.variant == var].set_index("book")
        ok = {k: bool(t.at["H5 vol target", f"mdd_{k}"] > t.at["B0 top 50", f"mdd_{k}"]
                      and t.at["H5 vol target", f"sharpe_{k}"] >= t.at["B0 top 50", f"sharpe_{k}"]) for k in ("train", "test")}
        decision.append({"variant": var, "pass_train": ok["train"], "pass_test": ok["test"],
                         "meets_rule": ok["train"] and ok["test"]})
        d = series[(var, "B1 spike filter")] - series[(var, "B0 top 50")]
        filt.append({"variant": var, **{f"{k}_{m}": val for k, w in windows(d).items()
                                         for m, val in (("excess", w.mean() * ann), ("t", base._tstat(w)))}})
    decision, filt = pd.DataFrame(decision), pd.DataFrame(filt)
    h = s["holdings"]
    conc = h.groupby("date").apply(lambda g: pd.Series({
        "n": len(g), "new": int(g["new"].sum()), "top1": g.sort_values("b0_weight").iloc[-1]["name"],
        "top1_weight": g["b0_weight"].max(), "top5_weight": g["b0_weight"].nlargest(5).sum(),
        "spike_excluded": ", ".join(g.loc[g["spike_excluded"], "name"].astype(str))}), include_groups=False)

    table.to_csv(OUT / "largecap_comparison.csv", index=False)
    decision.to_csv(OUT / "largecap_decision.csv", index=False)
    filt.to_csv(OUT / "largecap_filter.csv", index=False)
    h.to_csv(OUT / "largecap_holdings.csv", index=False)
    conc.to_csv(OUT / "largecap_quarterly.csv")
    daily = pd.DataFrame({k: v for (var, k), v in series.items() if var == "total"})
    daily["H5 exposure"] = exposure[("total", "H5 vol target")]
    daily.to_csv(OUT / "largecap_daily_returns.csv")
    fmt = lambda x: f"{x:.4f}"
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== decision (H5 vs B0: MDD smaller and Sharpe not lower, train and test) ===")
    print(decision.to_string(index=False))
    print("\n=== spike filter B1 - B0 ===")
    print(filt.to_string(index=False, float_format=fmt))
    print(f"\nH5 financing cost {fin:.4%} a year")
    print("\n=== quarterly ===")
    print(conc.to_string(float_format=fmt))
    plot(daily)


def plot(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1]})
    style = {"B0 top 50": ("B0 상위 50 시총가중", "black", 1.4), "B1 spike filter": ("B1 + 급등주 제외", "#585", 1.4),
             "H5 vol target": ("H5 = B1 + 변동성 타기팅", "#c33", 2.2), "KODEX 200": ("KODEX 200", "#36c", 1.4),
             "KODEX 200 + vol target": ("KODEX 200 + 변동성 타기팅", "#9bd", 1.0)}
    for k, (lab, color, lw) in style.items():
        eq = (1 + daily[k]).cumprod()
        axes[0].plot(eq.index, (eq - 1) * 100, label=lab, color=color, lw=lw)
        axes[1].plot(eq.index, (eq / eq.cummax() - 1) * 100, label=lab, color=color, lw=lw * 0.7)
    axes[2].plot(daily.index, daily["H5 exposure"] * 100, color="#c33", lw=1.2)
    axes[2].axhline(100, color="grey", lw=0.8)
    axes[0].set_title("H5 대형주 시총가중 + 급등주 제외 + 변동성 타기팅 (누적 수익률 %, 비용·배당 근사 포함)")
    axes[1].set_title("낙폭 (%)")
    axes[2].set_title("H5 주식 노출 (%)")
    axes[0].legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.axvline(SPLIT, color="black", ls="--", lw=1)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "largecap.png", dpi=110)
    print(f"\nSaved {OUT / 'largecap.png'}")


if __name__ == "__main__":
    main()
