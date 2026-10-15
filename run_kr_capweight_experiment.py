"""
Phase 4, H4: cap-weighted composite value (pre-registered in README, "H4. 시가총액 가중
복합 가치 (사후 가설, 사전 등록 2026-10-15)"). Fourth attempt, post hoc.

  rule        H1 exactly (composite of E/P, B/M, CF/P, S/P, EBIT/EV, >= 4 metrics, top 20%,
              quarterly, same universe / costs / split); the only change is weights
              proportional to the common-share market cap on the signal date, drifting
              between rebalances
  benchmark   the eligible universe weighted the same way (same costs); KODEX 200,
              eligible EW and H1 for reference
  decision    excess over the cap-weighted eligible basket: t >= 2.5 over the full
              period, positive in train and held-out (dividend-accrued version)
  crowding    top-5 weight and HHI per quarter; rerun without the two names with the
              largest full-period return contribution
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
T_MIN = 2.5


def capweight(sets: dict, mcap: pd.DataFrame) -> dict:
    return {d: mcap.loc[d, sorted(s)].dropna().to_dict() for d, s in sets.items()}


def windows(r: pd.Series) -> dict[str, pd.Series]:
    return {"full": r, "train": r[r.index <= SPLIT], "test": r[r.index > SPLIT]}


def stats(label: str, r: pd.Series, bench: pd.Series, ann: float, turnover: float) -> dict:
    row = {"book": label}
    for k, w in windows(r).items():
        row[f"cagr_{k}"] = base._cagr(w, ann)
        row[f"sharpe_{k}"] = base._sharpe(w, ann)
        row[f"mdd_{k}"] = base._mdd(w)
    row["vol"] = float(r.std() * np.sqrt(ann))
    row["beta_kodex200"] = float(np.cov(r, bench)[0, 1] / np.var(bench, ddof=1))
    row["turnover"] = turnover
    return row


def spread(a: pd.Series, b: pd.Series, ann: float) -> dict:
    d = a - b
    return {f"{k}_{m}": v for k, w in windows(d).items()
            for m, v in (("excess", w.mean() * ann), ("t", base._tstat(w)))}


def passes(s: dict) -> bool:
    return s["full_t"] >= T_MIN and s["train_excess"] > 0 and s["test_excess"] > 0


def concentration(sizes: dict) -> pd.DataFrame:
    rows = []
    for d, m in sizes.items():
        w = pd.Series(m) / sum(m.values())
        top = w.sort_values(ascending=False)
        rows.append({"date": d, "n": len(w), "top5_weight": float(top.iloc[:5].sum()),
                     "top1": top.index[0], "top1_weight": float(top.iloc[0]), "hhi": float((w ** 2).sum()),
                     "effective_n": float(1.0 / (w ** 2).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    days = krx_daily.trading_days(base.PRICE_START, base.cfg.END)
    panel = krx_daily.load_panel(days)
    per_year = pd.Series(days.year).value_counts().sort_index()
    ann = float(per_year.loc[days[0].year + 1: days[-1].year - 1].mean())
    rep = pd.read_pickle(OUT / "pit_reports.pkl")
    books, _, _ = base.build_signals(panel, rep, days)
    sigs = sorted(books["eligible EW"])
    exec_day = {d: days[days.get_loc(d) + 1] for d in sigs}
    divs = {exec_day[d]: krx_daily.fundamental_snapshot(d).set_index("ticker")["DIV"] / 100.0 / ann for d in sigs}
    div_daily = pd.DataFrame(divs).T.reindex(days).ffill()
    live = days > exec_day[sigs[0]]
    ret, mcap = panel["ret"], panel["mcap"]

    reported = pd.read_csv(OUT / "daily_returns.csv", index_col=0, parse_dates=True)
    etf = krx_daily.benchmark_prices(base.PRICE_START, base.cfg.END)["종가"].reindex(days).ffill()
    k200 = {"total": reported["KODEX 200"], "price": etf.pct_change().fillna(0.0)[live]}

    h4_sizes = capweight(books["H1 Q5"], mcap)
    cw_sizes = capweight(books["eligible EW"], mcap)
    plan = {"H4 cap-weighted value": h4_sizes, "eligible cap-weighted": cw_sizes,
            "H1 Q5 (EW)": books["H1 Q5"], "eligible EW": books["eligible EW"]}

    series, rows, contrib = {}, [], None
    for var, dd in (("total", div_daily), ("price", None)):
        for label, sets in plan.items():
            c = np.zeros(ret.shape[1]) if (var == "total" and label.startswith("H4")) else None
            b = base.run_book(sets, exec_day, ret, dd, contrib=c)
            net = b["net"][live]
            if c is not None:
                contrib = pd.Series(c, index=ret.columns)
            if var == "total" and label in ("H1 Q5 (EW)", "eligible EW"):
                ref = reported["H1 Q5" if label.startswith("H1") else "eligible EW"]
                assert net.index.equals(ref.index), f"{label}: sessions differ from item 18"
                gap = float((net - ref).abs().max())
                assert gap < 1e-12, f"{label}: differs from item 18 by {gap}"
            series[(var, label)] = net
            rows.append({"variant": var, **stats(label, net, k200[var], ann, float(b["turnover"][live].mean() * ann))})
        series[(var, "KODEX 200")] = k200[var]
        rows.append({"variant": var, **stats("KODEX 200", k200[var], k200[var], ann, 0.0)})
    print("H1 Q5 and eligible EW rebuilt identical to item 18")

    top2 = contrib.sort_values(ascending=False).index[:2].tolist()
    names = panel["name"].iloc[-1].reindex(contrib.index)
    top_tab = pd.DataFrame({"ticker": contrib.index, "name": names.to_numpy(), "contribution": contrib.to_numpy()})
    top_tab = top_tab.sort_values("contribution", ascending=False).head(10)
    ex_sizes = {d: {t: v for t, v in m.items() if t not in top2} for d, m in h4_sizes.items()}
    for var, dd in (("total", div_daily), ("price", None)):
        b = base.run_book(ex_sizes, exec_day, ret, dd)
        net = b["net"][live]
        series[(var, "H4 without top-2")] = net
        rows.append({"variant": var, **stats("H4 without top-2", net, k200[var], ann, float(b["turnover"][live].mean() * ann))})

    table = pd.DataFrame(rows)
    spreads, decision = [], []
    for var in ("total", "price"):
        s = {k: v for (vv, k), v in series.items() if vv == var}
        for a, b in (("H4 cap-weighted value", "eligible cap-weighted"), ("H4 without top-2", "eligible cap-weighted"),
                     ("H4 cap-weighted value", "KODEX 200"), ("eligible cap-weighted", "KODEX 200"),
                     ("H4 cap-weighted value", "H1 Q5 (EW)"), ("H4 cap-weighted value", "eligible EW"),
                     ("eligible cap-weighted", "eligible EW")):
            sp = spread(s[a], s[b], ann)
            spreads.append({"variant": var, "spread": f"{a} - {b}", **sp})
            if b == "eligible cap-weighted":
                decision.append({"variant": var, "book": a, **sp, "pass": passes(sp)})
    spreads, decision = pd.DataFrame(spreads), pd.DataFrame(decision)
    conc = concentration(h4_sizes).merge(
        concentration(cw_sizes)[["date", "top5_weight", "hhi"]].rename(columns={"top5_weight": "bench_top5_weight",
                                                                                "hhi": "bench_hhi"}), on="date")
    conc["top1_name"] = conc["top1"].map(names)

    table.to_csv(OUT / "capweight_comparison.csv", index=False)
    spreads.to_csv(OUT / "capweight_spreads.csv", index=False)
    decision.to_csv(OUT / "capweight_decision.csv", index=False)
    conc.to_csv(OUT / "capweight_concentration.csv", index=False)
    top_tab.to_csv(OUT / "capweight_top_contrib.csv", index=False)
    pd.DataFrame({k: v for (var, k), v in series.items() if var == "total"}).to_csv(OUT / "capweight_daily_returns.csv")
    fmt = lambda v: f"{v:.4f}"
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== spreads ===")
    print(spreads.to_string(index=False, float_format=fmt))
    print("\n=== decision (t >= 2.5 full, train and test > 0) ===")
    print(decision.to_string(index=False, float_format=fmt))
    print("\n=== top contributors (H4, dividend-accrued) ===")
    print(top_tab.to_string(index=False, float_format=fmt))
    print("\n=== concentration ===")
    print(conc.to_string(index=False, float_format=fmt))
    plot({k: v for (var, k), v in series.items() if var == "total"}, conc)


def plot(s: dict, conc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1.5]})
    ax = axes[0]
    style = {"H4 cap-weighted value": ("H4 시총가중 가치 상위 20%", "#c33", 2.2),
             "H4 without top-2": ("H4 (기여 상위 2종목 제외)", "#e99", 1.2),
             "eligible cap-weighted": ("적격 종목 시총가중", "black", 1.6),
             "KODEX 200": ("KODEX 200", "#36c", 1.6),
             "H1 Q5 (EW)": ("H1 동일가중 가치 상위 20%", "#585", 1.2),
             "eligible EW": ("적격 종목 동일가중", "grey", 1.0)}
    for k, (lab, color, lw) in style.items():
        v = s[k]
        ax.plot(v.index, ((1 + v).cumprod() - 1) * 100, label=lab, color=color, lw=lw)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("H4 시가총액 가중 복합 가치 (누적 수익률 %, 비용·배당 근사 포함)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    axes[1].plot(conc["date"], conc["top5_weight"] * 100, color="#c33", marker="o", label="H4 상위 5종목 비중")
    axes[1].plot(conc["date"], conc["bench_top5_weight"] * 100, color="black", marker=".", label="적격 시총가중 상위 5종목 비중")
    axes[1].set_title("분기별 상위 5종목 비중 (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capweight.png", dpi=110)
    print(f"\nSaved {OUT / 'capweight.png'}")


if __name__ == "__main__":
    main()
