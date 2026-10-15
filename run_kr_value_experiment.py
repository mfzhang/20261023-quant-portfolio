"""
Phase 4: Korean market value hypotheses (pre-registered in README, "Phase 4 계획").

  universe  every KOSPI + KOSDAQ common stock listed on the signal date (later
            delisted names included): financials (KSIC 64-66 as filed, holding
            companies inside 649 kept; filers of financial-format statements carry no
            data), preferred shares, SPACs, REITs, KOSDAQ 관리종목 and names not
            trading that day excluded; then the bottom 20% by market cap and names
            with a 60-session average traded value < 500M KRW; then a balance
            sheet at most 200 days old
  H1        E/P, B/M, CF/P, S/P, EBIT/EV percentile ranks, geometric mean, >= 4 of 5;
            top 20% equal weight (quintiles reported)
  H2        S-RIM fair value / market cap, Ke 8%, w 0.9, ROE = 3:2:1 weighted last three
            fiscal-year controlling ROE, B = latest controlling equity > 0; top 20% EW
  H3        (current assets - total liabilities) / market cap >= 1.5, all EW, cash if none
  timing    signal at the last session of Mar / Jun / Sep / Dec with filings received
            by that day; trades at the next session's close
  costs     commission 1.5bps + slippage 20bps per side, sell tax by date
Benchmarks: eligible-universe EW and KODEX 200 (069500). Train <= 2023-05-10.
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

from kr import config_kr as cfg
from kr.data import dart_pit, krx, krx_daily
from strategies.composite_value import composite_scores, quantile_sets

OUT = ROOT / "output" / "kr_value"
SPLIT = pd.Timestamp(cfg.SPLIT)
PRICE_START = "2017-12-01"
VALUE_METRICS = ("ep", "bm", "cfp", "sp", "ebit_ev")
KE, W = 0.08, 0.9
NCAV_MIN = 1.5
MIN_TRDVAL = 500_000_000
SIZE_CUT = 0.20
TRADE_BPS = (cfg.COMMISSION_BPS + cfg.SLIPPAGE_BPS) / 10_000.0

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def sell_tax(d: pd.Timestamp) -> float:
    rate = 0.0
    for start, r in cfg.SELL_TAX_SCHEDULE:
        if d >= pd.Timestamp(start):
            rate = r
    return rate


# --- universe and signals ----------------------------------------------------------

def is_common(ticker: str, name: str) -> bool:
    return ticker.endswith("0") and not str(name).rstrip().endswith(("우", "우B", "우C")) and "우(" not in str(name)


def is_financial(industry, name: str, ksic_now: str) -> bool:
    """KSIC 64-66 as filed, except holding companies inside 649 (기타 금융업 also holds
    지주회사): current DART code 64992, a current code outside 64-66, or 홀딩스/지주 in the name."""
    code = str(industry).strip()
    if code[:2] not in ("64", "65", "66"):
        return False
    if code.startswith("649"):
        k = str(ksic_now or "")
        if k.startswith("64992") or (k and k[:2] not in ("64", "65", "66")) or any(s in str(name) for s in ("홀딩스", "지주")):
            return False
    return True


def company_mcap(mcap_row: pd.Series) -> pd.Series:
    """Common + preferred market cap per common ticker (preferred share the first five characters)."""
    m = mcap_row.dropna()
    pref = m[~m.index.str.endswith("0")]
    add = pref.groupby(pref.index.str[:5]).sum()
    common = m[m.index.str.endswith("0")]
    return common + common.index.str[:5].map(add).fillna(0.0).to_numpy()


def signal_dates(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = days.to_series()
    ends = s.groupby(days.to_period("M")).max()
    ends = ends[ends.index.month.isin([3, 6, 9, 12]) & (ends >= pd.Timestamp(cfg.START))]
    return [d for d in ends if d < days[-1]]


def build_signals(panel: dict, rep: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    sigs = signal_dates(days)
    books: dict[str, dict] = {k: {} for k in
                              [f"H1 Q{i}" for i in range(1, 6)] + [f"H2 Q{i}" for i in range(1, 6)] + ["H3", "eligible EW"]}
    counts, metrics_rows = [], []
    trd60 = panel["trdval"].fillna(0.0).where(panel["close"].notna()).rolling(60, min_periods=1).mean()
    ksic = dart_pit.induty_codes(sorted(set(rep.loc[rep["industry"].astype(str).str.startswith("649"), "ticker"])))
    for d in sigs:
        close, name = panel["close"].loc[d], panel["name"].loc[d]
        listed = close.dropna().index
        c = {"date": d, "listed": len(listed)}
        common = [t for t in listed if is_common(t, name[t])]
        c["common"] = len(common)
        nm = name[common].astype(str)
        sect = panel["section"].loc[d, common].fillna("").astype(str)
        spac = nm.str.contains("스팩") | sect.str.contains("SPAC")
        reit = nm.str.contains("리츠")
        admin = sect.str.contains("관리종목")
        halted = (panel["trdval"].loc[d, common].fillna(0) <= 0) | (panel["open"].loc[d, common].fillna(0) <= 0)
        keep = pd.Index(common)[~(spac | reit | admin | halted).to_numpy()]
        c.update({"spac": int(spac.sum()), "reit": int(reit.sum()), "admin_kosdaq": int(admin.sum()),
                  "halted": int(halted.sum())})
        f = dart_pit.asof_fundamentals(rep, d)
        fin = [t for t in keep if t in f.index and is_financial(f.at[t, "industry"], f.at[t, "name"], ksic.get(t))]
        keep = keep.difference(fin)
        c["financial"], c["after_exclusions"] = len(fin), len(keep)
        mc_common = panel["mcap"].loc[d, keep]
        keep = mc_common[mc_common > mc_common.quantile(SIZE_CUT)].index
        c["after_size"] = len(keep)
        keep = keep[(trd60.loc[d, keep] >= MIN_TRDVAL).to_numpy()]
        c["after_liquidity"] = len(keep)
        keep = keep[keep.isin(f.index)]
        c["eligible"] = len(keep)

        mcap = company_mcap(panel["mcap"].loc[d]).reindex(keep)
        g = f.reindex(keep)
        m = pd.DataFrame(index=keep)
        m["mcap"] = mcap
        m["ep"] = g["ttm_ni_parent"] / mcap
        m["bm"] = g["equity_parent"] / mcap
        m["cfp"] = g["ttm_cfo"] / mcap
        m["sp"] = g["ttm_revenue"] / mcap
        ev = mcap + g["debt"].fillna(0.0) - g["cash"].fillna(0.0)
        m["ebit_ev"] = (g["ttm_op_income"] / ev).where(ev > 0)
        b, roe = g["equity_parent"], g["roe3"]
        fair = b + b * (roe - KE) * W / (1.0 + KE - W)
        m["srim"] = (fair / mcap).where((b > 0) & roe.notna())
        m["ncav"] = (g["current_assets"] - g["liabilities"]) / mcap
        m["roe3"], m["avail_src"], m["report"] = roe, g["avail_src"], g["report"]
        m["consolidated"] = g["consolidated"]

        geo = composite_scores(m, VALUE_METRICS, method="geometric", min_metrics=4)
        for i, s in enumerate(quantile_sets(geo), 1):
            books[f"H1 Q{i}"][d] = s
        sr = m["srim"].dropna()
        for i, s in enumerate(quantile_sets(sr), 1):
            books[f"H2 Q{i}"][d] = s
        books["H3"][d] = set(m.index[m["ncav"] >= NCAV_MIN])
        books["eligible EW"][d] = set(keep)
        c.update({"h1_scored": len(geo), "h1_top": len(books["H1 Q5"][d]), "h2_scored": len(sr),
                  "h2_top": len(books["H2 Q5"][d]), "h3": len(books["H3"][d]),
                  **{f"has_{k}": int(m[k].notna().sum()) for k in (*VALUE_METRICS, "srim", "ncav")},
                  "avail_deadline": int((m["avail_src"] == "deadline").sum())})
        counts.append(c)
        m["date"] = d
        metrics_rows.append(m.reset_index(names="ticker"))
        print(f"  {d.date()}: listed {c['listed']}, eligible {c['eligible']}, H1 top {c['h1_top']}, "
              f"H2 top {c['h2_top']}, H3 {c['h3']}")
    return books, pd.DataFrame(counts), pd.concat(metrics_rows, ignore_index=True)


# --- book engine ---------------------------------------------------------------------

def run_book(sets: dict, exec_day: dict, ret: pd.DataFrame, div_daily: pd.DataFrame | None) -> pd.DataFrame:
    """Equal weight at each execution close, drifting in between. A held name that
    stops being listed is sold at its last close (sell costs charged) and the
    proceeds stay in cash until the next rebalance. Costs only on actual trades."""
    days, cols = ret.index, ret.columns
    R = ret.to_numpy()
    D = div_daily.reindex(index=days, columns=cols).fillna(0.0).to_numpy() if div_daily is not None else None
    col_ix = {t: i for i, t in enumerate(cols)}
    execs = {exec_day[d]: d for d in sets}
    w = np.zeros(len(cols))
    out = np.zeros((len(days), 5))  # gross, cost, turnover, n_held, invested
    started = False
    for t, day in enumerate(days):
        r = R[t]
        cost = turn = 0.0
        if started:
            held = w > 0
            gone = held & np.isnan(r)
            if gone.any():
                sold = w[gone].sum()
                cost += sold * (TRADE_BPS + sell_tax(day))
                turn += sold
                w[gone] = 0.0
            rr = np.nan_to_num(r)
            dv = D[t] if D is not None else 0.0
            gross = float((w * (rr + dv)).sum())
            w = w * (1.0 + rr + dv) / (1.0 + gross)
        else:
            gross = 0.0
        if day in execs:
            names = [col_ix[n] for n in sets[execs[day]] if n in col_ix and not np.isnan(r[col_ix[n]])]
            target = np.zeros(len(cols))
            if names:
                target[names] = 1.0 / len(names)
            diff = target - w
            buy, sell = diff.clip(min=0).sum(), (-diff).clip(min=0).sum()
            cost += (buy + sell) * TRADE_BPS + sell * sell_tax(day)
            turn += buy + sell
            w = target
            started = True
        out[t] = (gross, cost, turn, (w > 0).sum(), w.sum())
    df = pd.DataFrame(out, index=days, columns=["gross", "cost", "turnover", "n_held", "invested"])
    df["net"] = df["gross"] - df["cost"]
    return df


# --- stats -----------------------------------------------------------------------------

def _cagr(r: pd.Series, ann: float) -> float:
    return float((1.0 + r).prod() ** (ann / max(len(r), 1)) - 1.0)


def _sharpe(r: pd.Series, ann: float) -> float:
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _tstat(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan


def stats_row(label: str, net: pd.Series, bench: pd.Series, ann: float, turnover: float, gross_cagr: float) -> dict:
    tr, te = net[net.index <= SPLIT], net[net.index > SPLIT]
    return {"book": label, "cagr": _cagr(net, ann), "cagr_train": _cagr(tr, ann), "cagr_test": _cagr(te, ann),
            "vol": float(net.std() * np.sqrt(ann)), "sharpe": _sharpe(net, ann), "mdd": _mdd(net),
            "beta_kodex200": float(np.cov(net, bench)[0, 1] / np.var(bench, ddof=1)),
            "turnover": turnover, "cagr_gross": gross_cagr}


# --- main --------------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = krx_daily.trading_days(PRICE_START, cfg.END)
    krx_daily.fetch_prices(days, verbose=False)
    panel = krx_daily.load_panel(days)
    per_year = pd.Series(days.year).value_counts().sort_index()
    ann = float(per_year.loc[days[0].year + 1: days[-1].year - 1].mean())
    print(f"Sessions {days[0].date()} ~ {days[-1].date()} ({len(days)}), {panel['close'].shape[1]} tickers, "
          f"{ann:.1f} sessions/year")

    rep_path = OUT / "pit_reports.pkl"
    if rep_path.exists():
        rep = pd.read_pickle(rep_path)
    else:
        rep = dart_pit.report_table(dart_pit.statements_long(), dart_pit.filing_dates())
        rep.to_pickle(rep_path)
    print(f"PIT reports: {len(rep):,} ({rep['ticker'].nunique()} tickers); availability from receipt date "
          f"{(rep['avail_src'] == 'rcept').mean():.1%}, statutory deadline otherwise")

    books, counts, metrics = build_signals(panel, rep, days)
    counts.to_csv(OUT / "quarterly_counts.csv", index=False)
    metrics.to_csv(OUT / "metrics_quarterly.csv", index=False)
    sigs = sorted(books["eligible EW"])
    exec_day = {d: days[days.get_loc(d) + 1] for d in sigs}

    # dividends: KRX trailing DPS / close at each signal date, accrued per session
    divs, k200_yield = {}, {}
    for d in sigs:
        fsnap = krx_daily.fundamental_snapshot(d).set_index("ticker")
        divs[exec_day[d]] = fsnap["DIV"] / 100.0 / ann
        with krx_daily._quiet():
            roster = [t for t in krx.kospi200_asof(d) if t in panel["mcap"].columns]
        mc = panel["mcap"].loc[d, roster].dropna()
        y = fsnap["DIV"].reindex(mc.index).fillna(0.0) / 100.0
        k200_yield[exec_day[d]] = float((y * mc).sum() / mc.sum())
    div_daily = pd.DataFrame(divs).T.reindex(days).ffill()

    ret = panel["ret"]
    start = exec_day[sigs[0]]
    live = days > start
    etf = krx_daily.benchmark_prices(PRICE_START, cfg.END)["종가"].reindex(days).ffill()
    k200_price = etf.pct_change().fillna(0.0)
    k200_div = pd.Series(k200_yield).reindex(days).ffill().fillna(0.0) / ann
    variants = {"total": div_daily, "price": None}
    series, results, held = {}, {}, {}
    order = ["H1 Q5", "H1 Q4", "H1 Q3", "H1 Q2", "H1 Q1", "H2 Q5", "H2 Q4", "H2 Q3", "H2 Q2", "H2 Q1", "H3",
             "eligible EW"]
    for var, dd in variants.items():
        bench = (k200_price + (k200_div if var == "total" else 0.0))[live]
        rows = []
        for label in order:
            b = run_book(books[label], exec_day, ret, dd)
            net = b["net"][live]
            series[(var, label)] = net
            if var == "total":
                held[label] = b["n_held"]
            rows.append(stats_row(label, net, bench, ann, float(b["turnover"][live].mean() * ann),
                                  _cagr(b["gross"][live], ann)))
        series[(var, "KODEX 200")] = bench
        rows.append(stats_row("KODEX 200", bench, bench, ann, 0.0, _cagr(bench, ann)))
        results[var] = pd.DataFrame(rows)
        results[var].to_csv(OUT / f"comparison_{var}.csv", index=False)

    tot = {k: v for (var, k), v in series.items() if var == "total"}
    pd.DataFrame(tot).to_csv(OUT / "daily_returns.csv")
    print(f"\n=== {days[live][0].date()} ~ {days[-1].date()}, dividends accrued (train <= {SPLIT.date()} < test) ===")
    print(results["total"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== price only (no dividends) ===")
    print(results["price"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    spreads, decision = [], []
    for var in variants:
        s = {k: v for (vv, k), v in series.items() if vv == var}
        for a, b in (("H1 Q5", "eligible EW"), ("H2 Q5", "eligible EW"), ("H3", "eligible EW"),
                     ("H1 Q5", "H1 Q1"), ("H2 Q5", "H2 Q1"), ("eligible EW", "KODEX 200"),
                     ("H1 Q5", "KODEX 200"), ("H2 Q5", "KODEX 200"), ("H3", "KODEX 200")):
            diff = s[a] - s[b]
            for per, msk in (("full", diff.index == diff.index), ("train", diff.index <= SPLIT),
                             ("test", diff.index > SPLIT)):
                spreads.append({"variant": var, "spread": f"{a} - {b}", "period": per,
                                "ann_mean": diff[msk].mean() * ann, "t": _tstat(diff[msk])})
        sp = pd.DataFrame([x for x in spreads if x["variant"] == var])
        for h, top in (("H1", "H1 Q5"), ("H2", "H2 Q5"), ("H3", "H3")):
            row = sp[sp["spread"] == f"{top} - eligible EW"].set_index("period")
            rc = np.nan
            if h != "H3":
                q = [_cagr(s[f"{h} Q{i}"], ann) for i in range(1, 6)]
                rc = float(pd.Series(q).rank().corr(pd.Series(range(1, 6)), method="spearman"))
            ok = (row.at["full", "t"] >= 2.5 and row.at["train", "ann_mean"] > 0 and row.at["test", "ann_mean"] > 0
                  and (h == "H3" or rc > 0))
            decision.append({"variant": var, "hypothesis": h, "excess_full": row.at["full", "ann_mean"],
                             "t_full": row.at["full", "t"], "excess_train": row.at["train", "ann_mean"],
                             "t_train": row.at["train", "t"], "excess_test": row.at["test", "ann_mean"],
                             "t_test": row.at["test", "t"], "quintile_rank_corr": rc, "pass": bool(ok)})
    pd.DataFrame(spreads).to_csv(OUT / "spreads.csv", index=False)
    dec = pd.DataFrame(decision)
    dec.to_csv(OUT / "decision.csv", index=False)
    print("\n=== Spreads vs eligible EW / decision (t >= 2.5 full, train & test > 0, quintile rank corr > 0) ===")
    print(dec.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in tot.items()})
    yearly.to_csv(OUT / "yearly.csv")
    print("\n=== Calendar years (dividends accrued) ===")
    print(yearly.to_string(float_format=lambda v: f"{v:.3f}"))

    h3n = counts.set_index("date")["h3"]
    sanity(panel, rep, metrics, counts, books, days)
    plot(tot, results["total"], h3n, ann)


def sanity(panel: dict, rep: pd.DataFrame, metrics: pd.DataFrame, counts: pd.DataFrame, books: dict,
           days: pd.DatetimeIndex) -> None:
    print("\n=== Universe per quarter (after each filter) ===")
    cols = ["date", "listed", "common", "spac", "reit", "admin_kosdaq", "halted", "financial", "after_exclusions",
            "after_size", "after_liquidity", "eligible", "h1_scored", "h2_scored", "h3"]
    print(counts[cols].to_string(index=False))
    cov = counts[[c for c in counts if c.startswith("has_")]].div(counts["eligible"], axis=0)
    print("\nMetric coverage (share of eligible, mean / min):")
    print(pd.DataFrame({"mean": cov.mean(), "min": cov.min()}).round(3).to_string())
    print(f"Eligible names whose latest report used the statutory deadline (no receipt date matched): "
          f"{counts['avail_deadline'].sum() / counts['eligible'].sum():.2%}")

    last = days[-1]
    alive = set(panel["close"].loc[last].dropna().index)
    for label in ("eligible EW", "H1 Q5", "H2 Q5", "H3"):
        obs = [(d, t) for d, s in books[label].items() for t in s]
        names = {t for _, t in obs}
        dead = {t for t in names if t not in alive}
        share = np.mean([t not in alive for _, t in obs]) if obs else np.nan
        print(f"{label}: {len(names)} distinct names, {len(dead)} no longer listed on {last.date()} "
              f"({share:.1%} of name-quarters)")

    rows = []
    for d in (pd.Timestamp("2019-06-28"), pd.Timestamp("2023-06-30"), sorted(books["eligible EW"])[-1]):
        d = max(x for x in books["eligible EW"] if x <= d)
        fs = krx_daily.fundamental_snapshot(d).set_index("ticker")
        m = metrics[metrics["date"] == d].set_index("ticker")
        for t, nm in (("005930", "삼성전자"), ("000660", "SK하이닉스"), ("005380", "현대차"), ("000990", "DB하이텍"),
                      ("052400", "코나아이")):
            mc_common = panel["mcap"].at[d, t] if t in panel["mcap"] else np.nan
            row = {"date": d.date(), "ticker": t, "name": nm, "mcap_common_조": mc_common / 1e12}
            if t in m.index:
                row.update({"mcap_company_조": m.at[t, "mcap"] / 1e12, "PBR": 1 / m.at[t, "bm"],
                            "PER": 1 / m.at[t, "ep"], "eligible": True})
            else:
                row["eligible"] = False
            if t in fs.index:
                row.update({"KRX_PBR": fs.at[t, "PBR"], "KRX_PER": fs.at[t, "PER"]})
            rows.append(row)
    chk = pd.DataFrame(rows)
    chk.to_csv(OUT / "sanity_names.csv", index=False)
    print("\nKnown names: our PBR / PER (company market cap incl. preferred, controlling equity, TTM controlling NI)"
          " vs KRX (common price / latest annual BPS, EPS):")
    print(chk.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    s = panel["ret"]["005930"]
    print(f"\nSamsung 2018-05-04 (50:1 split) daily return used: {s.loc['2018-05-04']:+.4f}; "
          f"unadjusted reverse-split days priced by the market-cap ratio: {panel.get('n_broken_base', 0)}")
    big = panel["ret"].stack()
    print(f"Largest daily returns used: {big.max():+.2f} / {big.min():+.2f}")


def plot(tot: dict, table: pd.DataFrame, h3n: pd.Series, ann: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [3, 2, 1.3]})
    ax = axes[0]
    labels = {"H1 Q5": "H1 복합 가치 상위 20%", "H2 Q5": "H2 S-RIM 상위 20%", "H3": "H3 순유동자산 ≥ 1.5",
              "eligible EW": "적격 종목 동일가중", "KODEX 200": "KODEX 200"}
    for k, lab in labels.items():
        v = tot[k]
        ax.plot(v.index, ((1 + v).cumprod() - 1) * 100, label=lab, lw=2 if k.startswith("H") else 1.2)
    ax.axvline(SPLIT, color="black", ls="--", lw=1, label=f"학습 | 보류 ({SPLIT.date()})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("한국 가치 가설 H1·H2·H3 (누적 수익률 %, 비용·배당 근사 포함)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    t = table.set_index("book")["cagr"] * 100
    x = np.arange(5)
    axes[1].bar(x - 0.2, [t[f"H1 Q{i}"] for i in range(1, 6)], 0.4, label="H1 복합 가치", color="#c33")
    axes[1].bar(x + 0.2, [t[f"H2 Q{i}"] for i in range(1, 6)], 0.4, label="H2 S-RIM", color="#36c")
    axes[1].axhline(t["eligible EW"], color="black", ls="--", lw=1, label="적격 종목 동일가중")
    axes[1].set_xticks(x, [f"Q{i}" for i in range(1, 6)])
    axes[1].set_title("점수 5분위별 CAGR (%) — Q1 가장 비쌈, Q5 가장 쌈")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    axes[2].bar(h3n.index, h3n.values, width=60, color="#585")
    axes[2].set_title("H3 분기별 보유 종목 수")
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "kr_value.png", dpi=110)
    print(f"\nSaved {OUT / 'kr_value.png'}")


if __name__ == "__main__":
    main()
