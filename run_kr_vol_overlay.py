"""
Phase 4 overlay: volatility targeting (pre-registered in README, "하락 대응 오버레이:
변동성 타기팅 (사전 등록, 2026-10-14)"), Moreira & Muir 2017 style.

  signal    at each month-end session, sigma_short = annualized realized vol of the
            book's own daily returns over the last 21 sessions (month end included);
            sigma_long = the same over every live session so far (expanding).
            Realized vol = sqrt(mean of squared daily returns x sessions/year).
            Weight 1 until 252 live sessions exist.
  weight    min(1, sigma_long / sigma_short), rest in cash at 0%
  timing    traded at the next session's close (same as the base books), held until
            the next adjustment; between adjustments the book share drifts with the book
  costs     |change in exposure| x (commission + slippage), plus sell tax on reductions
  books     H1 Q5, H2 Q5, H3, eligible EW from run_kr_value_experiment.py (item 18),
            dividend-accrued (primary) and price-only

Scaling the book's net daily return is an approximation of scaling every position:
the base book's own rebalancing costs scale with exposure, and positions are not
traded name by name.
Decision: adopt if MDD is smaller AND Sharpe is not lower, in both train and held-out.
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
SHORT, LONG_MIN = 21, 252
BOOKS = ["H1 Q5", "H2 Q5", "H3", "eligible EW"]
EPISODES = {
    "2018 Q4": ("2018-10-01", "2018-12-31"),
    "2020-03 코로나": ("2020-02-01", "2020-03-31"),
    "2022 약세장": ("2022-01-01", "2022-12-31"),
    "2024 하반기": ("2024-07-01", "2024-12-31"),
}


def overlay(r: pd.Series, ann: float) -> pd.DataFrame:
    """Vol-targeted version of a book's daily net returns."""
    idx, x = r.index, r.to_numpy()
    n = len(x)
    month_end = np.zeros(n, dtype=bool)
    month_end[:-1] = idx.month[:-1] != idx.month[1:]
    out = np.zeros((n, 4))  # net, exposure (end of day), trade, weight target
    f, pending, target = 1.0, None, 1.0
    for t in range(n):
        g = f * x[t]
        f = f * (1.0 + x[t]) / (1.0 + g) if 1.0 + g != 0 else 0.0
        cost = trade = 0.0
        if pending == t:
            d = target - f
            trade = abs(d)
            cost = trade * base.TRADE_BPS + max(-d, 0.0) * base.sell_tax(idx[t])
            f, pending = target, None
        out[t, :3] = (g - cost, f, trade)
        if month_end[t]:
            if t + 1 >= LONG_MIN:
                s_short = np.sqrt(np.mean(x[t - SHORT + 1: t + 1] ** 2) * ann)
                s_long = np.sqrt(np.mean(x[: t + 1] ** 2) * ann)
                target = min(1.0, s_long / s_short) if s_short > 0 else 1.0
            else:
                target = 1.0
            pending = t + 1
        out[t, 3] = target
    return pd.DataFrame(out, index=idx, columns=["net", "exposure", "trade", "target"])


def _windows(r: pd.Series) -> dict[str, pd.Series]:
    return {"full": r, "train": r[r.index <= SPLIT], "test": r[r.index > SPLIT]}


def stats(r: pd.Series, ann: float, exposure: pd.Series | None, trade: pd.Series | None) -> dict:
    row = {}
    for k, w in _windows(r).items():
        row[f"cagr_{k}"] = base._cagr(w, ann)
        row[f"sharpe_{k}"] = base._sharpe(w, ann)
        row[f"mdd_{k}"] = base._mdd(w)
    row["vol"] = float(r.std() * np.sqrt(ann))
    row["avg_exposure"] = float(exposure.mean()) if exposure is not None else 1.0
    row["overlay_turnover"] = float(trade.sum() / len(r) * ann) if trade is not None else 0.0
    return row


def worst_episode(r: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    return peak, trough


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

    reported = pd.read_csv(OUT / "daily_returns.csv", index_col=0, parse_dates=True)
    series = {}
    for var, dd in (("total", div_daily), ("price", None)):
        for label in BOOKS:
            net = base.run_book(books[label], exec_day, panel["ret"], dd)["net"][live]
            if var == "total":
                assert net.index.equals(reported.index), f"{label}: sessions differ from item 18"
                gap = float((net - reported[label]).abs().max())
                assert gap < 1e-12, f"{label}: rebuilt total-return book differs from item 18 by {gap}"
            series[(var, label)] = net
    print(f"Rebuilt books match item 18 daily returns; {ann:.1f} sessions/year, "
          f"{series[('total', 'H1 Q5')].index[0].date()} ~ {days[-1].date()}")

    rows, decisions, episodes, daily = [], [], [], {}
    for (var, label), r in series.items():
        ov = overlay(r, ann)
        b_row = {"variant": var, "book": label, "overlay": "none", **stats(r, ann, None, None)}
        o_row = {"variant": var, "book": label, "overlay": "vol target",
                 **stats(ov["net"], ann, ov["exposure"].shift(1).fillna(1.0), ov["trade"])}
        rows += [b_row, o_row]
        ok = {k: (o_row[f"mdd_{k}"] > b_row[f"mdd_{k}"]) and (o_row[f"sharpe_{k}"] >= b_row[f"sharpe_{k}"])
              for k in ("train", "test")}
        decisions.append({"variant": var, "book": label,
                          **{f"d_{m}_{k}": o_row[f"{m}_{k}"] - b_row[f"{m}_{k}"]
                             for m in ("cagr", "sharpe", "mdd") for k in ("full", "train", "test")},
                          "pass_train": ok["train"], "pass_test": ok["test"],
                          "adopt": ok["train"] and ok["test"]})
        if var == "total":
            daily[f"{label} base"] = r
            daily[f"{label} overlay"] = ov["net"]
            daily[f"{label} exposure"] = ov["exposure"].shift(1).fillna(1.0)
            peak, trough = worst_episode(r)
            wins = dict(EPISODES)
            wins[f"최악 낙폭 ({peak:%Y-%m-%d}~{trough:%Y-%m-%d})"] = (peak, trough)
            for name, (a, b) in wins.items():
                m = (r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
                episodes.append({"book": label, "episode": name,
                                 "ret_base": float((1 + r[m]).prod() - 1), "ret_overlay": float((1 + ov["net"][m]).prod() - 1),
                                 "mdd_base": base._mdd(r[m]), "mdd_overlay": base._mdd(ov["net"][m]),
                                 "avg_exposure": float(daily[f"{label} exposure"][m].mean())})

    table = pd.DataFrame(rows)
    dec = pd.DataFrame(decisions)
    epi = pd.DataFrame(episodes)
    table.to_csv(OUT / "vol_overlay_summary.csv", index=False)
    dec.to_csv(OUT / "vol_overlay_decision.csv", index=False)
    epi.to_csv(OUT / "vol_overlay_episodes.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "vol_overlay_daily.csv")
    fmt = lambda v: f"{v:.4f}"
    print("\n=== overlay vs none ===")
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== decision (MDD smaller and Sharpe not lower, train and test) ===")
    print(dec.to_string(index=False, float_format=fmt))
    print("\n=== episodes (dividend-accrued) ===")
    print(epi.to_string(index=False, float_format=fmt))
    plot(pd.DataFrame(daily))


def plot(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]})
    for j, (label, title) in enumerate((("H2 Q5", "H2 S-RIM 상위 20%"), ("eligible EW", "적격 종목 동일가중"))):
        for kind, color, lab in (("base", "#888", "오버레이 없음"), ("overlay", "#c33", "변동성 타기팅")):
            eq = (1 + daily[f"{label} {kind}"]).cumprod()
            axes[0, j].plot(eq.index, (eq - 1) * 100, color=color, lw=1.5, label=lab)
            axes[1, j].plot(eq.index, (eq / eq.cummax() - 1) * 100, color=color, lw=1.2, label=lab)
        axes[2, j].plot(daily.index, daily[f"{label} exposure"] * 100, color="#36c", lw=1.2)
        axes[0, j].set_title(f"{title}: 누적 수익률 (%, 비용·배당 근사 포함)")
        axes[1, j].set_title("낙폭 (%)")
        axes[2, j].set_title("주식 비중 (%)")
        axes[2, j].set_ylim(0, 105)
        for ax in axes[:, j]:
            ax.axvline(SPLIT, color="black", ls="--", lw=1)
            ax.grid(alpha=0.3)
        axes[0, j].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "vol_overlay.png", dpi=110)
    print(f"\nSaved {OUT / 'vol_overlay.png'}")


if __name__ == "__main__":
    main()
