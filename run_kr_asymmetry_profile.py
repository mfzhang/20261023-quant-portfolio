"""Describe train-window stocks that fell less than the market in down
episodes and rose more in up episodes. Point-in-time top 100 by market cap.
Not a trading rule. Recorded in README section 23."""

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

KSIC2 = {
    "10": "식료품", "11": "음료", "13": "섬유", "14": "의복", "16": "목재", "17": "펄프종이",
    "19": "석유정제", "20": "화학", "21": "의약품", "22": "고무플라스틱", "23": "비금속광물",
    "24": "1차금속", "25": "금속가공", "26": "전자부품·반도체·통신장비", "27": "의료정밀광학",
    "28": "전기장비", "29": "기타기계", "30": "자동차", "31": "기타운송장비", "32": "가구",
    "33": "기타제조", "35": "전기가스", "41": "종합건설", "42": "전문건설", "45": "자동차판매",
    "46": "도매", "47": "소매", "49": "육상운송", "50": "수상운송", "51": "항공운송",
    "52": "창고운송지원", "58": "출판·게임", "59": "영상", "60": "방송", "61": "통신",
    "62": "소프트웨어", "63": "정보서비스", "64": "금융", "65": "보험", "66": "금융지원",
    "68": "부동산", "70": "지주·본사", "71": "과학기술서비스", "72": "전문서비스",
    "73": "사업지원", "85": "교육", "86": "보건", "91": "스포츠",
}
DOWNS = [("2018-04-02", "2018-12-28"), ("2020-02-03", "2020-03-19"), ("2021-07-01", "2022-09-30")]
UPS = [("2019-01-02", "2019-12-30"), ("2020-03-20", "2020-12-30"), ("2023-01-02", "2023-05-10")]
SPLIT = pd.Timestamp("2023-05-10")
ASOF = pd.Timestamp("2023-03-31")


def _episode(r: pd.Series, start: str, end: str, need: int) -> float:
    w = r.loc[start:end].dropna()
    if len(w) < need:
        return np.nan
    return float((1.0 + w).prod() - 1.0)


def main() -> None:
    s = lc.load(base.cfg.END)
    panel, days = s["panel"], s["days"]
    ret = panel["ret"].loc[:SPLIT]
    kodex = lc.krx_daily.benchmark_prices(base.PRICE_START, base.cfg.END)["종가"].reindex(days).ffill().pct_change()
    kodex = kodex.loc[:SPLIT]
    sigs = [d for d in s["sigs"] if d <= SPLIT]
    universe = set()
    for d in sigs:
        mc = lc.universe_mcap(panel, d)
        universe |= set(mc.index[:100])
    names = panel["name"].loc[sigs[-1]].astype(str)

    def captures(windows: list[tuple[str, str]]) -> pd.DataFrame:
        rows = []
        for a, b in windows:
            idx = ret.loc[a:b].index
            need = int(np.ceil(0.8 * len(idx)))
            mkt = _episode(kodex, a, b, need)
            stock = ret.loc[a:b].apply(lambda col: _episode(col, a, b, need))
            rows.append(pd.DataFrame({"stock": stock, "mkt": mkt, "cap": stock / mkt}))
        # equal weight across episodes
        stk = pd.concat([x["stock"] for x in rows], axis=1).mean(axis=1)
        mkt = float(np.mean([x["mkt"].iloc[0] for x in rows]))
        cap = pd.concat([x["cap"] for x in rows], axis=1)
        return pd.DataFrame({"ret": stk, "capture": cap.mean(axis=1), "n": cap.notna().sum(axis=1), "mkt": mkt})

    down, up = captures(DOWNS), captures(UPS)
    both = down.join(up, lsuffix="_down", rsuffix="_up")
    both = both.loc[both.index.isin(universe)]
    both = both[(both["n_down"] >= 2) & (both["n_up"] >= 2)].copy()
    both["name"] = names.reindex(both.index)
    both["pass"] = (both["capture_down"] < 1.0) & (both["capture_up"] > 1.0)

    rep = pd.read_pickle(base.OUT / "pit_reports.pkl")
    items = pd.read_pickle(base.OUT / "quality_items.pkl")
    rep = rep.copy()
    rep["ticker"] = rep["ticker"].astype(str).str.zfill(6)
    rep = rep.merge(items, on=["ticker", "consolidated", "period_end", "report"], how="left")
    known = rep[rep["avail"] <= ASOF]
    latest = known.sort_values(["ticker", "period_end", "avail"]).groupby("ticker").tail(1)
    latest = latest[(ASOF - latest["period_end"]).dt.days <= 400].set_index("ticker")
    mcap = panel["mcap"].loc[ASOF]
    ratio = pd.DataFrame(index=latest.index)
    ratio["k2"] = latest["industry"].astype(str).str[:2]
    ratio["category"] = ratio["k2"].map(KSIC2).fillna(ratio["k2"])
    ratio["gross_margin"] = ((latest["revenue"] - latest["cogs"]) / latest["revenue"]).where(
        (latest["revenue"] > 0) & (latest["cogs"] > 0))
    ratio["op_margin"] = (latest["op_income"] / latest["revenue"]).where(latest["revenue"] > 0)
    ratio["equity_ratio"] = (latest["equity_parent"] / latest["assets"]).where(latest["assets"] > 0)
    ratio["ca_ratio"] = (latest["current_assets"] / latest["assets"]).where(latest["assets"] > 0)
    ratio["debt_ratio"] = (latest["liabilities"] / latest["equity_parent"]).where(latest["equity_parent"] > 0)
    ratio["bm"] = (latest["equity_parent"] / mcap.reindex(latest.index)).where(mcap.reindex(latest.index) > 0)
    both = both.join(ratio, how="left")

    cat = both.groupby("category").apply(lambda g: pd.Series({
        "n": len(g), "n_pass": int(g["pass"].sum()), "pass_rate": g["pass"].mean(),
        "down_capture": g["capture_down"].median(), "up_capture": g["capture_up"].median(),
        "down_ret": g["ret_down"].median(), "up_ret": g["ret_up"].median(),
    }), include_groups=False)
    cat = cat[cat["n"] >= 5].sort_values("pass_rate", ascending=False)

    cols = ["gross_margin", "op_margin", "equity_ratio", "ca_ratio", "debt_ratio", "bm"]
    med = pd.DataFrame({
        "통과": both.loc[both["pass"], cols].median(),
        "나머지": both.loc[~both["pass"], cols].median(),
    })
    med["차이"] = med["통과"] - med["나머지"]

    print(f"universe {len(both)}, pass {int(both['pass'].sum())}")
    print(f"market down episodes {both['mkt_down'].iloc[0]:.1%}, up episodes {both['mkt_up'].iloc[0]:.1%}")
    print("\n=== category (n>=5), sorted by pass rate ===")
    print(cat.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== ratios at 2023-03, median ===")
    print(med.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== passing names ===")
    show = both.loc[both["pass"], ["name", "category", "capture_down", "capture_up", "ret_down", "ret_up"]]
    print(show.sort_values("capture_up", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
    cat.to_csv(base.OUT / "asymmetry_category.csv")
    med.to_csv(base.OUT / "asymmetry_ratios.csv")
    both.to_csv(base.OUT / "asymmetry_names.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
