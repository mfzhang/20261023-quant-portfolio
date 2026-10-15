"""
Point-in-time financial statement table for Korean listed companies.

Values come from the OpenDART bulk files (kr/data/dart_bulk.py). Availability
date = receipt date of the first filing of that report (rcept_dt from the
all-company periodic filing list, matched on corp_code + report type + period
end). When no receipt date matches, the statutory deadline is used instead
(annual: period end + 90 days, half / quarterly: + 45 days), which is never
earlier than the actual filing for timely filers.

Consolidated (CFS) statements are preferred, separate (OFS) used otherwise.
Flows: quarterly reports carry 3-month and cumulative columns; TTM is built
from fiscal-year cumulative values: TTM(Qk) = cum(Qk) + annual(prev FY) - cum(Qk, prev FY).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from kr.data import dart, dart_bulk

CACHE = dart.CACHE_DIR
FAMILY = {"사업보고서": "annual", "반기보고서": "half", "분기보고서": "quarter"}
REPORT_FAMILY = {"annual": "annual", "half": "half", "q1": "quarter", "q3": "quarter"}
_RPT = re.compile(r"(사업보고서|반기보고서|분기보고서)\s*\((\d{4})\.(\d{2})\)")


def filing_dates(start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    """All periodic filings (pblntf_ty A) of every company, month by month.
    Columns: corp_code, stock_code, corp_name, family, period (YYYY-MM), rcept_dt."""
    end_ts = pd.Timestamp(end or pd.Timestamp.today())
    path = CACHE / "periodic_filings.csv"
    done = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()
    have = set(done["month"]) if not done.empty else set()
    rows = []
    months = pd.period_range(pd.Timestamp(start), end_ts, freq="M")
    for i, m in enumerate(months):
        key = str(m)
        # the current and previous month are re-fetched (still receiving filings)
        if key in have and i < len(months) - 2:
            continue
        bgn, fin = m.start_time, min(m.end_time, end_ts)
        page = 1
        while True:
            params = {"bgn_de": bgn.strftime("%Y%m%d"), "end_de": fin.strftime("%Y%m%d"), "pblntf_ty": "A",
                      "page_no": page, "page_count": 100}
            data = _retry(dart._get_json, "list", params) if i < len(months) - 2 else _retry(_get_fresh, params)
            for r in data.get("list", []):
                rows.append({**{k: r.get(k, "") for k in ("corp_code", "stock_code", "corp_name", "corp_cls",
                                                           "report_nm", "rcept_no", "rcept_dt")}, "month": key})
            if page >= int(data.get("total_page", 1) or 1):
                break
            page += 1
        done = done[done["month"] != key] if not done.empty else done
    new = pd.DataFrame(rows)
    allf = pd.concat([done, new], ignore_index=True) if not new.empty else done
    allf.to_csv(path, index=False)
    m = allf["report_nm"].str.extract(_RPT)
    allf = allf.assign(family=m[0].map(FAMILY), period=m[1] + "-" + m[2])
    allf = allf.dropna(subset=["family"])
    allf["rcept_dt"] = pd.to_datetime(allf["rcept_dt"])
    return allf


def _retry(fn, *args, tries: int = 5):
    import time
    import requests
    for k in range(tries):
        try:
            return fn(*args)
        except (requests.ConnectionError, requests.Timeout):
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def induty_codes(tickers: list[str]) -> dict[str, str]:
    """5-digit KSIC industry code (DART company info, current) per ticker."""
    corp = dart.corp_codes().drop_duplicates("stock_code").set_index("stock_code")["corp_code"]
    out = {}
    for t in tickers:
        if t in corp.index:
            data = _retry(dart._get_json, "company", {"corp_code": corp[t]})
            out[t] = str(data.get("induty_code", "") or "")
    return out


def _get_fresh(params: dict) -> dict:
    import requests
    resp = requests.get(f"{dart.BASE}/list.json", params={"crtfc_key": dart._key(), **params}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "020":
        raise RuntimeError("DART daily request limit reached")
    return data


# --- bulk statements -> per-report item values ------------------------------------

def _tag(code: str) -> str:
    """'ifrs-full_Revenue' / 'ifrs_Revenue' -> 'Revenue'; entity-specific tags keep their prefix."""
    code = code.strip()
    for p in ("ifrs-full_", "ifrs_", "dart_"):
        if code.startswith(p):
            return code[len(p):]
    return code


# item -> (statement, candidate tags in priority order, Korean name patterns used
# when no standard tag is present)
ITEMS = {
    "current_assets": ("BS", ["CurrentAssets"], None),
    "liabilities": ("BS", ["Liabilities"], r"^부채\s*총계$"),
    "equity_total": ("BS", ["Equity"], r"^자본\s*총계$"),
    "equity_parent": ("BS", ["EquityAttributableToOwnersOfParent"], r"지배기업.*소유주.*지분|지배기업\s*지분"),
    "cash": ("BS", ["CashAndCashEquivalents"], None),
    "revenue": ("PL", ["Revenue", "Revenues"], r"^(?:매출액|수익\s*\(매출액\)|영업수익|매출)$"),
    "op_income": ("PL", ["OperatingIncomeLoss"], r"^영업이익(?:\s*\(손실\))?$|^영업손익$"),
    "net_income": ("PL", ["ProfitLoss"], r"^(?:당기순이익|분기순이익|반기순이익)(?:\s*\(손실\))?$"),
    "ni_parent": ("PL", ["ProfitLossAttributableToOwnersOfParent"], r"지배기업.*소유주|지배기업\s*지분"),
    "cfo": ("CF", ["CashFlowsFromUsedInOperatingActivities"], r"^영업활동\s*(?:으로\s*인한\s*)?현금\s*흐름"),
}
# Interest-bearing debt (borrowings and bonds; leases excluded as in the US work):
# the first tag present in each group, summed over groups. Groups hold
# alternative tags for the same line (gross / net presentations).
DEBT_GROUPS = [
    ["ShorttermBorrowings"],
    ["CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings"],
    ["CurrentPortionOfLongtermBorrowings"],
    ["CurrentBondsIssuedAndCurrentPortionOfNoncurrentBondsIssued", "CurrentPortionOfBonds", "CurrentPortionOfBondsIssued"],
    ["CurrentPortionOfConvertibleBonds"],
    ["CurrentPortionOfBondWithWarrant"],
    ["CurrentPortionOfExchangeableBond"],
    ["OtherCurrentBorrowingsAndCurrentPortionOfOtherNoncurrentBorrowings"],
    ["LongtermBorrowings", "LongTermBorrowingsGross"],
    ["NoncurrentPortionOfNoncurrentBondsIssued", "BondsIssued"],
    ["ConvertibleBondsNet", "ConvertibleBonds"],
    ["BondWithWarrantNet", "BondWithWarrant"],
    ["ExchangeableBondsNet", "ExchangeableBonds"],
    ["NoncurrentPortionOfOtherNoncurrentBorrowings"],
]
VALUE_COLS = ["v_main", "v_3m", "v_prev", "v_prev_cum"]


def _debt(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["종목코드", "consolidated", "결산기준일", "통화"]
    parts = []
    for g in DEBT_GROUPS:
        sub = df[df["tag"].isin(g)].copy()
        if sub.empty:
            continue
        sub["rank"] = sub["tag"].map({t: i for i, t in enumerate(g)})
        parts.append(sub.sort_values("rank", kind="stable").drop_duplicates(keys))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).groupby(keys, as_index=False)[VALUE_COLS].sum(min_count=1)
    out["item"], out["tag_rank"], out["tag"] = "debt", 0, "debt"
    return out


def parse_period(year: str, q: str) -> pd.DataFrame:
    """Long table for one bulk report period (q in 1Q/2Q/3Q/4Q): ticker, consolidated,
    period_end, report, item, v_main (cumulative for flows), v_3m, v_prev, plus
    name / market / industry / fiscal-year month as filed."""
    out, metas = [], []
    for stmt in ("BS", "PL", "CF"):
        hit = dart_bulk._cached(f"{year}_{q}_{stmt}_x.zip")
        if hit is None:
            continue
        df = dart_bulk.read_zip(hit)
        if df.empty:
            continue
        metas.append(df[["종목코드", "회사명", "시장구분", "업종", "업종명", "결산월"]].drop_duplicates("종목코드"))
        df["tag"] = df["항목코드"].map(_tag)
        df["label"] = df["항목명"].str.strip()
        key = pd.MultiIndex.from_frame(df[["종목코드", "consolidated"]])
        for name, (s, tags, pattern) in ITEMS.items():
            if s != stmt:
                continue
            m = df["tag"].isin(tags)
            if pattern:
                # label fallback only for filers without a standard tag for this item
                has_tag = pd.MultiIndex.from_frame(df.loc[m, ["종목코드", "consolidated"]].drop_duplicates())
                m = m | (df["label"].str.contains(pattern, regex=True, na=False) & ~key.isin(has_tag))
            sub = df.loc[m, ["종목코드", "consolidated", "결산기준일", "통화", "tag", *VALUE_COLS]].copy()
            if sub.empty:
                continue
            sub["item"] = name
            sub["tag_rank"] = sub["tag"].map({t: i for i, t in enumerate(tags)}).fillna(len(tags))
            out.append(sub)
        if stmt == "BS":
            out.append(_debt(df))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df = df[df["통화"].str.strip().isin(["KRW", ""])]
    df["ticker"] = df["종목코드"].str.strip("[] ")
    df["period_end"] = pd.to_datetime(df["결산기준일"], errors="coerce")
    df["report"] = {"1Q": "q1", "2Q": "half", "3Q": "q3", "4Q": "annual"}[q]
    keys = ["ticker", "consolidated", "item", "period_end", "report"]
    # PL zip holds both 손익계산서 and 포괄손익계산서; the first row per item wins
    vals = (df.sort_values("tag_rank", kind="stable")
            .drop_duplicates(keys, keep="first")[keys + VALUE_COLS])
    meta = pd.concat(metas).drop_duplicates("종목코드")
    meta["ticker"] = meta["종목코드"].str.strip("[] ")
    meta = meta.drop(columns="종목코드").rename(columns={"회사명": "name", "시장구분": "market", "업종": "industry",
                                                     "업종명": "industry_name", "결산월": "fy_month"})
    return vals.merge(meta, on="ticker", how="left")


DEADLINE_DAYS = {"annual": 90, "half": 45, "q1": 45, "q3": 45}


def report_table(long: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, period_end, report) on a single basis (consolidated when
    the filer reports consolidated statements, else separate), with columns
    <item> (current / cumulative), <item>_prev (prior FY or prior year-end) and
    <item>_pcum (prior-year same-period cumulative), plus 'avail' (availability
    date), 'avail_src' ('rcept' or 'deadline') and filer metadata."""
    keys = ["ticker", "period_end", "report"]
    has_cfs = long[long["consolidated"] & long["item"].isin(["equity_total", "liabilities"])][keys].drop_duplicates()
    has_cfs["use_cfs"] = True
    df = long.merge(has_cfs, on=keys, how="left")
    df = df[df["consolidated"] == df["use_cfs"].eq(True)]
    wide = df.pivot_table(index=keys, columns="item", values=["v_main", "v_prev", "v_prev_cum"], aggfunc="first")
    wide.columns = [it if v == "v_main" else f"{it}_{'prev' if v == 'v_prev' else 'pcum'}" for v, it in wide.columns]
    wide = wide.reset_index()
    meta = (df.drop_duplicates(keys)[keys + ["name", "market", "industry", "fy_month", "consolidated"]])
    wide = wide.merge(meta, on=keys, how="left")
    # controlling-shareholder figures; filers without NCI report only the totals
    for base, total in (("ni_parent", "net_income"), ("equity_parent", "equity_total")):
        for suf in ("", "_prev", "_pcum"):
            a, b = base + suf, total + suf
            if b in wide:
                wide[a] = wide[a].combine_first(wide[b]) if a in wide else wide[b]

    # receipt dates: earliest filing of the same report family and period
    corp = dart.corp_codes()[["corp_code", "stock_code"]]
    f = filings.merge(corp.rename(columns={"stock_code": "sc2"}), on="corp_code", how="left")
    f["tk"] = f["stock_code"].where(f["stock_code"].fillna("").str.len() == 6, f["sc2"])
    by_tk = (f.dropna(subset=["tk"]).groupby(["tk", "family", "period"])["rcept_dt"].min()
             .rename("rcept").reset_index())
    by_nm = (f.groupby(["corp_name", "family", "period"])["rcept_dt"].min().rename("rcept_nm").reset_index())
    wide["family"] = wide["report"].map(REPORT_FAMILY)
    wide["period"] = wide["period_end"].dt.strftime("%Y-%m")
    wide = wide.merge(by_tk, left_on=["ticker", "family", "period"], right_on=["tk", "family", "period"], how="left")
    wide = wide.merge(by_nm, left_on=["name", "family", "period"], right_on=["corp_name", "family", "period"],
                      how="left")
    deadline = wide["period_end"] + pd.to_timedelta(wide["report"].map(DEADLINE_DAYS), unit="D")
    rcept = wide["rcept"].combine_first(wide["rcept_nm"])
    ok = rcept.notna() & (rcept > wide["period_end"])
    wide["avail"] = rcept.where(ok, deadline)
    wide["avail_src"] = np.where(ok, "rcept", "deadline")
    return wide.drop(columns=["tk", "corp_name", "rcept", "rcept_nm", "family", "period"])


def _fy_table(rep: pd.DataFrame) -> pd.DataFrame:
    """Annual figures per (ticker, fiscal-year end) from annual reports already
    available: net income and equity (controlling) at the year end and the
    beginning of the year. A year missing its own report is filled from the next
    annual report's prior-year column."""
    ann = rep[rep["report"] == "annual"].sort_values(["ticker", "period_end"])
    own = ann[["ticker", "period_end", "ni_parent", "equity_parent", "equity_parent_prev"]].rename(
        columns={"ni_parent": "ni", "equity_parent": "eq_end", "equity_parent_prev": "eq_beg"})
    prior = ann[["ticker", "period_end", "ni_parent_prev", "equity_parent_prev"]].copy()
    prior["period_end"] = prior["period_end"] - pd.DateOffset(years=1)
    prior = prior.rename(columns={"ni_parent_prev": "ni_p", "equity_parent_prev": "eq_end_p"})
    fy = own.merge(prior, on=["ticker", "period_end"], how="outer")
    fy["ni"] = fy["ni"].combine_first(fy["ni_p"])
    fy["eq_end"] = fy["eq_end"].combine_first(fy["eq_end_p"])
    fy = fy.sort_values(["ticker", "period_end"])
    prev_end = fy.groupby("ticker")["eq_end"].shift(1)
    gap = fy["period_end"] - fy.groupby("ticker")["period_end"].shift(1)
    fy["eq_beg"] = fy["eq_beg"].combine_first(prev_end.where(gap.dt.days.between(330, 400)))
    return fy[["ticker", "period_end", "ni", "eq_end", "eq_beg"]]


def asof_fundamentals(rep: pd.DataFrame, d: pd.Timestamp, max_age_days: int = 200) -> pd.DataFrame:
    """Per ticker, figures known on `d` (avail <= d): latest balance sheet, TTM flows
    from the latest report, and the 3:2:1-weighted ROE of the latest three fiscal
    years (controlling net income / average controlling equity)."""
    known = rep[rep["avail"] <= d]
    latest = known.sort_values(["ticker", "period_end", "avail"]).groupby("ticker").tail(1)
    latest = latest[(d - latest["period_end"]).dt.days <= max_age_days].set_index("ticker")
    out = latest[["period_end", "report", "avail", "avail_src", "name", "industry", "market", "consolidated",
                  "current_assets", "liabilities", "equity_parent", "cash", "debt"]].copy()

    for item in ("ni_parent", "cfo", "revenue", "op_income"):
        cur = latest[item]
        is_ann = latest["report"] == "annual"
        # previous fiscal year's annual value: the latest annual report ending
        # 9-15 months before this quarter's period end
        prev_fy = pd.Series(np.nan, index=latest.index)
        q = latest[~is_ann]
        if not q.empty:
            a = known[known["report"] == "annual"][["ticker", "period_end", item]].rename(
                columns={"period_end": "fy_end", item: "fy_val"})
            m = q[["period_end"]].reset_index().merge(a, on="ticker", how="left")
            lag = (m["period_end"] - m["fy_end"]).dt.days
            m = m[lag.between(80, 300)].sort_values("fy_end").groupby("ticker").tail(1).set_index("ticker")
            prev_fy.loc[m.index] = m["fy_val"]
        pcum = latest[f"{item}_pcum"] if f"{item}_pcum" in latest else pd.Series(np.nan, index=latest.index)
        if pcum.isna().any():
            # fallback: last year's same report
            same = known[["ticker", "period_end", "report", item]].copy()
            same["period_end"] = same["period_end"] + pd.DateOffset(years=1)
            same = same.set_index(["ticker", "period_end", "report"])[item]
            key = pd.MultiIndex.from_arrays([latest.index, latest["period_end"], latest["report"]])
            alt = pd.Series(same.reindex(key).to_numpy(), index=latest.index)
            pcum = pcum.combine_first(alt)
        out[f"ttm_{item}"] = np.where(is_ann, cur, cur + prev_fy - pcum)

    fy = _fy_table(known)
    fy = fy[fy["ticker"].isin(out.index)]
    roe = {}
    for tk, g in fy.groupby("ticker"):
        g = g.dropna(subset=["ni", "eq_end"]).tail(3)
        if len(g) < 3 or (d - g["period_end"].iloc[-1]).days > 480:
            continue
        gaps = g["period_end"].diff().dt.days.iloc[1:]
        if not gaps.between(330, 400).all():
            continue
        avg = np.where(g["eq_beg"].notna(), (g["eq_beg"] + g["eq_end"]) / 2.0, g["eq_end"])
        if (avg <= 0).any():
            continue
        r = g["ni"].to_numpy() / avg
        roe[tk] = (r[0] * 1 + r[1] * 2 + r[2] * 3) / 6.0
    out["roe3"] = pd.Series(roe)
    return out


def statements_long() -> pd.DataFrame:
    """Every bulk report period stacked (cached)."""
    path = CACHE / "bulk_long.pkl"
    if path.exists():
        return pd.read_pickle(path)
    dart_bulk.download_all()
    files = sorted({tuple(p.name.split("_")[:2]) for p in dart_bulk.CACHE.glob("*.zip")})
    parts = []
    for year, q in files:
        df = parse_period(year, q)
        print(f"  parsed {year} {q}: {len(df):,} rows")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    pd.to_pickle(df, path)
    return df
