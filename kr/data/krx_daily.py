"""
Daily cross-sectional KRX snapshots (one request per market per day, so names
that were later delisted are included automatically):

  prices      [12001] 전종목 시세: close, 등락률, open, traded value, market cap,
              listed shares and the KOSDAQ section (소속부: 관리종목 / SPAC / 외국기업 ...)
  fundamental [12021] PER/PBR/배당수익률: BPS, PER, PBR, EPS, DIV, DPS (rebalance dates only)

Needs a KRX data-portal login (KRX_ID / KRX_PW in .env); pykrx re-logs in when
the one-hour session expires. Login messages are swallowed so credentials never
reach stdout. Cached per day under data/cache/kr/daily/.
"""

from __future__ import annotations

import contextlib
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from kr.data import krx

CACHE = krx.CACHE_DIR / "daily"
MARKETS = {"STK": "KOSPI", "KSQ": "KOSDAQ"}
PRICE_COLS = {"ISU_SRT_CD": "ticker", "ISU_ABBRV": "name", "SECT_TP_NM": "section", "TDD_CLSPRC": "close",
              "FLUC_RT": "chg_pct", "TDD_OPNPRC": "open", "ACC_TRDVOL": "volume", "ACC_TRDVAL": "trdval",
              "MKTCAP": "mcap", "LIST_SHRS": "shares"}
NUMERIC = ["close", "chg_pct", "open", "volume", "trdval", "mcap", "shares"]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


_POOL = ThreadPoolExecutor(max_workers=4)


def _fetch_prices(day: str, mkt: str) -> pd.DataFrame:
    with _quiet():
        krx._stock()
        from pykrx.website.krx.market.core import 전종목시세
        for attempt in range(6):
            try:
                # pykrx sets no socket timeout; a stalled read is abandoned after 60s
                df = _POOL.submit(전종목시세().fetch, day, mkt).result(timeout=60)
                break
            except Exception:  # transient network / session errors, timeouts
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"KRX price snapshot failed: {day} {mkt}")
    if df.empty:
        return pd.DataFrame(columns=list(PRICE_COLS.values()))
    df = df[list(PRICE_COLS)].rename(columns=PRICE_COLS)
    for c in NUMERIC:
        df[c] = _num(df[c])
    return df


def trading_days(start: str, end: str | None = None) -> pd.DatetimeIndex:
    """Sessions from the KODEX 200 (069500) price history."""
    return benchmark_prices(start, end).index


def benchmark_prices(start: str, end: str | None = None, ticker: str = "069500") -> pd.DataFrame:
    end_d = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y%m%d")
    path = CACHE / f"etf_{ticker}_{pd.Timestamp(start):%Y%m%d}_{end_d}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    with _quiet():
        df = krx._stock().get_etf_ohlcv_by_date(pd.Timestamp(start).strftime("%Y%m%d"), end_d, ticker)
    df.index = pd.to_datetime(df.index)
    df = df[df["종가"] > 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def fetch_prices(days: pd.DatetimeIndex, verbose: bool = True) -> None:
    """Resumable: skips days already cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days for m in MARKETS if not (CACHE / f"px_{d:%Y%m%d}_{m}.csv").exists()]
    if verbose:
        print(f"KRX price snapshots to fetch: {len(todo)}")
    for i, d in enumerate(days):
        for m in MARKETS:
            path = CACHE / f"px_{d:%Y%m%d}_{m}.csv"
            if path.exists():
                continue
            df = _fetch_prices(d.strftime("%Y%m%d"), m)
            if df.empty:
                raise RuntimeError(f"empty KRX snapshot on session {d.date()} {m}")
            df.to_csv(path, index=False)
            time.sleep(0.3)
        if verbose and i % 100 == 0:
            print(f"  {d.date()} done")


def fundamental_snapshot(day: pd.Timestamp) -> pd.DataFrame:
    """KRX BPS/PER/PBR/EPS/DIV/DPS for KOSPI+KOSDAQ on `day` (cached)."""
    path = CACHE / f"fund_{day:%Y%m%d}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"ticker": str})
    out = []
    with _quiet():
        stock = krx._stock()
        for m in MARKETS.values():
            df = stock.get_market_fundamental_by_ticker(day.strftime("%Y%m%d"), m)
            df = df.reset_index().rename(columns={df.index.name or "index": "ticker", "티커": "ticker"})
            df["market"] = m
            out.append(df)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_panel(days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) frames: close, ret, mcap, trdval, open, shares, plus
    market / section / name of the latest appearance. A ticker absent on a day
    was not listed (or not yet / no longer) that day."""
    pkl = CACHE / f"panel_{days[0]:%Y%m%d}_{days[-1]:%Y%m%d}.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    frames = []
    for d in days:
        for m, label in MARKETS.items():
            df = pd.read_csv(CACHE / f"px_{d:%Y%m%d}_{m}.csv", dtype={"ticker": str, "section": str})
            df["date"], df["market"] = d, label
            frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["section"] = long["section"].fillna("")
    out = {c: long.pivot(index="date", columns="ticker", values=c).reindex(days)
           for c in ("close", "chg_pct", "open", "trdval", "mcap", "shares")}
    out["market"] = long.pivot(index="date", columns="ticker", values="market").reindex(days)
    out["section"] = long.pivot(index="date", columns="ticker", values="section").reindex(days)
    out["name"] = long.pivot(index="date", columns="ticker", values="name").reindex(days)
    close = out["close"]
    simple = close / close.shift(1) - 1.0
    krx_ret = out["chg_pct"] / 100.0
    # close-to-close is exact on normal days; on corporate-action days (split,
    # rights, bonus) KRX's 등락률 is measured against the adjusted base price
    use_krx = (simple - krx_ret).abs() > 0.01
    ret = simple.where(~use_krx, krx_ret)
    # A reverse split whose base price KRX did not adjust (e.g. 052670 2026-02-09:
    # 등락률 +29,948% while market cap fell 80%): listed shares fall by more than
    # half the same day and the return still exceeds +100%; the market-cap ratio
    # is the economic return there. Share counts are not used otherwise: new
    # shares of bonus / rights issues list weeks after the ex-date.
    share_ratio = out["shares"] / out["shares"].shift(1)
    mcap_ret = out["mcap"] / out["mcap"].shift(1) - 1.0
    broken = (share_ratio < 0.5) & (ret > 1.0)
    ret = ret.where(~broken, mcap_ret)
    out["n_broken_base"] = int(broken.sum().sum())
    ret[close.shift(1).isna() & close.notna()] = np.nan  # first listed day
    out["ret"] = ret
    pd.to_pickle(out, pkl)
    return out
