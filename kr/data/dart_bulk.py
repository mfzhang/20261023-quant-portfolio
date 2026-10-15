"""
OpenDART 재무정보 일괄다운로드 (bulk XBRL financial statements, tab-separated
text per report period, all listed filers, 2015 FY onward). One zip per
(business year, report, statement): BS / PL / CF / CE, each holding separate
and consolidated (연결) files. No API key or request quota; cached under
data/cache/kr/dart/bulk/.

The files carry the fiscal period end (결산기준일) and report type but no
receipt date; pit_statements() joins receipt dates from the DART filing list.
Values are as of the file's generation date, so later amendments (정정) replace
the originally filed numbers.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT_URL = "https://opendart.fss.or.kr"
MAIN = f"{ROOT_URL}/disclosureinfo/fnltt/dwld/main.do"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "kr" / "dart" / "bulk"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.get(MAIN, timeout=30)
    return s


def file_list(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or _session()
    html = s.post(f"{ROOT_URL}/disclosureinfo/fnltt/dwld/list.do", timeout=30,
                  headers={"Referer": MAIN, "X-Requested-With": "XMLHttpRequest"}).text
    calls = re.findall(r"download_ext002\('(\d+)','(\w+)', '(\w+)', '([^']+)'\)", html)
    return pd.DataFrame(calls, columns=["year", "period", "stmt", "fname"])


def _cached(fname: str) -> Path | None:
    prefix = "_".join(fname.split("_")[:3]) + "_"
    hits = sorted(CACHE.glob(prefix + "*.zip"))
    return hits[-1] if hits else None


def download_all(stmts: tuple[str, ...] = ("BS", "PL", "CF")) -> list[Path]:
    """Fetch every listed zip not yet cached (any generation date counts as cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    s = _session()
    files = file_list(s)
    paths = []
    for row in files[files["stmt"].isin(stmts)].itertuples():
        hit = _cached(row.fname)
        if hit is None:
            for attempt in range(3):
                r = s.get(f"{ROOT_URL}/cmm/downloadFnlttZip.do?fl_nm={row.fname}", timeout=600,
                          headers={"Referer": MAIN})
                if r.content[:2] == b"PK":
                    break
                time.sleep(5 * (attempt + 1))
                s = _session()
            else:
                raise RuntimeError(f"bulk download failed: {row.fname}")
            hit = CACHE / row.fname
            hit.write_bytes(r.content)
            print(f"  downloaded {row.fname} ({len(r.content) / 1e6:.1f} MB)")
            time.sleep(1.0)
        paths.append(hit)
    return paths


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


FIN_FORMATS = ("금융기타", "보험", "은행", "증권")  # financial-company statement layouts


def _value_columns(cols: list[str]) -> tuple[str | None, str | None, str | None, str | None]:
    """Headers of (current cumulative / point value, current 3-month value, prior
    fiscal year / prior year-end value, prior-year same-period cumulative value).
    Quarterly PL: '당기 1분기 3개월', '당기 1분기 누적', '전기 1분기 3개월', '전기 1분기 누적',
    '전기'; CF: '당기반기', '전기반기', '전기'; BS: '당기 3분기말', '전기말'; annual: '당기', '전기'."""
    cur = [c for c in cols if c.startswith("당기")]
    main = next((c for c in cur if "누적" in c), None) or next((c for c in cur if "3개월" not in c), None)
    three = next((c for c in cur if "3개월" in c), None)
    prev = next((c for c in cols if c in ("전기", "전기말")), None)
    past = [c for c in cols if c.startswith("전기") and c not in ("전기", "전기말") and "3개월" not in c]
    prev_cum = next((c for c in past if "누적" in c), None) or (past[0] if past else None)
    return main, three, prev, prev_cum


def read_zip(path: Path) -> pd.DataFrame:
    """Non-financial statement files of one zip, stacked, with value columns
    normalized to v_main / v_3m / v_prev. Adds 'consolidated' (bool)."""
    out = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename if info.flag_bits & 0x800 else \
                info.filename.encode("cp437", errors="ignore").decode("cp949", errors="ignore")
            if any(f in name for f in FIN_FORMATS):
                continue
            text = z.read(info).decode("cp949", errors="replace")
            df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False,
                             on_bad_lines="skip", engine="python", quoting=3)
            df.columns = [c.strip() for c in df.columns]
            main, three, prev, prev_cum = _value_columns(list(df.columns))
            df["v_main"] = _num(df[main]) if main else np.nan
            df["v_3m"] = _num(df[three]) if three else np.nan
            df["v_prev"] = _num(df[prev]) if prev else np.nan
            df["v_prev_cum"] = _num(df[prev_cum]) if prev_cum else np.nan
            keep = [c for c in ID_COLS if c in df.columns] + ["v_main", "v_3m", "v_prev", "v_prev_cum"]
            df = df[keep].copy()
            df["consolidated"] = "연결" in name
            df["file"] = name
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


ID_COLS = ["재무제표종류", "종목코드", "회사명", "시장구분", "업종", "업종명", "결산월", "결산기준일", "보고서종류", "통화",
           "항목코드", "항목명"]
