"""Korean market settings. Secrets (DART_API_KEY, KRX_ID, KRX_PW) live in .env."""

START = "2018-01-01"
END = "2026-10-14"  # last session of the recorded runs (None = latest)
INDEX_CODE = "1028"  # KRX KOSPI 200
SPLIT = "2023-05-10"  # same train / held-out boundary as the US work

# Costs: commission + slippage per unit turnover; securities transaction tax on sells only
COMMISSION_BPS = 1.5
SLIPPAGE_BPS = 20.0
# Sell-side tax by trade date. KOSPI = transaction tax + 0.15% special rural tax, KOSDAQ =
# transaction tax only; the totals have been equal in every period, so one schedule covers both.
# Year-start changes legally apply by settlement date (T+2); trades in the last two December
# sessions are approximated with the old rate (books only rebalance in early January).
SELL_TAX_SCHEDULE = [
    ("2018-01-01", 0.0030),
    ("2019-05-30", 0.0025),
    ("2021-01-01", 0.0023),
    ("2023-01-01", 0.0020),
    ("2024-01-01", 0.0018),
    ("2025-01-01", 0.0015),
    ("2026-01-01", 0.0020),
]
BORROW_FEE_ANNUAL = 0.02

# New short sales were not possible in these windows (existing shorts could be covered).
# Between them only KOSPI200 / KOSDAQ150 names could be shorted, which covers this universe.
SHORT_BAN_PERIODS = [
    ("2020-03-16", "2021-05-02"),
    ("2023-11-06", "2025-03-30"),
]
