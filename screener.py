import os
import json
import time
import math
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==========================================
# Environment Variables & Risk Management
# ==========================================
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = str(os.getenv("CRYPTOCOMPARE_API_KEY") or "").strip()
# [#5] Financial Modeling Prep API key — used for the macro economic calendar
# (FOMC / CPI / NFP, etc.). Free tier: https://site.financialmodelingprep.com/
FMP_API_KEY = str(os.getenv("FMP_API_KEY") or "").strip()

PROXY_URL = os.getenv("PROXY_URL", "").strip()
PROXIES   = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

PORTFOLIO_USDT     = 1500.0
RISK_PER_TRADE_PCT = 2.0
MAX_TOTAL_RISK_PCT = 6.0

# ------------------------------------------
# [Portfolio] Position Count & Correlation Caps
# ------------------------------------------
# Hard cap on number of concurrently open positions, regardless of how
# small individual risk % looks. Prevents over-diversification into many
# correlated alts during a single market regime.
MAX_CONCURRENT_POSITIONS = 5
# If a new candidate's correlation to BTC is >= this threshold AND there
# are already this many highly-correlated positions open, skip the new
# signal entirely (concentration risk), even if its score qualifies.
HIGH_CORR_BTC_THRESHOLD = 0.75
MAX_HIGH_CORR_POSITIONS = 3

COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "FLOKI", "SHIB", "EIGEN", "OP", "DOGE", "NEAR",
    "TRX", "AVAX", "SUI",
]
# WATCHLIST = alias ของ COINS — ใช้ชื่อนี้ใน scan_market เพื่อความชัดเจน
WATCHLIST = COINS

# ==========================================
# Constants & Hyperparameters
# ==========================================
API_RATE_LIMIT_DELAY = 0.3    # Binance public: 1200 req/min — 0.3s ปลอดภัยและเร็วกว่ามาก
API_MAX_RETRIES      = 3
BINANCE_LIMIT        = 500
CC_HISTOHOUR_LIMIT   = 2000
CC_HISTODAY_LIMIT    = 500
CG_OHLC_DAYS_4H      = 90
CG_OHLC_DAYS_1D      = 365
CACHE_TTL_SECONDS    = 3600
DATA_STALENESS_WARN_HOURS  = 5.0
DATA_STALENESS_SKIP_HOURS  = 10.0
# TF-aware aliases — code ใช้ชื่อ _4H/_1D; aliases defined here
DATA_STALENESS_WARN_HOURS_4H = DATA_STALENESS_WARN_HOURS
DATA_STALENESS_SKIP_HOURS_4H = DATA_STALENESS_SKIP_HOURS
DATA_STALENESS_WARN_HOURS_1D = 26.0   # 1D candle เปิดอยู่ตลอดวัน (0-24h ปกติ)
DATA_STALENESS_SKIP_HOURS_1D = 50.0   # skip ถ้าเก่ากว่า 50h (เก่า 2 วัน)

# Time-based stop: if a setup hasn't progressed within this many 4H bars
# (approx. days), flag it as "stale" in the trend/exit narrative.
TIME_STOP_BARS_4H = 42   # ~7 days on 4H candles

# Position state persistence (JSON file committed back to repo via CI)
POSITIONS_FILE = os.getenv("POSITIONS_FILE", "positions.json")
# [Ledger] Append-only log of realized PnL events (partial closes, full
# closes) for later performance evaluation. JSON Lines format (one JSON
# object per line) so it can be appended without re-parsing the whole file.
TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trade_log.jsonl")
# Time-based stop for an actual open position: if it hasn't hit TP1 or SL
# within this many hours, surface a "พิจารณาปิด" warning regardless of score.
POSITION_TIME_STOP_HOURS = 7 * 24  # 7 days
# [Fix I] SL Proximity Warning: แจ้งเตือนเมื่อราคาเข้าใกล้ SL
# threshold คือ % ของ entry price — ถ้าราคาห่างจาก SL น้อยกว่านี้ → เตือน
SL_PROXIMITY_WARNING_PCT = 1.5  # เตือนเมื่อราคาห่างจาก SL น้อยกว่า 1.5% ของ entry

# [ATR-based EMA200 Proximity] ใช้ ATR แทน % คงที่ เพราะ volatility ต่างกันมากระหว่างเหรียญ
# ถ้าราคาห่าง EMA200 น้อยกว่า threshold นี้ × ATR -> ถือว่า "ใกล้เส้น" ไม่ใช่ "หลุดจริง"
# อ้างอิงกรอบ technical analysis ทั่วไป:
#   0   - 0.5x ATR = แตะเส้น (ตัดสินไม่ได้ชัดเจน)
#   0.5 - 1.5x ATR = ใกล้เส้น (รอ confirm)
#   1.5 - 3x   ATR = ห่างปานกลาง (trend ชัดขึ้น)
#   > 3x       ATR = ห่างมาก (trend ชัดเจนมาก)
EMA200_PROXIMITY_ATR_MULTIPLIER = 0.75  # ค่ากึ่งกลางที่สมดุล
# Trailing stop: once price has moved this fraction of the way from entry
# to TP1, move SL up to break-even (entry price).
BREAKEVEN_TRIGGER_PCT = 0.5  # 50% of the way to TP1
# Once TP1 is hit, trail SL using this ATR multiple below current price.
TRAIL_ATR_MULTIPLIER = 1.5

# ------------------------------------------
# [#1] ATR-Based Take Profit
# ------------------------------------------
# TP1/TP2 are now expressed as multiples of ATR (at signal time) added to
# entry price, instead of fixed percentages. The tier-based dynamic ATR
# multiplier (get_dynamic_atr_multiplier) still scales these based on
# ADX/volatility regime, so a "major" coin in a low-ADX/low-vol regime gets
# tighter TPs than a "small" coin in a high-ADX/high-vol breakout.
ATR_TP1_MULTIPLIER = 1.5
ATR_TP2_MULTIPLIER = 3.0
# Minimum SL distance as a fraction of price, to avoid unrealistically tight
# stops when ATR is very small relative to price (e.g. low-vol majors).
MIN_SL_DISTANCE_PCT = 0.01

# ------------------------------------------
# [#2] Partial Take Profit / Runner
# ------------------------------------------
# Fraction of the position to (notionally) close when TP1 is hit. The
# remainder ("runner") stays open and is managed by the trailing stop.
PARTIAL_TP1_CLOSE_PCT = 0.5
# Once the runner phase begins (after partial TP1), trail tighter than the
# initial breakeven-trail phase to lock in more of the move.
RUNNER_TRAIL_ATR_MULTIPLIER = 1.0

# ------------------------------------------
# [Multi-Stage TP] TP1.5 — intermediate partial close
# ------------------------------------------
# A second partial close between TP1 and TP2, expressed as an ATR multiple
# (like TP1/TP2). Hit between TP1 and TP2: lock in more profit before the
# final runner phase, reducing give-back if price reverses hard after TP1.
ATR_TP1_5_MULTIPLIER = 2.25   # halfway between ATR_TP1_MULTIPLIER and ATR_TP2_MULTIPLIER
TP1_5_CLOSE_PCT = 0.25        # close another 25% of ORIGINAL size at TP1.5

# ------------------------------------------
# [Runner Extension] Trend-strength-based trail widening
# ------------------------------------------
# While in the runner phase, if ADX is still rising AND price keeps making
# higher highs (new high-water-mark each bar), temporarily WIDEN the trail
# instead of always tightening — lets a strong trend "breathe".
RUNNER_EXTEND_ATR_MULTIPLIER = 2.0   # widened trail multiple during strong continuation
RUNNER_EXTEND_MIN_ADX = 25            # ADX must be at/above this to qualify
RUNNER_EXTEND_ADX_RISING_LOOKBACK = 3 # bars to confirm ADX is rising

# ------------------------------------------
# [Give-back Exit] High-Water-Mark based early exit
# ------------------------------------------
# Track the highest price seen since entry (per position). If price falls
# back from that peak by more than GIVE_BACK_EXIT_PCT of the total gain
# achieved (peak vs entry), and we're still in the pre-TP1 phase, treat it
# as an early-warning exit signal (separate from the hard ATR-based SL).
GIVE_BACK_EXIT_PCT = 0.50  # if 50%+ of the peak gain (since entry) is given back -> warn/exit
# Minimum peak gain (in % from entry) required before give-back logic
# activates — avoids triggering on tiny noise near entry.
GIVE_BACK_MIN_PEAK_GAIN_PCT = 1.5

# ------------------------------------------
# [Volatility-Adjusted SL] Widen SL in choppy/low-ADX regimes
# ------------------------------------------
# In low-ADX (sideways/choppy) regimes, tight ATR-based stops get whipsawed
# before the real move develops. When ADX < this threshold at entry, widen
# the SL distance by the given multiplier (and the position-size calc will
# naturally shrink size to compensate, keeping $ risk constant).
CHOPPY_ADX_THRESHOLD = 15
CHOPPY_SL_WIDEN_MULTIPLIER = 1.5

# ------------------------------------------
# [Entry Filter] 1D Trend Alignment (anti bag-holding)
# ------------------------------------------
# Require daily-timeframe trend confirmation, not just 4H > EMA200. A coin
# can look "bullish" on 4H while the daily trend is rolling over — this
# filters out those "dip that becomes a downtrend" traps.
REQUIRE_1D_EMA50_ALIGNMENT = True
EMA_SHORT_1D = 50  # 1D EMA period used for the daily trend check

# ------------------------------------------
# [Entry Filter] 24h Volume Floor (liquidity filter)
# ------------------------------------------
# Minimum 24h USDT quote volume (from Binance ticker) required for a coin to
# be eligible for new signals. Filters out illiquid alts where technical
# setups look clean but have no real follow-through.
MIN_24H_QUOTE_VOLUME_USDT = 5_000_000.0
# เหรียญ WATCHLIST ที่รู้ว่ามี volume สูงจริงๆ — ยกเว้นจาก volume filter
# เพราะ Binance 24h endpoint บางครั้ง return ค่าผิดปกติสำหรับบาง symbol
VOLUME_FILTER_WHITELIST = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
                           "SHIB", "TRX", "AVAX", "NEAR", "OP", "SUI", "FLOKI"}

# ------------------------------------------
# [Entry Filter] Re-entry Cooldown
# ------------------------------------------
# After a position is closed via SL (full close, before TP1), don't allow a
# new signal on the same coin for this many hours — prevents repeatedly
# buying the same falling knife if technical conditions re-qualify quickly.
REENTRY_COOLDOWN_HOURS = 24
COOLDOWN_FILE = os.getenv("COOLDOWN_FILE", "reentry_cooldown.json")

# ------------------------------------------
# [#4] BTC Dominance (BTC.D) Filter
# ------------------------------------------
# BTC.D rising sharply usually means capital is rotating OUT of altcoins and
# INTO BTC (or stablecoins) — altcoin longs are higher-risk in this regime.
# We track BTC.D % change over a short lookback window using CoinGecko's
# /api/v3/global endpoint (1 call per run, cached like other globals).
BTC_DOMINANCE_LOOKBACK_SNAPSHOTS = 6   # cached snapshots (~ spans CACHE_TTL_SECONDS * N runs)
BTC_DOMINANCE_RISING_THRESHOLD_PCT = 1.0   # BTC.D up >1% over lookback -> warn/penalize alts
BTC_DOMINANCE_FALLING_THRESHOLD_PCT = -1.0  # BTC.D down >1% -> alt-friendly, small bonus
BTC_DOMINANCE_HISTORY_FILE = os.getenv("BTC_DOMINANCE_HISTORY_FILE", "btc_dominance_history.json")

# ------------------------------------------
# [#5] Macro / Economic News Filter
# ------------------------------------------
# Uses Financial Modeling Prep's economic calendar (free tier) to detect
# high-impact USD macro events (FOMC rate decisions, CPI, NFP, etc.) on the
# current UTC date. On flagged days, new entry signals are suppressed (or
# flagged with a warning) to avoid getting chopped around a volatility spike.
MACRO_FILTER_ENABLED = bool(FMP_API_KEY)
# Keywords (case-insensitive substring match) identifying high-impact events
# worth reacting to. FMP's "event" field text varies, so we match broadly.
MACRO_HIGH_IMPACT_KEYWORDS = [
    "fomc", "fed interest rate", "federal funds rate", "interest rate decision",
    "cpi", "consumer price index", "non farm payroll", "nonfarm payroll", "nfp",
    "ppi", "producer price index", "core pce", "pce price index",
    "gdp", "fed chair", "powell",
]
# Only consider USD-denominated events (crypto trades vs USD liquidity regime)
MACRO_FILTER_COUNTRY = "US"
# Block brand-new entries if a high-impact event is scheduled within this many
# hours from "now" (before or after) on the current UTC day.
MACRO_BLOCK_WINDOW_HOURS = 3

# ------------------------------------------
# [#3] Funding Rate / Open Interest Filters
# ------------------------------------------
# Binance Futures perpetual funding rate is paid every 8h. A long-biased
# funding rate above this threshold means longs are paying a meaningful
# premium to shorts -> market is crowded long -> new long signals are
# suppressed (not blocked entirely, but flagged + score-penalized).
FUNDING_RATE_MAX_LONG = 0.0005   # 0.05% per 8h
FUNDING_RATE_WARN_LONG = 0.0003  # 0.03% per 8h -> warn but don't block
# Open Interest history lookback (number of 5m OI snapshots from Binance's
# /futures/data/openInterestHist, period=4h aligns roughly with our 4H bars).
OI_HIST_PERIOD = "4h"
OI_HIST_LIMIT  = 7     # ~ last 28h of 4H OI snapshots
# If price rose but OI did NOT grow with it (or fell), the rally may be
# short-covering / low-conviction rather than fresh long positioning.
OI_PRICE_UP_NO_OI_GROWTH_PCT = 1.0  # OI growth below this % while price up -> warn

# [B] Binance Endpoints
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

# [B-Futures] Binance Futures (USDT-M) Endpoints - for funding rate / OI
BINANCE_FUTURES_ENDPOINTS = [
    "https://fapi.binance.com",
]

# [C] CoinGecko Coin ID Map
COINGECKO_IDS = {
    "BTC":   "bitcoin",       "ETH":   "ethereum",
    "BNB":   "binancecoin",   "SOL":   "solana",
    "XRP":   "ripple",        "ADA":   "cardano",
    "DOGE":  "dogecoin",      "SHIB":  "shiba-inu",
    "AVAX":  "avalanche-2",   "TRX":   "tron",
    "NEAR":  "near",          "OP":    "optimism",
    "SUI":   "sui",           "FLOKI": "floki",
    "EIGEN": "eigenlayer",
}

# --- Indicators & Logic Constants ---
RSI_PERIOD     = 14
EMA_SHORT      = 50
EMA_LONG       = 200
RSI_OVERSOLD   = 32
RSI_OVERBOUGHT = 70
ATR_PERIOD     = 14

RSI_RECOVERY_THRESHOLD = 45
RSI_PULLBACK_THRESHOLD = 55
RSI_RECOVERY_LOOKBACK  = 5

RSI_BULL_DIV_MAX   = 45
RSI_BEAR_DIV_MIN   = 55
LOOKBACK_BARS      = 15
LOOKBACK_SKIP_BARS = 3

TREND_SLOPE_BARS      = 5
TREND_MIN_CONSECUTIVE = 3

RSI_BOUNCE_CONFIRM_BARS = 2
RSI_BOUNCE_MIN_RISE     = 3.0

OB_LOOKBACK        = 20
OB_IMBALANCE_RATIO = 1.5
FVG_THRESHOLD_PCT  = 0.2
MINIMUM_SIGNAL_SCORE = 50

# Exit-side score threshold: minimum exit-warning score to surface a coin
# in the "พิจารณาปิดสถานะ" (exit watch) section.
MINIMUM_EXIT_SCORE = 40

# RSI Hard Override thresholds สำหรับ Exit Watch
# RSI >= RSI_EXIT_HARD_OVERRIDE (78) -> force เข้า exit watch ทันทีโดยไม่ต้องรอ score
RSI_EXIT_HARD_OVERRIDE = 78
# RSI >= RSI_EXIT_WARN_THRESHOLD (72) -> ลด effective threshold เป็น 25 แทน 40
# ตั้ง 72 ไม่ใช่ 70 เพื่อให้มี gap ระหว่าง RSI_OVERBOUGHT(70) กับ soft-override:
#   RSI 70-71: ได้คะแนน 28 + threshold ยังเป็น 40 → ต้องมี factor อื่นช่วย
#   RSI 72-77: ลด threshold เป็น 25 → ผ่านด้วย RSI เดี่ยวได้
#   RSI 78+:   hard override → force ผ่านเสมอ
RSI_EXIT_WARN_THRESHOLD = 72

# รายชื่อเหรียญที่รองรับการดึง On-chain (Blockchain Histo) จาก CryptoCompare
ONCHAIN_SUPPORTED_COINS = {"BTC", "ETH", "ADA", "DOGE", "LTC", "BCH", "LINK"}

# NOTE: TP_TIERS percentages are now only used as a FALLBACK if ATR is
# unavailable/NaN at signal time (e.g. very short data history).
TP_TIERS = {
    "major": {"tp1": 0.10, "tp2": 0.15, "sl_buffer": 0.025},
    "mid":   {"tp1": 0.15, "tp2": 0.20, "sl_buffer": 0.050},
    "small": {"tp1": 0.20, "tp2": 0.35, "sl_buffer": 0.080},
}

COIN_TIER = {
    "BTC": "major", "ETH": "major",
    "BNB": "mid",   "SOL": "mid",   "XRP":  "mid",
    "ADA": "mid",   "NEAR": "mid",  "OP":   "mid",
    "TRX": "mid",   "AVAX": "mid",
    "FLOKI": "small", "SHIB": "small", "EIGEN": "small",
    "DOGE":  "small", "SUI":  "small",
}

# ==========================================
# Global API Session
# ==========================================
api_session = requests.Session()
api_session.headers.update({
    "User-Agent": "CryptoScreenerBot/5.0 (ATR-TP + Partial/Runner + Funding/OI Filters)",
    "Accept":     "application/json",
})

if PROXIES:
    api_session.proxies.update(PROXIES)

retry_strategy = Retry(
    total=1,                   # ลดจาก 3 → 1 (เรามี multi-endpoint fallback ของตัวเองอยู่แล้ว)
    backoff_factor=0.5,        # ลดจาก 2.0 → 0.5 (worst case retry = 0.5s แทน 14s)
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
api_session.mount("https://", adapter)
api_session.mount("http://", adapter)

# ==========================================
# In-Memory Cache
# ==========================================
_cache_4h:        dict  = {}
_cache_1d:        dict  = {}
_cache_1h:        dict  = {}   # [Approach A] 1H RSI for entry timing
_cache_onchain:   dict  = {}
_cache_funding:   dict  = {}
_cache_btcd:      dict  = {}
_cache_oi_run:    dict  = {}
_cache_macro:     dict  = {}
_cache_24hvol:    dict  = {}
_cache_ts_4h:     float = 0.0
_cache_ts_1d:     float = 0.0
_cache_ts_1h:     float = 0.0
_cache_ts_onchain: float = 0.0
_cache_ts_funding: float = 0.0
_cache_ts_btcd:    float = 0.0
_cache_ts_macro:   float = 0.0
_cache_ts_24hvol:  float = 0.0

# ==========================================
# Telegram
# ==========================================
def send_telegram_messages(chunks: list) -> None:
    token   = str(TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = str(TELEGRAM_CHAT_ID   or "").strip()
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID ไม่ได้ตั้งค่า")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for idx, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            continue
        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}
        try:
            resp = api_session.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Telegram ส่งสำเร็จ (ส่วนที่ {idx}/{len(chunks)})")
            else:
                logger.warning(f"Telegram ล้มเหลว (ส่วนที่ {idx}): {resp.text}")
        except Exception as e:
            logger.error(f"Telegram error (ส่วนที่ {idx}): {e}")
        if idx < len(chunks):
            time.sleep(0.5)

# ==========================================
# Position State Persistence (positions.json)
# ==========================================
def load_positions() -> dict:
    """
    โครงสร้าง positions.json (v9):
    {
      "BTC": {
        "entry_price": 65000.0,
        "entry_time": "2026-06-10T12:00:00+00:00",
        "sl": 63000.0, "original_sl": 63000.0,
        "tp1": 70000.0, "tp2": 73000.0,
        "tier": "major", "atr_at_entry": 1234.5,
        "tp1_hit": false, "tp1_5_hit": false,
        "partial_closed": false, "remaining_size_pct": 1.0,
        "high_water_mark": 65000.0, "status": "open"
      }
    }

    [Migration Guard] positions ที่สร้างจาก version ก่อน v9 อาจขาด fields ใหม่
    เช่น original_sl, tp1_5_hit, high_water_mark — ใส่ค่า default ให้อัตโนมัติ
    """
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}

        # Migration: ใส่ default ให้ fields ที่ version เก่าไม่มี
        for coin, pos in data.items():
            if not isinstance(pos, dict):
                continue
            # v9 new fields
            if "original_sl" not in pos:
                pos["original_sl"] = pos.get("sl", 0.0)
            if "tp1_5_hit" not in pos:
                pos["tp1_5_hit"] = False
            if "high_water_mark" not in pos:
                pos["high_water_mark"] = pos.get("entry_price", 0.0)
            if "remaining_size_pct" not in pos:
                pos["remaining_size_pct"] = 0.5 if pos.get("tp1_hit") else 1.0
            if "partial_closed" not in pos:
                pos["partial_closed"] = pos.get("tp1_hit", False)
            if "atr_at_entry" not in pos:
                pos["atr_at_entry"] = None  # unknown for legacy positions

        return data
    except Exception as e:
        logger.warning(f"⚠️ อ่าน {POSITIONS_FILE} ไม่ได้: {e}")
        return {}

def save_positions(positions: dict) -> None:
    try:
        with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 บันทึก {POSITIONS_FILE} สำเร็จ ({len(positions)} positions)")
    except Exception as e:
        logger.error(f"❌ บันทึก {POSITIONS_FILE} ล้มเหลว: {e}")

# ==========================================
# [Entry Filter] Re-entry Cooldown Persistence
# ==========================================
def load_cooldowns() -> dict:
    """
    โครงสร้าง reentry_cooldown.json: {"COIN": "2026-06-13T12:00:00+00:00", ...}
    เก็บเวลาที่ position ของเหรียญนั้นถูกปิดด้วย close_sl ครั้งล่าสุด
    """
    if not os.path.exists(COOLDOWN_FILE):
        return {}
    try:
        with open(COOLDOWN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"⚠️ อ่าน {COOLDOWN_FILE} ไม่ได้: {e}")
        return {}

def save_cooldowns(cooldowns: dict) -> None:
    try:
        with open(COOLDOWN_FILE, "w", encoding="utf-8") as f:
            json.dump(cooldowns, f, indent=2)
    except Exception as e:
        logger.error(f"❌ บันทึก {COOLDOWN_FILE} ล้มเหลว: {e}")

def is_in_cooldown(coin: str, cooldowns: dict) -> tuple[bool, float]:
    """คืน (in_cooldown, hours_remaining)"""
    ts = cooldowns.get(coin)
    if not ts:
        return False, 0.0
    try:
        closed_time = datetime.fromisoformat(ts)
        hours_since = (datetime.now(timezone.utc) - closed_time).total_seconds() / 3600.0
        remaining = REENTRY_COOLDOWN_HOURS - hours_since
        if remaining > 0:
            return True, remaining
    except Exception:
        pass
    return False, 0.0

def append_trade_log(entry: dict) -> None:
    """
    บันทึก event การปิด/ปิดบางส่วนของ position ลง TRADE_LOG_FILE (JSON Lines)
    เพื่อใช้คำนวณ realized PnL / win-rate ในอนาคต โดยไม่ต้องไปนับจาก positions.json
    (เพราะ positions.json เก็บเฉพาะ position ที่ "เปิดอยู่")
    """
    try:
        with open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"📝 บันทึก trade log: {entry.get('coin')} | {entry.get('event')} | PnL: {entry.get('pnl_pct', 'N/A')}%")
    except Exception as e:
        logger.error(f"❌ บันทึก {TRADE_LOG_FILE} ล้มเหลว: {e}")

def _calc_pnl_pct(entry_price: float, exit_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return round(((exit_price - entry_price) / entry_price) * 100, 3)

def open_position(positions: dict, coin: str, entry_price: float, sl: float, tp1: float, tp2: float, tier: str, atr_at_entry: float) -> None:
    positions[coin] = {
        "entry_price": entry_price,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "sl": sl,
        "original_sl": sl,  # preserved for give-back % calc reference
        "tp1": tp1,
        "tp2": tp2,
        "tier": tier,
        "atr_at_entry": atr_at_entry,
        "tp1_hit": False,
        "tp1_5_hit": False,
        "partial_closed": False,
        "remaining_size_pct": 1.0,
        "high_water_mark": entry_price,
        "status": "open",
    }

def log_partial_close(position: dict, coin: str, exit_price: float, event: str = "partial_tp1", size_closed_pct: float | None = None) -> None:
    """บันทึก Partial TP event ลง trade log (position ยังไม่ถูกลบจาก positions.json)"""
    if size_closed_pct is None:
        size_closed_pct = round(PARTIAL_TP1_CLOSE_PCT * 100, 1)
    append_trade_log({
        "coin": coin,
        "event": event,
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "pnl_pct": _calc_pnl_pct(position["entry_price"], exit_price),
        "size_closed_pct": size_closed_pct,
        "entry_time": position["entry_time"],
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "tier": position.get("tier"),
    })

def close_position(positions: dict, coin: str, reason: str, exit_price: float | None = None, cooldowns: dict | None = None) -> None:
    if coin in positions:
        pos = positions[coin]

        if exit_price is not None:
            # remaining_size_pct reflects what's left to close at this point
            # (1.0 if closed before any partial TP, else the runner's remainder)
            size_closed_pct = round(pos.get("remaining_size_pct", 1.0) * 100, 1)
            append_trade_log({
                "coin": coin,
                "event": reason,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": _calc_pnl_pct(pos["entry_price"], exit_price),
                "size_closed_pct": size_closed_pct,
                "entry_time": pos["entry_time"],
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "tier": pos.get("tier"),
                "tp1_hit_before_close": pos.get("tp1_hit", False),
                "high_water_mark": pos.get("high_water_mark"),
            })

        # [Re-entry Cooldown] Only start a cooldown for full SL closes
        # BEFORE any TP1/partial profit was taken — i.e. a "clean loss".
        # Runner stop-outs after TP1 already banked profit, so re-entry
        # isn't "chasing a falling knife" in the same sense.
        if cooldowns is not None and reason == "close_sl":
            cooldowns[coin] = datetime.now(timezone.utc).isoformat()

        pos["status"] = f"closed ({reason})"
        pos["closed_time"] = datetime.now(timezone.utc).isoformat()
        # Drop closed positions from the active file to keep it lean,
        # but keep a short note in the log for traceability.
        logger.info(f"📕 ปิด position {coin}: {reason}")
        del positions[coin]

def update_position_trailing_stop(position: dict, current_price: float, current_atr: float, adx_rising: bool = False, current_adx: float = 0.0) -> tuple[float, list]:
    """
    ปรับ SL ตาม trailing logic:
    1. ถ้าราคาวิ่งไปแล้วครึ่งทางสู่ TP1 (และยังไม่ถึง TP1) -> เลื่อน SL ไป breakeven (entry price)
    2. ถ้าราคาทะลุ TP1 ไปแล้ว (เข้าสู่ Runner phase หลัง Partial TP) ->
       - ปกติ: trail SL ด้วย ATR * RUNNER_TRAIL_ATR_MULTIPLIER ใต้ราคาปัจจุบัน (แน่นกว่า phase ก่อน TP1)
       - [Runner Extension] ถ้า ADX ยัง >= RUNNER_EXTEND_MIN_ADX และกำลังเพิ่มขึ้น (เทรนด์แข็งแกร่งต่อเนื่อง)
         ใช้ trail กว้างขึ้น (RUNNER_EXTEND_ATR_MULTIPLIER) เพื่อให้เทรนด์ "หายใจ" ได้
    SL เลื่อนขึ้นได้เท่านั้น ไม่เลื่อนลง
    คืนค่า (new_sl, change_notes)
    """
    entry, sl, tp1 = position["entry_price"], position["sl"], position["tp1"]
    notes = []
    new_sl = sl

    halfway_to_tp1 = entry + (tp1 - entry) * BREAKEVEN_TRIGGER_PCT

    if position.get("tp1_hit", False):
        if not pd.isna(current_atr) and current_atr > 0:
            if adx_rising and current_adx >= RUNNER_EXTEND_MIN_ADX:
                # [Runner Extension] strong continuing trend -> wider trail
                trail_mult = RUNNER_EXTEND_ATR_MULTIPLIER
                ext_note = f" 🔥 Runner Extension (ADX {current_adx:.1f} เพิ่มขึ้น)"
            else:
                trail_mult = RUNNER_TRAIL_ATR_MULTIPLIER
                ext_note = ""

            trail_sl = current_price - (current_atr * trail_mult)
            if trail_sl > new_sl:
                new_sl = trail_sl
                notes.append(f"📈 Runner: เลื่อน Trailing SL ขึ้นเป็น {format_price(new_sl)} (ATR x{trail_mult}){ext_note}")
    elif current_price >= halfway_to_tp1 and new_sl < entry:
        # Move to breakeven once price has covered 50% of the distance to TP1
        new_sl = entry
        notes.append(f"🛡️ ราคาวิ่งเกินครึ่งทางสู่ TP1 — เลื่อน SL ไป Breakeven ({format_price(entry)})")

    return new_sl, notes

def check_position_status(position: dict, current_price: float, current_atr: float, adx_rising: bool = False, current_adx: float = 0.0) -> dict:
    """
    ตรวจสอบ position ที่เปิดอยู่เทียบกับราคาปัจจุบัน:
    - แตะ SL แล้ว -> ต้องปิด (loss / breakeven / runner stop-out)
    - แตะ TP1 ครั้งแรก (ยังไม่ partial close) -> สั่ง Partial TP, เริ่ม Runner phase
    - แตะ TP1.5 (ระหว่าง TP1-TP2, ยังไม่เคยแตะ) -> ปิดบางส่วนเพิ่ม (TP1_5_CLOSE_PCT)
    - แตะ TP2 (หรือ SL หลัง partial) -> ปิด runner ที่เหลือทั้งหมด
    - อัปเดต High-Water-Mark + Give-back exit (pre-TP1)
    - อัปเดต trailing SL ถ้ายังไม่ถึงเงื่อนไขปิด
    - ตรวจ time-based stop ถ้าเปิดมานานเกินกำหนดและยังไม่ไป TP1/SL
    คืน dict: {"action": ..., "new_sl": float, "notes": [...]}

    Possible actions:
      "close_sl"        -> full close, SL hit before TP1 (full loss/breakeven)
      "partial_tp1"     -> TP1 hit for the first time: close PARTIAL_TP1_CLOSE_PCT,
                            remainder becomes the "runner"
      "partial_tp1_5"   -> TP1.5 hit: close another TP1_5_CLOSE_PCT of original size
      "close_tp2"       -> runner hit TP2: close remainder fully
      "close_runner_sl" -> runner trailing stop hit after TP1: close remainder
      "give_back_warn"  -> pre-TP1 give-back from high-water-mark exceeds threshold
      "update"          -> trailing SL adjusted, position stays open
      "time_stop"       -> still open too long without reaching TP1
    """
    result = {"action": "update", "new_sl": position["sl"], "notes": []}

    sl, tp1, tp2 = position["sl"], position["tp1"], position["tp2"]
    entry = position["entry_price"]
    tp1_hit = position.get("tp1_hit", False)
    tp1_5_hit = position.get("tp1_5_hit", False)

    # --- High-Water-Mark update (always, regardless of phase) ---
    hwm = position.get("high_water_mark", entry)
    if current_price > hwm:
        position["high_water_mark"] = current_price
        hwm = current_price

    # --- SL hit BEFORE TP1: full close (original stop, full size) ---
    if not tp1_hit and current_price <= sl:
        result["action"] = "close_sl"
        result["notes"].append(f"❌ ราคาแตะ SL ({format_price(sl)}) — ปิดสถานะเต็มจำนวน")
        return result

    # --- TP1 hit for the first time: Partial TP, start Runner ---
    if not tp1_hit and current_price >= tp1:
        position["tp1_hit"] = True
        position["partial_closed"] = True
        position["remaining_size_pct"] = round(1.0 - PARTIAL_TP1_CLOSE_PCT, 4)
        result["action"] = "partial_tp1"
        result["notes"].append(
            f"🎯 TP1 ถูกแตะแล้ว ({format_price(tp1)}) — "
            f"ปิด {PARTIAL_TP1_CLOSE_PCT*100:.0f}% ของไม้ (Partial TP)"
        )
        result["notes"].append(
            f"🏃 เหลือ {position['remaining_size_pct']*100:.0f}% เป็น Runner — "
            f"เริ่ม Trailing Stop แน่นขึ้น (ATR x{RUNNER_TRAIL_ATR_MULTIPLIER})"
        )
        # Immediately compute the first runner trailing SL (don't wait a cycle)
        new_sl, trail_notes = update_position_trailing_stop(position, current_price, current_atr, adx_rising, current_adx)
        result["new_sl"] = new_sl
        result["notes"].extend(trail_notes)
        return result

    # --- After TP1: managing the Runner ---
    if tp1_hit:
        # Runner hits TP2 -> close remainder fully
        if current_price >= tp2:
            result["action"] = "close_tp2"
            result["notes"].append(f"🏁 Runner แตะ TP2 ({format_price(tp2)}) — ปิดสถานะส่วนที่เหลือทั้งหมด")
            return result

        # [Multi-Stage TP] TP1.5 hit for the first time (between TP1 and TP2)
        tp1_5_price = entry + (tp2 - entry) * (ATR_TP1_5_MULTIPLIER - ATR_TP1_MULTIPLIER) / max(ATR_TP2_MULTIPLIER - ATR_TP1_MULTIPLIER, 0.0001)
        # NOTE: tp1_5_price is interpolated between tp1 and tp2 using the ATR
        # multiplier ratios (since tp1/tp2 themselves were computed from ATR
        # multiples — see calculate_atr_based_tp_sl). This keeps TP1.5
        # self-consistent without re-fetching ATR here.
        tp1_5_price = tp1 + (tp2 - tp1) * (ATR_TP1_5_MULTIPLIER - ATR_TP1_MULTIPLIER) / max(ATR_TP2_MULTIPLIER - ATR_TP1_MULTIPLIER, 0.0001)

        if not tp1_5_hit and current_price >= tp1_5_price:
            position["tp1_5_hit"] = True
            position["remaining_size_pct"] = round(max(position["remaining_size_pct"] - TP1_5_CLOSE_PCT, 0.0), 4)
            result["action"] = "partial_tp1_5"
            result["notes"].append(
                f"🎯 TP1.5 ถูกแตะแล้ว ({format_price(tp1_5_price)}) — "
                f"ปิดอีก {TP1_5_CLOSE_PCT*100:.0f}% ของขนาดเดิม "
                f"(เหลือ {position['remaining_size_pct']*100:.0f}%)"
            )
            new_sl, trail_notes = update_position_trailing_stop(position, current_price, current_atr, adx_rising, current_adx)
            result["new_sl"] = new_sl
            result["notes"].extend(trail_notes)
            return result

        # Runner trailing stop hit -> close remainder
        if current_price <= sl:
            result["action"] = "close_runner_sl"
            result["notes"].append(f"🔚 Runner โดน Trailing SL ({format_price(sl)}) — ปิดสถานะส่วนที่เหลือ")
            return result

    # --- [Give-back Exit] Pre-TP1 only: warn if price retraced too much from HWM ---
    if not tp1_hit:
        peak_gain_pct = ((hwm - entry) / entry) * 100 if entry > 0 else 0.0
        if peak_gain_pct >= GIVE_BACK_MIN_PEAK_GAIN_PCT:
            give_back_pct = ((hwm - current_price) / (hwm - entry)) if (hwm - entry) > 0 else 0.0
            if give_back_pct >= GIVE_BACK_EXIT_PCT and current_price > sl:
                result["action"] = "give_back_warn"
                result["notes"].append(
                    f"⚠️ ราคาขึ้นไปสูงสุด {format_price(hwm)} (+{peak_gain_pct:.1f}%) "
                    f"แล้วย่อกลับมาแล้ว {give_back_pct*100:.0f}% ของกำไรสูงสุด ก่อนถึง TP1 "
                    f"— พิจารณาปิดสถานะเพื่อรักษากำไรที่เหลือ"
                )
                # Not a hard close — surfaced as a warning (info action below
                # still applies trailing SL updates); caller decides whether
                # to treat this as actionable.

    # --- Otherwise: update trailing SL (no closure this cycle) ---
    new_sl, trail_notes = update_position_trailing_stop(position, current_price, current_atr, adx_rising, current_adx)
    result["new_sl"] = new_sl
    result["notes"].extend(trail_notes)

    # Time-based stop on the actual open position (only relevant pre-TP1)
    try:
        entry_time = datetime.fromisoformat(position["entry_time"])
        hours_open = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0
        if hours_open >= POSITION_TIME_STOP_HOURS and not tp1_hit:
            result["action"] = "time_stop"
            result["notes"].append(
                f"⏳ เปิดสถานะมาแล้ว {hours_open/24:.1f} วัน ยังไม่ถึง TP1 — พิจารณาปิด (Time-Stop)"
            )
    except Exception:
        pass

    # [Fix I] SL Proximity Warning — เตือนล่วงหน้าก่อนโดน SL
    # ถ้าราคาเข้าใกล้ SL ภายใน SL_PROXIMITY_WARNING_PCT% → แจ้งเตือน
    # ทำให้มีเวลาตัดสินใจ manual ก่อนโดน SL จริง
    if current_price > sl and entry > 0:
        sl_distance_pct = ((current_price - sl) / entry) * 100
        sl_total_pct    = ((entry - sl) / entry) * 100
        sl_proximity_threshold = SL_PROXIMITY_WARNING_PCT  # % ของ entry
        if sl_distance_pct <= sl_proximity_threshold and sl_distance_pct > 0:
            pct_to_sl = round(((current_price - sl) / current_price) * 100, 2)
            result["notes"].append(
                f"⚠️ SL Proximity: ราคาห่างจาก SL ({format_price(sl)}) เพียง {pct_to_sl:.2f}% "
                f"— เฝ้าระวัง อาจพิจารณา Manual Stop"
            )

    return result


def _parse_binance_klines(data: list, coin: str, tf: str) -> pd.DataFrame | None:
    """
    Binance klines format (12 columns):
    [0]  open_time
    [1]  open
    [2]  high
    [3]  low
    [4]  close
    [5]  volume          (base asset volume — e.g. EIGEN coins)
    [6]  close_time
    [7]  quote_volume    (quote asset volume — USDT) ← ใช้อันนี้สำหรับ volumeto
    [8]  trades
    [9]  taker_base_vol
    [10] taker_quote_vol
    [11] ignore
    """
    try:
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore",
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # rename quote_volume → volumeto (USDT value)
        df = df.rename(columns={"quote_volume": "volumeto"})
        df = df[["open", "high", "low", "close", "volumeto"]].dropna()

        if len(df) < 10:
            return None

        # Data quality check: validate close prices are not all identical (frozen)
        if _is_df_frozen(df, coin):
            logger.error(f"❌ {coin} Binance {tf}: frozen data detected — skip")
            return None

        hours_old = _check_df_staleness(df, coin, f"Binance {tf}")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            logger.error(f"❌ {coin} Binance {tf} data เก่าเกิน {hours_old:.1f}h — ข้าม")
            return None

        logger.info(f"✅ {coin} {tf} Binance สำเร็จ ({len(df)} แท่ง, อายุ {hours_old:.1f}h)")
        return df
    except Exception as e:
        logger.warning(f"{coin} {tf} Binance parse error: {e}")
        return None

def _fetch_from_binance(symbol: str, interval: str, limit: int, coin: str, tf: str) -> pd.DataFrame | None:
    """
    [Perf Fix] timeout ลดจาก 15s → 8s ต่อ endpoint, endpoints ลดจาก 6 → 3 ตัว
    worst case เดิม: 6 endpoints × 15s = 90s/coin
    worst case ใหม่: 3 endpoints × 8s = 24s/coin (ลดลง ~73%)
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}/api/v3/klines"
        try:
            resp = api_session.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                return _parse_binance_klines(resp.json(), coin, tf)
            elif resp.status_code == 451:
                logger.warning(f"{coin} {tf} {base_url} → 451 geo-block ข้าม")
            else:
                logger.warning(f"{coin} {tf} {base_url} → HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"{coin} {tf} {base_url} → {e}")
    return None

def _fetch_4h_from_cryptocompare(coin: str) -> pd.DataFrame | None:
    url    = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {"fsym": coin, "tsym": "USD", "limit": CC_HISTOHOUR_LIMIT}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    try:
        resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return None
        res_json = resp.json()
        if res_json.get("Response") == "Error":
            return None
        raw = res_json.get("Data", {}).get("Data")
        if not raw:
            return None
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        if "volumeto" not in df.columns:
            df["volumeto"] = df["volumefrom"] if "volumefrom" in df.columns else 0.0
        required = ["open", "high", "low", "close", "volumeto"]
        for col in required:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

        df_4h = df[required].resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volumeto": "sum",
        }).dropna()

        if len(df_4h) < 10:
            return None
        hours_old = _check_df_staleness(df_4h, coin, "CryptoCompare 4H")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            logger.error(f"❌ {coin} CC data เก่าเกิน {hours_old:.1f}h — ข้าม")
            return None
        logger.info(f"✅ {coin} 4H CC สำเร็จ ({len(df_4h)} แท่ง 4H, อายุ {hours_old:.1f}h)")
        return df_4h
    except Exception as e:
        logger.error(f"❌ {coin} 4H CC error: {e}")
        return None

def _fetch_1d_from_cryptocompare(coin: str) -> pd.DataFrame | None:
    url    = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": coin, "tsym": "USD", "limit": CC_HISTODAY_LIMIT}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    try:
        resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return None
        raw = resp.json().get("Data", {}).get("Data")
        if not raw:
            return None
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        if "volumeto" not in df.columns:
            df["volumeto"] = df["volumefrom"] if "volumefrom" in df.columns else 0.0
        required = ["open", "high", "low", "close", "volumeto"]
        for col in required:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
        df_out = df[required].dropna(subset=["open", "high", "low", "close"])
        logger.info(f"✅ {coin} 1D CC สำเร็จ ({len(df_out)} วัน)")
        return df_out
    except Exception:
        return None

def _fetch_live_price_binance(coin: str) -> float | None:
    """
    [Approach 2 - Live EMA200 Check] ดึงราคาปัจจุบัน (real-time) จาก Binance ticker
    ใช้สำหรับเช็คว่าราคาจริง ณ ขณะนี้ได้ทะลุกลับเหนือ EMA200 แล้วหรือยัง
    แม้แท่ง 4H ล่าสุดที่ใช้คำนวณ indicator จะยังไม่ปิดก็ตาม

    เร็วกว่า CoinGecko และไม่มี rate limit เข้มงวด — ใช้ /ticker/price endpoint
    (เบากว่า klines มาก เพราะคืนแค่ราคาล่าสุด ไม่ใช่ array ของแท่ง)

    [Perf Fix] timeout สั้นลง (3s) + ไม่ retry CoinGecko บน 429 (เพื่อไม่ให้
    1 coin ค้างได้สูงสุดถึง 32s) — ถ้าทุก source ล้มเหลว ให้ fail fast แทน
    """
    symbol = f"{coin}USDT"
    for base_url in ["https://api.binance.com", "https://data-api.binance.vision"]:
        try:
            resp = api_session.get(f"{base_url}/api/v3/ticker/price",
                                    params={"symbol": symbol}, timeout=3)
            if resp.status_code == 200:
                price = resp.json().get("price")
                if price:
                    return float(price)
        except Exception:
            continue
    # Fallback: CoinGecko (ช้ากว่า, timeout สั้น, ไม่ retry บน 429)
    return _fetch_realtime_price_coingecko(coin, fast_mode=True)


def _fetch_realtime_price_coingecko(coin: str, fast_mode: bool = False) -> float | None:
    """
    ดึง current price จาก CoinGecko /simple/price — real-time ไม่มี bucket delay
    ใช้ patch candle ล่าสุดใน df ถ้า OHLC data เก่าเกิน threshold

    fast_mode=True: ใช้ timeout สั้น (4s) และไม่ retry บน 429
    (ใช้เมื่อเรียกจาก _fetch_live_price_binance ที่ต้องการความเร็วสำหรับ
    หลาย coins ในรอบเดียว — ไม่อยากให้ 1 coin ค้าง 10s+ จาก rate limit)
    """
    cg_id = COINGECKO_IDS.get(coin)
    if not cg_id:
        return None
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": cg_id, "vs_currencies": "usd"}
    timeout_s = 4 if fast_mode else 10
    try:
        resp = api_session.get(url, params=params, timeout=timeout_s)
        if resp.status_code == 429:
            if fast_mode:
                return None  # fail fast แทนการรอ 10s
            time.sleep(10)
            resp = api_session.get(url, params=params, timeout=timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
        price = data.get(cg_id, {}).get("usd")
        return float(price) if price else None
    except Exception:
        return None

def _patch_df_with_realtime_price(df: pd.DataFrame, coin: str, realtime_price: float) -> pd.DataFrame:
    """
    แทนที่ close price ของ candle ล่าสุดด้วยราคา real-time
    เพื่อแก้ปัญหา CoinGecko OHLC bucket delay
    ปรับ high/low ให้สอดคล้องด้วยถ้าราคา real-time อยู่นอก range
    """
    if df is None or len(df) == 0:
        return df
    last_idx = df.index[-1]
    old_close = df.loc[last_idx, "close"]
    df.loc[last_idx, "close"] = realtime_price
    # ปรับ high/low ถ้าราคา real-time อยู่นอก range ของ candle นั้น
    if realtime_price > df.loc[last_idx, "high"]:
        df.loc[last_idx, "high"] = realtime_price
    if realtime_price < df.loc[last_idx, "low"]:
        df.loc[last_idx, "low"] = realtime_price
    logger.info(f"🔄 {coin} CoinGecko price patched: {old_close:.6f} → {realtime_price:.6f} (real-time)")
    return df

def _check_df_staleness(df: pd.DataFrame, coin: str, source: str) -> float:
    """
    ตรวจสอบว่า candle ล่าสุดเก่าแค่ไหน คืน hours_old (ชั่วโมง)
    """
    if df is None or len(df) == 0:
        return 999.0
    try:
        last_ts = df.index[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        now_utc = datetime.now(timezone.utc)
        hours_old = (now_utc - last_ts).total_seconds() / 3600.0
        if hours_old > DATA_STALENESS_WARN_HOURS_4H:
            logger.warning(
                f"⚠️ {coin} {source}: candle ล่าสุดเก่า {hours_old:.1f}h "
                f"(threshold warn={DATA_STALENESS_WARN_HOURS}h skip={DATA_STALENESS_SKIP_HOURS_4H}h)"
            )
        return hours_old
    except Exception:
        return 999.0

def _fetch_4h_from_coingecko(coin: str) -> pd.DataFrame | None:
    """
    ดึง 4H OHLC จาก CoinGecko โดยใช้ /market_chart แทน /ohlc
    
    สาเหตุที่เปลี่ยน:
    - /ohlc ใช้ server-side bucket system → candle อาจเก่า 4-8h เสมอ
      ไม่ว่าจะใส่ no-cache header หรือดึงซ้ำกี่ครั้ง
    - /market_chart ให้ prices รายชั่วโมง (free tier) แบบ real-time กว่ามาก
      แล้ว resample เป็น 4H เอง → candle สดกว่า /ohlc เสมอ
    """
    cg_id = COINGECKO_IDS.get(coin)
    if not cg_id:
        logger.warning(f"❌ {coin} ไม่พบ CoinGecko ID — ข้าม")
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(CG_OHLC_DAYS_4H), "interval": "hourly"}
    try:
        resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            logger.warning(f"⚠️ {coin} market_chart CoinGecko rate limit — รอ 15s")
            time.sleep(5)
            resp = api_session.get(url, params=params, timeout=20)

        if resp.status_code != 200:
            logger.warning(f"⚠️ {coin} market_chart HTTP {resp.status_code} — ลอง /ohlc")
            return _fetch_4h_from_coingecko_ohlc(coin)  # fallback to old method

        data = resp.json()
        prices = data.get("prices", [])
        if not prices or len(prices) < 48:
            return _fetch_4h_from_coingecko_ohlc(coin)

        # Build OHLCV from hourly price data
        df_h = pd.DataFrame(prices, columns=["time", "close"])
        df_h["time"] = pd.to_datetime(df_h["time"], unit="ms", utc=True)
        df_h.set_index("time", inplace=True)
        df_h["close"] = pd.to_numeric(df_h["close"], errors="coerce")
        df_h["open"]  = df_h["close"].shift(1)
        df_h["high"]  = df_h["close"]
        df_h["low"]   = df_h["close"]

        # Enrich with volume if available
        volumes = data.get("total_volumes", [])
        if volumes:
            df_v = pd.DataFrame(volumes, columns=["time", "volumeto"])
            df_v["time"] = pd.to_datetime(df_v["time"], unit="ms", utc=True)
            df_v.set_index("time", inplace=True)
            df_v["volumeto"] = pd.to_numeric(df_v["volumeto"], errors="coerce")
            df_h = df_h.join(df_v, how="left")
        else:
            df_h["volumeto"] = 0.0

        # Resample to 4H OHLCV
        df_4h = df_h[["open", "high", "low", "close", "volumeto"]].resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volumeto": "sum",
        }).dropna(subset=["open", "high", "low", "close"])

        if len(df_4h) < 10:
            return None

        # Staleness check
        hours_old = _check_df_staleness(df_4h, coin, "CoinGecko market_chart")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            logger.error(f"❌ {coin} market_chart เก่าเกิน {hours_old:.1f}h")
            return None
        if hours_old > DATA_STALENESS_WARN_HOURS_4H:
            rt_price = _fetch_realtime_price_coingecko(coin)
            if rt_price:
                df_4h = _patch_df_with_realtime_price(df_4h, coin, rt_price)

        # Frozen data check: ถ้า unique close prices < 30% → data frozen → return None
        # ให้ caller (get_historical_data) ลอง source อื่นแทน
        if len(df_4h) >= 20:
            recent_c = df_4h["close"].iloc[-20:]; n_unique = recent_c.nunique(); cv = recent_c.std() / recent_c.mean() if recent_c.mean() != 0 else 0
            if cv < 0.0002 and n_unique < max(len(df_4h)//5, 4):
                logger.warning(f"⚠️ {coin} CoinGecko market_chart: frozen data ({n_unique} unique prices) — ข้าม")
                return None

        logger.info(f"✅ {coin} 4H CoinGecko market_chart สำเร็จ ({len(df_4h)} แท่ง, อายุ {hours_old:.1f}h)")
        return df_4h
    except Exception as e:
        logger.error(f"❌ {coin} 4H CoinGecko market_chart error: {e}")
        return _fetch_4h_from_coingecko_ohlc(coin)


def _fetch_4h_from_coingecko_ohlc(coin: str) -> pd.DataFrame | None:
    """Fallback: original /ohlc endpoint (bucket-based, may be stale)"""
    cg_id = COINGECKO_IDS.get(coin)
    if not cg_id:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(CG_OHLC_DAYS_4H)}
    try:
        resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            time.sleep(5)
            resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        df["volumeto"] = 0.0
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close", "volumeto"]].dropna(
            subset=["open", "high", "low", "close"])
        if len(df) < 10:
            return None
        hours_old = _check_df_staleness(df, coin, "CoinGecko OHLC fallback")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            logger.error(f"❌ {coin} OHLC fallback เก่าเกิน {hours_old:.1f}h")
            return None
        if hours_old > DATA_STALENESS_WARN_HOURS_4H:
            rt = _fetch_realtime_price_coingecko(coin)
            if rt:
                df = _patch_df_with_realtime_price(df, coin, rt)
        # Frozen check
        if len(df) >= 20:
            recent_c = df["close"].iloc[-20:]; n_unique = recent_c.nunique(); cv = recent_c.std() / recent_c.mean() if recent_c.mean() != 0 else 0
            if cv < 0.0002 and n_unique < max(len(df)//5, 4):
                logger.warning(f"⚠️ {coin} OHLC fallback: frozen ({n_unique} unique prices) — return None")
                return None
        logger.info(f"✅ {coin} 4H CoinGecko OHLC fallback ({len(df)} แท่ง, อายุ {hours_old:.1f}h)")
        return df
    except Exception:
        return None


def _fetch_4h_from_gateio(coin: str) -> pd.DataFrame | None:
    """
    [New Source] Gate.io Spot — เพิ่มเป็น fallback สำหรับเหรียญที่ไม่มีบน Binance/CC
    เช่น EIGEN, เหรียญใหม่ที่ยังไม่ list บน major exchanges
    API: https://api.gateio.ws/api/v4/spot/candlesticks
    """
    symbol = f"{coin}_USDT"
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {
        "currency_pair": symbol,
        "interval": "4h",
        "limit": 500,
    }
    try:
        resp = api_session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"⚠️ {coin} Gate.io HTTP {resp.status_code}")
            return None
        data = resp.json()
        if not data or not isinstance(data, list) or len(data) < 10:
            return None
        # Gate.io format: [timestamp, volume, close, high, low, open, ...]
        rows = []
        for candle in data:
            try:
                ts  = pd.to_datetime(int(candle[0]), unit="s", utc=True)
                vol = float(candle[1])
                close_ = float(candle[2])
                high_  = float(candle[3])
                low_   = float(candle[4])
                open_  = float(candle[5])
                rows.append({"time": ts, "open": open_, "high": high_,
                             "low": low_, "close": close_, "volumeto": vol})
            except (IndexError, ValueError):
                continue
        if len(rows) < 10:
            return None
        df = pd.DataFrame(rows).set_index("time").sort_index()
        hours_old = _check_df_staleness(df, coin, "Gate.io")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            return None
        logger.info(f"✅ {coin} 4H Gate.io สำเร็จ ({len(df)} แท่ง, อายุ {hours_old:.1f}h)")
        return df
    except Exception as e:
        logger.warning(f"⚠️ {coin} Gate.io error: {e}")
        return None


def _fetch_4h_from_kucoin(coin: str) -> pd.DataFrame | None:
    """
    [New Source] KuCoin Spot — fallback ที่ 2 สำหรับเหรียญไม่มีบน Binance
    API: https://api.kucoin.com/api/v1/market/candles
    """
    symbol = f"{coin}-USDT"
    url = "https://api.kucoin.com/api/v1/market/candles"
    import time as time_mod
    end_ts = int(time_mod.time())
    start_ts = end_ts - (500 * 4 * 3600)  # ~500 4H candles back
    params = {
        "symbol": symbol,
        "type": "4hour",
        "startAt": start_ts,
        "endAt": end_ts,
    }
    try:
        resp = api_session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        body = resp.json()
        if body.get("code") != "200000":
            return None
        data = body.get("data", [])
        if not data or len(data) < 10:
            return None
        # KuCoin format: [timestamp, open, close, high, low, volume, turnover]
        rows = []
        for candle in data:
            try:
                ts     = pd.to_datetime(int(candle[0]), unit="s", utc=True)
                open_  = float(candle[1])
                close_ = float(candle[2])
                high_  = float(candle[3])
                low_   = float(candle[4])
                vol    = float(candle[5])
                rows.append({"time": ts, "open": open_, "high": high_,
                             "low": low_, "close": close_, "volumeto": vol})
            except (IndexError, ValueError):
                continue
        if len(rows) < 10:
            return None
        df = pd.DataFrame(rows).set_index("time").sort_index()
        hours_old = _check_df_staleness(df, coin, "KuCoin")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            return None
        logger.info(f"✅ {coin} 4H KuCoin สำเร็จ ({len(df)} แท่ง, อายุ {hours_old:.1f}h)")
        return df
    except Exception as e:
        logger.warning(f"⚠️ {coin} KuCoin error: {e}")
        return None


    cg_id = COINGECKO_IDS.get(coin)
    if not cg_id:
        logger.warning(f"❌ {coin} ไม่พบ CoinGecko ID — ข้าม")
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(CG_OHLC_DAYS_4H)}
    try:
        resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            logger.warning(f"⚠️ {coin} 4H CoinGecko rate limit — รอ 15s")
            time.sleep(5)
            resp = api_session.get(url, params=params, timeout=20)

        if resp.status_code != 200:
            logger.error(f"❌ {coin} 4H CoinGecko HTTP {resp.status_code}")
            return None

        data = resp.json()
        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df["volumeto"] = 0.0

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["open", "high", "low", "close", "volumeto"]].dropna(
            subset=["open", "high", "low", "close"]
        )

        if len(df) < 10:
            return None

        # Staleness check + real-time price patch
        hours_old = _check_df_staleness(df, coin, "CoinGecko OHLC")
        if hours_old > DATA_STALENESS_SKIP_HOURS_4H:
            logger.error(
                f"❌ {coin} CoinGecko OHLC เก่าเกินไป ({hours_old:.1f}h > {DATA_STALENESS_SKIP_HOURS_4H}h) "
                f"— ข้ามเหรียญนี้เพื่อป้องกันสัญญาณผิดพลาด"
            )
            return None  # ข้าม coin ที่ข้อมูลเก่าเกินไป

        if hours_old > DATA_STALENESS_WARN_HOURS_4H:
            # Patch last candle with real-time price
            rt_price = _fetch_realtime_price_coingecko(coin)
            if rt_price is not None:
                df = _patch_df_with_realtime_price(df, coin, rt_price)
            else:
                logger.warning(f"⚠️ {coin}: real-time price fetch ล้มเหลว — ใช้ข้อมูลเก่า {hours_old:.1f}h")

        logger.info(f"✅ {coin} 4H CoinGecko สำเร็จ ({len(df)} แท่ง, candle อายุ {hours_old:.1f}h)")
        return df
    except Exception as e:
        logger.error(f"❌ {coin} 4H CoinGecko error: {e}")
        return None

def _fetch_1d_from_coingecko(coin: str) -> pd.DataFrame | None:
    cg_id = COINGECKO_IDS.get(coin)
    if not cg_id:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(CG_OHLC_DAYS_1D)}
    try:
        resp = api_session.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            logger.warning(f"⚠️ {coin} 1D CoinGecko rate limit — รอ 15s")
            time.sleep(5)
            resp = api_session.get(url, params=params, timeout=20)

        if resp.status_code != 200:
            return None

        data = resp.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df["volumeto"] = 0.0
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df_1d = df[["open", "high", "low", "close", "volumeto"]].resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volumeto": "sum",
        }).dropna(subset=["open", "high", "low", "close"])
        return df_1d
    except Exception:
        return None

def _fetch_4h_from_binance_futures(coin: str) -> pd.DataFrame | None:
    """
    ดึง 4H OHLC จาก Binance Futures (USDT-M perpetual)
    สำหรับเหรียญที่ไม่มีบน Binance Spot แต่มี perpetual futures
    เช่น EIGEN, เหรียญใหม่ที่ list futures ก่อน spot
    """
    symbol = f"{coin}USDT"
    params = {"symbol": symbol, "interval": "4h", "limit": BINANCE_LIMIT}
    for base_url in BINANCE_FUTURES_ENDPOINTS:
        url = f"{base_url}/fapi/v1/klines"
        try:
            resp = api_session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                df = _parse_binance_klines(resp.json(), coin, "4H-Futures")
                if df is not None:
                    hours_old = _check_df_staleness(df, coin, "Binance Futures")
                    if hours_old <= DATA_STALENESS_SKIP_HOURS_4H:
                        logger.info(f"✅ {coin} 4H Binance Futures สำเร็จ ({len(df)} แท่ง)")
                        return df
            elif resp.status_code in (400, 451):
                continue
        except Exception as e:
            logger.warning(f"⚠️ {coin} Binance Futures: {e}")
    return None


def _is_df_frozen(df: pd.DataFrame, coin: str = "") -> bool:
    """
    ตรวจ degenerate/frozen data:
    1. CV < 0.0002 = ราคาแทบไม่ขยับ
    2. unique <= 2 values ใน 20 แท่ง = degenerate (แม้ cv จะสูง)
       เพราะ 0.2125 ×19 + 0.2441 ×1 = cv~0.07 แต่ข้อมูลยังไม่มีความหมาย
    """
    if df is None or len(df) < 10:
        return False
    recent = df["close"].iloc[-20:]
    mean = recent.mean()
    if mean == 0:
        return True
    cv = recent.std() / mean
    n_unique = recent.nunique()
    # Degenerate: <= 2 unique values ใน 20 แท่ง = data ไม่มีความหมายพอสำหรับ RSI
    if n_unique <= 2:
        if coin:
            logger.warning(f"🧊 {coin}: degenerate data (unique={n_unique}/20, cv={cv:.4f}) — frozen")
        return True
    # Classic frozen: CV ต่ำมาก
    frozen = cv < 0.0002 and n_unique < max(len(recent) // 5, 4)
    if frozen and coin:
        logger.warning(f"🧊 {coin}: frozen data (cv={cv:.6f}, unique={n_unique}/20)")
    return frozen


def get_historical_data(coin: str) -> pd.DataFrame | None:
    """
    Fetch chain (ลำดับ priority):
    1. Binance Spot      — ข้อมูลดีที่สุด, real-time
    2. Binance Futures   — สำหรับเหรียญที่มี futures แต่ไม่มี spot (เช่น EIGEN)
    3. CryptoCompare     — fallback hourly
    4. Gate.io Spot      — fallback สำหรับเหรียญเล็ก
    5. KuCoin Spot       — fallback เพิ่มเติม
    6. CoinGecko market_chart — hourly resample (ดีกว่า /ohlc)
    ทุก source ผ่าน frozen check — ถ้า frozen → ลอง source ถัดไป
    """
    symbol   = f"{coin}USDT"
    min_bars = EMA_LONG + 10

    df = _fetch_from_binance(symbol, "4h", BINANCE_LIMIT, coin, "4H")
    if df is not None and len(df) >= min_bars and not _is_df_frozen(df, coin):
        return df

    df = _fetch_4h_from_binance_futures(coin)
    if df is not None and len(df) >= min_bars and not _is_df_frozen(df, coin):
        return df

    df = _fetch_4h_from_cryptocompare(coin)
    if df is not None and len(df) >= min_bars and not _is_df_frozen(df, coin):
        return df

    df = _fetch_4h_from_gateio(coin)
    if df is not None and len(df) >= min_bars and not _is_df_frozen(df, coin):
        return df

    df = _fetch_4h_from_kucoin(coin)
    if df is not None and len(df) >= min_bars and not _is_df_frozen(df, coin):
        return df

    df = _fetch_4h_from_coingecko(coin)
    if df is not None and _is_df_frozen(df, coin):
        logger.error(f"❌ {coin}: ทุก source ให้ frozen data — return None")
        return None
    return df


def get_histoday_data(coin: str) -> pd.DataFrame | None:
    symbol   = f"{coin}USDT"
    min_bars = 30

    df = _fetch_from_binance(symbol, "1d", BINANCE_LIMIT, coin, "1D")
    if df is not None and len(df) >= min_bars: return df

    df = _fetch_1d_from_cryptocompare(coin)
    if df is not None and len(df) >= min_bars: return df

    return _fetch_1d_from_coingecko(coin)

# ==========================================
# On-Chain Fetching (1M Trend)
# ==========================================
def get_onchain_data(coin: str) -> dict:
    """ดึงข้อมูล Active Addresses รายวัน 60 วันย้อนหลัง แล้วประเมินเทรนด์ 1 เดือน"""
    result = {
        "has_data": False,
        "active_addresses_trend": 0.0,
        "onchain_label": "⚪ N/A (ไม่มีข้อมูลบนเชน)"
    }

    if coin not in ONCHAIN_SUPPORTED_COINS:
        return result

    url = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
    params = {"fsym": coin, "limit": 60}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY

    try:
        resp = api_session.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return result

        data = resp.json().get("Data", {}).get("Data", [])
        if not data or len(data) < 60:
            return result

        df = pd.DataFrame(data)

        current_month = df.iloc[-30:]["active_addresses"].mean()
        previous_month = df.iloc[-60:-30]["active_addresses"].mean()

        if previous_month > 0:
            growth_pct = ((current_month - previous_month) / previous_month) * 100
            result["active_addresses_trend"] = growth_pct
            result["has_data"] = True

            if growth_pct >= 15:
                result["onchain_label"] = f"🔥 เติบโตแข็งแกร่ง (+{growth_pct:.1f}%)"
            elif growth_pct > 0:
                result["onchain_label"] = f"🟢 เติบโต (+{growth_pct:.1f}%)"
            elif growth_pct > -10:
                result["onchain_label"] = f"🟡 ชะลอตัว ({growth_pct:.1f}%)"
            else:
                result["onchain_label"] = f"🔴 หดตัวหนัก ({growth_pct:.1f}%)"

        logger.info(f"✅ {coin} 1M On-Chain สำเร็จ ({result['onchain_label']})")
    except Exception as e:
        logger.warning(f"❌ {coin} On-Chain Error: {e}")

    return result

# ==========================================
# [#3] Funding Rate & Open Interest (Binance Futures)
# ==========================================
def fetch_all_funding_rates() -> dict:
    """
    ดึง Funding Rate + Mark Price ของทุก symbol จาก Binance Futures ในครั้งเดียว
    (/fapi/v1/premiumIndex ไม่มี param symbol = คืนค่าทุกเหรียญ)
    คืนค่า dict: {coin: {"funding_rate": float, "mark_price": float}}
    """
    result = {}
    for base_url in BINANCE_FUTURES_ENDPOINTS:
        url = f"{base_url}/fapi/v1/premiumIndex"
        try:
            resp = api_session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Funding rate bulk fetch → HTTP {resp.status_code}")
                continue
            data = resp.json()
            if not isinstance(data, list):
                continue
            for item in data:
                symbol = item.get("symbol", "")
                if not symbol.endswith("USDT"):
                    continue
                coin = symbol[:-4]  # strip "USDT"
                try:
                    result[coin] = {
                        "funding_rate": float(item.get("lastFundingRate", 0.0)),
                        "mark_price": float(item.get("markPrice", 0.0)),
                    }
                except (TypeError, ValueError):
                    continue
            if result:
                logger.info(f"✅ Funding Rate bulk fetch สำเร็จ ({len(result)} symbols)")
                return result
        except Exception as e:
            logger.warning(f"Funding rate bulk fetch → {e}")
    return result

def bulk_fetch_funding(coins: list) -> dict:
    """Bulk-cached funding rate / mark price lookup, keyed by coin symbol."""
    global _cache_funding, _cache_ts_funding
    now = time.time()
    if _cache_funding and (now - _cache_ts_funding) < CACHE_TTL_SECONDS:
        return _cache_funding
    result = fetch_all_funding_rates()
    _cache_funding = result
    _cache_ts_funding = now
    return result

# ==========================================
# [Entry Filter] 24h Quote Volume (liquidity floor)
# ==========================================
def fetch_all_24h_volumes() -> dict:
    """
    ดึง 24h quote volume (USDT) ของทุก symbol จาก Binance Spot ในครั้งเดียว
    (/api/v3/ticker/24hr ไม่มี param symbol = คืนค่าทุกเหรียญ)
    คืนค่า dict: {coin: quote_volume_usdt}
    """
    result = {}
    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}/api/v3/ticker/24hr"
        try:
            resp = api_session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"24h volume bulk fetch ({base_url}) → HTTP {resp.status_code}")
                continue
            data = resp.json()
            if not isinstance(data, list):
                continue
            for item in data:
                symbol = item.get("symbol", "")
                if not symbol.endswith("USDT"):
                    continue
                coin = symbol[:-4]
                try:
                    result[coin] = float(item.get("quoteVolume", 0.0))
                except (TypeError, ValueError):
                    continue
            if result:
                logger.info(f"✅ 24h Volume bulk fetch สำเร็จ ({len(result)} symbols)")
                return result
        except Exception as e:
            logger.warning(f"24h volume bulk fetch ({base_url}) → {e}")
    return result

def bulk_fetch_24h_volume(coins: list) -> dict:
    """Bulk-cached 24h quote volume lookup, keyed by coin symbol."""
    global _cache_24hvol, _cache_ts_24hvol
    now = time.time()
    if _cache_24hvol and (now - _cache_ts_24hvol) < CACHE_TTL_SECONDS:
        return _cache_24hvol
    result = fetch_all_24h_volumes()
    _cache_24hvol = result
    _cache_ts_24hvol = now
    return result

def get_volume_filter_info(coin: str, volume_data: dict) -> dict:
    """
    ตรวจ 24h quote volume เทียบ MIN_24H_QUOTE_VOLUME_USDT
    - เหรียญใน VOLUME_FILTER_WHITELIST ผ่านเสมอ (ป้องกัน Binance API return ผิดปกติ)
    - ถ้า volume data ดูผิดปกติ (< $1,000 สำหรับ top coin) → fail open
    คืน dict: {"has_data": bool, "volume_24h": float|None, "passes": bool, "label": str}
    """
    info = {"has_data": False, "volume_24h": None, "passes": True, "label": ""}

    # Whitelisted coins ผ่านเสมอ
    if coin in VOLUME_FILTER_WHITELIST:
        info["label"] = f"✅ {coin} ใน whitelist — ข้าม volume filter"
        return info

    vol = volume_data.get(coin)
    if vol is None:
        return info  # fail open

    # Sanity check: ถ้า volume < $10,000 สำหรับ coin ทั่วไป = API likely returned base volume
    # ไม่ใช่ quote volume → fail open (อย่าบล็อก)
    if vol < 10_000:
        logger.warning(f"⚠️ {coin}: 24h volume=${vol:,.0f} ต่ำผิดปกติ — อาจเป็น base volume ไม่ใช่ USDT → ข้าม filter")
        info["label"] = f"⚠️ Volume data ผิดปกติ (${vol:,.0f}) — ข้าม filter"
        return info  # fail open

    info.update({"has_data": True, "volume_24h": vol})
    if vol < MIN_24H_QUOTE_VOLUME_USDT:
        info["passes"] = False
        info["label"] = (
            f"🔇 24h Volume ${vol:,.0f} ต่ำกว่าเกณฑ์ "
            f"(${MIN_24H_QUOTE_VOLUME_USDT:,.0f}) — สภาพคล่องต่ำ"
        )
    return info

def get_funding_filter_info(coin: str, funding_data: dict) -> dict:
    """
    ตีความ Funding Rate สำหรับเหรียญนี้:
    - block: ห้ามเปิด Long ใหม่ (funding สูงเกินไป — long ฝั่งจ่ายแพง = crowded long)
    - warn:  ยังเปิดได้ แต่มีคำเตือน / หักคะแนนเล็กน้อย
    """
    info = {
        "has_data": False, "funding_rate": None,
        "block_long": False, "warn_long": False,
        "funding_label": "⚪ N/A Funding"
    }
    fd = funding_data.get(coin)
    if not fd:
        return info

    fr = fd["funding_rate"]
    info.update({"has_data": True, "funding_rate": fr})
    fr_pct = fr * 100

    if fr >= FUNDING_RATE_MAX_LONG:
        info["block_long"] = True
        info["funding_label"] = f"🔥 Funding สูงมาก ({fr_pct:.3f}%/8h) — Long แน่นเกินไป งดเปิดใหม่"
    elif fr >= FUNDING_RATE_WARN_LONG:
        info["warn_long"] = True
        info["funding_label"] = f"🟡 Funding ค่อนข้างสูง ({fr_pct:.3f}%/8h) — ระมัดระวัง"
    elif fr <= 0:
        info["funding_label"] = f"🟢 Funding ติดลบ/เป็นกลาง ({fr_pct:.3f}%/8h) — เอื้อต่อ Long"
    else:
        info["funding_label"] = f"⚪ Funding ปกติ ({fr_pct:.3f}%/8h)"

    return info

def fetch_oi_history(coin: str) -> pd.DataFrame | None:
    """
    ดึงประวัติ Open Interest ของเหรียญจาก Binance Futures
    (/futures/data/openInterestHist) — เรียกเฉพาะ coin ที่เป็น candidate signal
    เพื่อไม่ให้เปลือง API quota กับทุกเหรียญทุกรอบ

    Cached per-run (in _cache_oi_run) เพื่อไม่ fetch ซ้ำถ้า coin เดียวกัน
    ถูกประเมินมากกว่าหนึ่งครั้งใน scan รอบเดียว
    """
    if coin in _cache_oi_run:
        return _cache_oi_run[coin]

    symbol = f"{coin}USDT"
    result = None
    for base_url in BINANCE_FUTURES_ENDPOINTS:
        url = f"{base_url}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": OI_HIST_PERIOD, "limit": OI_HIST_LIMIT}
        try:
            resp = api_session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"{coin} OI hist → HTTP {resp.status_code}")
                continue
            data = resp.json()
            if not data or not isinstance(data, list):
                continue
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            df = df.dropna(subset=["sumOpenInterest"]).set_index("timestamp")
            if len(df) < 2:
                continue
            result = df
            break
        except Exception as e:
            logger.warning(f"{coin} OI hist → {e}")

    _cache_oi_run[coin] = result
    return result

def get_oi_filter_info(coin: str, df_4h: pd.DataFrame) -> dict:
    """
    ตรวจ Open Interest trend เทียบกับการเคลื่อนไหวของราคาในช่วงเดียวกัน:
    - ราคาขึ้น + OI โต -> สัญญาณ Long ใหม่เข้าจริง (good)
    - ราคาขึ้น + OI ไม่โต/ลด -> อาจเป็น Short-covering เท่านั้น (warn)
    - มีข้อมูลไม่พอ -> N/A, ไม่ส่งผลต่อคะแนน
    """
    info = {
        "has_data": False, "oi_change_pct": None, "price_change_pct": None,
        "weak_conviction": False, "oi_label": "⚪ N/A OI"
    }
    oi_df = fetch_oi_history(coin)
    if oi_df is None or len(oi_df) < 2 or df_4h is None or len(df_4h) < OI_HIST_LIMIT:
        return info

    try:
        oi_now, oi_prev = oi_df["sumOpenInterest"].iloc[-1], oi_df["sumOpenInterest"].iloc[0]
        if oi_prev <= 0:
            return info
        oi_change_pct = ((oi_now - oi_prev) / oi_prev) * 100

        price_now  = df_4h["close"].iloc[-1]
        price_prev = df_4h["close"].iloc[-OI_HIST_LIMIT]
        price_change_pct = ((price_now - price_prev) / price_prev) * 100 if price_prev > 0 else 0.0

        info.update({
            "has_data": True,
            "oi_change_pct": round(oi_change_pct, 2),
            "price_change_pct": round(price_change_pct, 2),
        })

        if price_change_pct > 0 and oi_change_pct < OI_PRICE_UP_NO_OI_GROWTH_PCT:
            info["weak_conviction"] = True
            info["oi_label"] = (
                f"⚠️ ราคา +{price_change_pct:.1f}% แต่ OI {oi_change_pct:+.1f}% "
                f"— อาจเป็น Short-Covering (conviction ต่ำ)"
            )
        elif price_change_pct > 0 and oi_change_pct >= OI_PRICE_UP_NO_OI_GROWTH_PCT:
            info["oi_label"] = f"🟢 ราคา +{price_change_pct:.1f}% & OI +{oi_change_pct:.1f}% — Long เข้าจริง"
        else:
            info["oi_label"] = f"⚪ OI {oi_change_pct:+.1f}% / Price {price_change_pct:+.1f}%"

    except Exception as e:
        logger.warning(f"{coin} OI filter calc error: {e}")

    return info

# ==========================================
# [#4] BTC Dominance (BTC.D) Filter
# ==========================================
def _load_btcd_history() -> list:
    if not os.path.exists(BTC_DOMINANCE_HISTORY_FILE):
        return []
    try:
        with open(BTC_DOMINANCE_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"⚠️ อ่าน {BTC_DOMINANCE_HISTORY_FILE} ไม่ได้: {e}")
        return []

def _save_btcd_history(history: list) -> None:
    try:
        with open(BTC_DOMINANCE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"❌ บันทึก {BTC_DOMINANCE_HISTORY_FILE} ล้มเหลว: {e}")

def fetch_btc_dominance() -> float | None:
    """ดึง BTC Market Cap Dominance (%) ปัจจุบันจาก CoinGecko /api/v3/global"""
    url = "https://api.coingecko.com/api/v3/global"
    try:
        resp = api_session.get(url, timeout=15)
        if resp.status_code == 429:
            logger.warning("⚠️ BTC.D CoinGecko rate limit — รอ 15s")
            time.sleep(5)
            resp = api_session.get(url, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"BTC.D fetch → HTTP {resp.status_code}")
            return None
        data = resp.json().get("data", {})
        btc_d = data.get("market_cap_percentage", {}).get("btc")
        if btc_d is None:
            return None
        return float(btc_d)
    except Exception as e:
        logger.warning(f"BTC.D fetch error: {e}")
        return None

def get_btc_dominance_info() -> dict:
    """
    ดึงค่า BTC.D ปัจจุบัน + เทียบกับค่าย้อนหลัง (เก็บ history เป็นไฟล์ JSON
    เพราะ GitHub Actions เป็น ephemeral VM, in-memory cache ไม่ข้าม run)

    คืน dict: {
      "has_data": bool, "btc_d": float, "btc_d_change_pct": float,
      "regime": "alt_unfriendly" | "alt_friendly" | "neutral",
      "btcd_label": str
    }
    """
    global _cache_btcd, _cache_ts_btcd
    now = time.time()
    if _cache_btcd and (now - _cache_ts_btcd) < CACHE_TTL_SECONDS:
        return _cache_btcd

    result = {
        "has_data": False, "btc_d": None, "btc_d_change_pct": None,
        "regime": "neutral", "btcd_label": "⚪ N/A BTC.D"
    }

    btc_d_now = fetch_btc_dominance()
    if btc_d_now is None:
        _cache_btcd, _cache_ts_btcd = result, now
        return result

    history = _load_btcd_history()
    history.append({"ts": datetime.now(timezone.utc).isoformat(), "btc_d": btc_d_now})
    # Keep only the most recent N snapshots
    history = history[-BTC_DOMINANCE_LOOKBACK_SNAPSHOTS:]
    _save_btcd_history(history)

    result.update({"has_data": True, "btc_d": round(btc_d_now, 2)})

    if len(history) >= 2:
        btc_d_prev = history[0]["btc_d"]
        if btc_d_prev > 0:
            change_pct = ((btc_d_now - btc_d_prev) / btc_d_prev) * 100
            result["btc_d_change_pct"] = round(change_pct, 3)

            if change_pct >= BTC_DOMINANCE_RISING_THRESHOLD_PCT:
                result["regime"] = "alt_unfriendly"
                result["btcd_label"] = (
                    f"🔴 BTC.D {btc_d_now:.2f}% (+{change_pct:.2f}%) — "
                    f"เงินไหลเข้า BTC, ระมัดระวัง Alt Long"
                )
            elif change_pct <= BTC_DOMINANCE_FALLING_THRESHOLD_PCT:
                result["regime"] = "alt_friendly"
                result["btcd_label"] = (
                    f"🟢 BTC.D {btc_d_now:.2f}% ({change_pct:.2f}%) — "
                    f"เงินไหลเข้า Alt (Alt Season Bias)"
                )
            else:
                result["btcd_label"] = f"⚪ BTC.D {btc_d_now:.2f}% ({change_pct:+.2f}%) — ทรงตัว"
    else:
        result["btcd_label"] = f"⚪ BTC.D {btc_d_now:.2f}% (ยังไม่มีข้อมูลย้อนหลังพอสำหรับเทรนด์)"

    _cache_btcd, _cache_ts_btcd = result, now
    return result

def get_btc_dominance_filter_for_coin(coin: str, btcd_info: dict) -> dict:
    """
    แปลงผล BTC.D regime เป็นผลกระทบต่อคะแนนของแต่ละเหรียญ
    - BTC ไม่ถูกกระทบ (BTC.D ขึ้นไม่ใช่ปัญหาของ BTC เอง)
    - Altcoin: alt_unfriendly -> หักคะแนน, alt_friendly -> บวกคะแนนเล็กน้อย
    """
    info = {"score_delta": 0, "note": ""}
    if not btcd_info.get("has_data") or coin == "BTC":
        return info

    regime = btcd_info.get("regime", "neutral")
    if regime == "alt_unfriendly":
        info["score_delta"] = -8
        info["note"] = btcd_info["btcd_label"]
    elif regime == "alt_friendly":
        info["score_delta"] = 3
        info["note"] = btcd_info["btcd_label"]
    return info

# ==========================================
# [#5] Macro / Economic News Filter
# ==========================================
def fetch_macro_calendar_today() -> list:
    """
    ดึง Economic Calendar วันนี้ (UTC) จาก Financial Modeling Prep (free tier)
    คืน list ของ event dicts ที่ผ่านการกรอง: ประเทศ US + คำสำคัญ high-impact
    """
    global _cache_macro, _cache_ts_macro
    now = time.time()
    if _cache_macro and (now - _cache_ts_macro) < CACHE_TTL_SECONDS:
        return _cache_macro.get("events", [])

    result = {"events": []}
    if not MACRO_FILTER_ENABLED:
        _cache_macro, _cache_ts_macro = result, now
        return result["events"]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {"from": today, "to": today, "apikey": FMP_API_KEY}
    # FMP has been migrating v3 endpoints to /stable/. Try /stable/ first,
    # fall back to the legacy /api/v3/ path if it 404s/fails.
    urls = [
        "https://financialmodelingprep.com/stable/economic-calendar",
        "https://financialmodelingprep.com/api/v3/economic_calendar",
    ]
    data = None
    for url in urls:
        try:
            resp = api_session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Macro calendar fetch ({url}) → HTTP {resp.status_code}")
                continue
            parsed = resp.json()
            if isinstance(parsed, list):
                data = parsed
                break
            else:
                logger.warning(f"Macro calendar fetch ({url}) → unexpected response shape")
        except Exception as e:
            logger.warning(f"Macro calendar fetch ({url}) error: {e}")

    if data is None:
        _cache_macro, _cache_ts_macro = result, now
        return result["events"]

    try:
        filtered = []
        for item in data:
            country = str(item.get("country", "")).upper()
            event_name = str(item.get("event", ""))
            if country != MACRO_FILTER_COUNTRY:
                continue
            event_lower = event_name.lower()
            if any(kw in event_lower for kw in MACRO_HIGH_IMPACT_KEYWORDS):
                filtered.append({
                    "event": event_name,
                    "date": item.get("date", ""),
                    "impact": item.get("impact", ""),
                })

        result["events"] = filtered
        if filtered:
            logger.info(f"📅 Macro Calendar: พบ {len(filtered)} high-impact events วันนี้")
    except Exception as e:
        logger.warning(f"Macro calendar parse error: {e}")

    _cache_macro, _cache_ts_macro = result, now
    return result["events"]

def get_macro_filter_info() -> dict:
    """
    ตรวจว่าวันนี้มี high-impact macro event ที่ใกล้เวลาปัจจุบัน (ภายใน
    MACRO_BLOCK_WINDOW_HOURS ชม.) หรือไม่ -> ถ้ามี ให้ block สัญญาณเข้าใหม่ทั้งหมด

    คืน dict: {"has_data": bool, "block_new_entries": bool, "events": [...], "macro_label": str}
    """
    info = {"has_data": False, "block_new_entries": False, "events": [], "macro_label": ""}

    if not MACRO_FILTER_ENABLED:
        info["macro_label"] = "⚪ Macro Filter ปิดอยู่ (ไม่ได้ตั้งค่า FMP_API_KEY)"
        return info

    events = fetch_macro_calendar_today()
    info["has_data"] = True
    info["events"] = events

    if not events:
        info["macro_label"] = "🟢 ไม่มี Macro Event สำคัญวันนี้"
        return info

    now_utc = datetime.now(timezone.utc)
    near_events = []
    for ev in events:
        try:
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            if ev_time.tzinfo is None:
                ev_time = ev_time.replace(tzinfo=timezone.utc)
            hours_diff = abs((ev_time - now_utc).total_seconds()) / 3600.0
            if hours_diff <= MACRO_BLOCK_WINDOW_HOURS:
                near_events.append(ev["event"])
        except Exception:
            # If date parsing fails but it's flagged "today", be conservative
            near_events.append(ev["event"])

    if near_events:
        info["block_new_entries"] = True
        info["macro_label"] = (
            f"🚨 <b>Macro Event ใกล้เวลานี้ (±{MACRO_BLOCK_WINDOW_HOURS}ชม.):</b> "
            + ", ".join(near_events) + " — งดเปิด Position ใหม่"
        )
    else:
        event_names = ", ".join(ev["event"] for ev in events)
        info["macro_label"] = f"🟡 มี Macro Event วันนี้ (ไม่ใกล้เวลานี้): {event_names}"

    return info


def bulk_fetch_4h(coins: list) -> dict:
    global _cache_4h, _cache_ts_4h
    now = time.time()
    if _cache_4h and (now - _cache_ts_4h) < CACHE_TTL_SECONDS:
        return _cache_4h
    result = {}
    for coin in coins:
        df = get_historical_data(coin)
        if df is not None:
            result[coin] = df
        time.sleep(API_RATE_LIMIT_DELAY)
    _cache_4h = result
    _cache_ts_4h = now
    return result

def bulk_fetch_1d(coins: list) -> dict:
    global _cache_1d, _cache_ts_1d
    now = time.time()
    if _cache_1d and (now - _cache_ts_1d) < CACHE_TTL_SECONDS:
        return _cache_1d
    result = {}
    for coin in coins:
        df = get_histoday_data(coin)
        if df is not None:
            result[coin] = df
        time.sleep(API_RATE_LIMIT_DELAY)
    _cache_1d = result
    _cache_ts_1d = now
    return result


def get_1h_data(coin: str) -> pd.DataFrame | None:
    """
    [Approach A] ดึง 1H OHLC จาก Binance สำหรับ Entry Timing
    ใช้ 100 แท่ง (~4 วัน) — พอสำหรับคำนวณ RSI 1H และ slope
    """
    symbol = f"{coin}USDT"
    limit  = 100
    df = _fetch_from_binance(symbol, "1h", limit, coin, "1H")
    if df is not None and len(df) >= 20:
        return df
    try:
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": limit}
        resp = api_session.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            df2 = _parse_binance_klines(resp.json(), coin, "1H-Vision")
            if df2 is not None and len(df2) >= 20:
                return df2
    except Exception:
        pass
    return None


def bulk_fetch_1h(coins: list) -> dict:
    """[Approach A] Bulk fetch 1H data for all coins"""
    global _cache_1h, _cache_ts_1h
    now = time.time()
    if _cache_1h and (now - _cache_ts_1h) < CACHE_TTL_SECONDS:
        return _cache_1h
    result = {}
    for coin in coins:
        df = get_1h_data(coin)
        if df is not None:
            result[coin] = df
        time.sleep(API_RATE_LIMIT_DELAY)
    _cache_1h = result
    _cache_ts_1h = now
    return result


def get_1h_rsi_info(df_1h: pd.DataFrame) -> dict:
    """
    [Approach A] วิเคราะห์ RSI 1H สำหรับ entry timing:
    - is_bouncing: RSI 1H กำลัง bounce จาก trough (ดีดขึ้น <= 3 แท่ง 1H)
    - timing_score: 0-10 (บวกใน confluence score)
    - timing_label: ข้อความสำหรับ Telegram
    """
    result = {
        "rsi_1h": None, "slope_1h": 0.0,
        "is_bouncing": False, "timing_score": 0,
        "timing_label": "", "trough_rsi": None,
    }
    if df_1h is None or len(df_1h) < 20:
        return result
    try:
        df = df_1h.copy()
        if "RSI" not in df.columns:
            delta = df["close"].diff()
            gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
            ag = gain.ewm(com=RSI_PERIOD-1, adjust=False).mean()
            al = loss.ewm(com=RSI_PERIOD-1, adjust=False).mean()
            rs = ag / al.replace(0, np.nan)
            fill = np.where(ag > 1e-10, 100.0, 50.0)
            df["RSI"] = (100 - 100/(1+rs)).fillna(pd.Series(fill, index=rs.index))

        rsi_s    = df["RSI"].iloc[-20:]
        rsi_now  = float(rsi_s.iloc[-1])
        slope_1h = float(rsi_s.diff().iloc[-3:].mean())
        rsi_min  = float(rsi_s.min())
        min_idx  = rsi_s.argmin()
        bars_since = len(rsi_s) - 1 - min_idx

        result.update({"rsi_1h": round(rsi_now, 2),
                        "slope_1h": round(slope_1h, 3),
                        "trough_rsi": round(rsi_min, 2)})

        is_bouncing = (rsi_min <= 40 and bars_since <= 3
                       and slope_1h > 0 and rsi_now > rsi_min)
        result["is_bouncing"] = is_bouncing

        score = 0
        if rsi_now <= RSI_OVERSOLD: score += 4
        elif rsi_now <= 40:         score += 2
        if slope_1h > 2:    score += 3
        elif slope_1h > 0:  score += 2
        if is_bouncing:     score += 3
        result["timing_score"] = min(score, 10)

        if is_bouncing and rsi_min <= RSI_OVERSOLD:
            result["timing_label"] = (
                f"\u26a1 1H RSI Bounce! trough={rsi_min:.1f}\u2192{rsi_now:.1f} "
                f"(+{rsi_now-rsi_min:.1f}pts ใน {bars_since} แท่ง 1H) \u2014 เข้าได้เลย"
            )
        elif rsi_now <= RSI_OVERSOLD:
            result["timing_label"] = f"\U0001f535 1H RSI Oversold ({rsi_now:.1f}) \u2014 รอสัญญาณดีด"
        elif slope_1h > 0 and rsi_now <= 45:
            result["timing_label"] = f"\U0001f7e1 1H RSI ดีด ({rsi_now:.1f} slope:{slope_1h:+.1f})"
        else:
            result["timing_label"] = f"\u26aa 1H RSI: {rsi_now:.1f}"
    except Exception as e:
        logger.debug(f"get_1h_rsi_info error: {e}")
    return result


def bulk_fetch_onchain(coins: list) -> dict:
    global _cache_onchain, _cache_ts_onchain
    now = time.time()
    if _cache_onchain and (now - _cache_ts_onchain) < CACHE_TTL_SECONDS:
        return _cache_onchain
    result = {}
    for coin in coins:
        if coin in ONCHAIN_SUPPORTED_COINS:
            result[coin] = get_onchain_data(coin)
            time.sleep(API_RATE_LIMIT_DELAY)
    _cache_onchain = result
    _cache_ts_onchain = now
    return result

# ==========================================
# Core Indicators & Analysis
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ indicators — มี 2 fixes สำหรับ EIGEN และเหรียญ low-liquidity:

    Fix 1: Strip leading zero-price rows ONLY (not constant-price rows)
      ตัดเฉพาะ rows ที่ close = 0 หรือ NaN ออก (ก่อน listing จริง)
      ไม่ตัด rows ที่ราคา constant แต่ไม่เป็น 0 เพราะอาจทำลาย EMA ของ BTC/ETH

    Fix 2: RSI fillna(50) แทน fillna(100)
      เดิม: loss=0 → rs=NaN → RSI=fillna(100) → RSI 98-100 (false overbought)
      ใหม่: loss=0 → rs=NaN → RSI=fillna(50) → neutral, ไม่ trigger exit watch

    Fix 3: RSI Sanity Clamp หลังคำนวณ
      ถ้า RSI > 90 และ price variance ต่ำผิดปกติ (cv<0.05 หรือ unique<=3) → clamp ≤75
      เป็น safety net สำหรับ EIGEN-type data ที่ผ่าน Fix 1 มาได้
    """
    # Fix 1: Strip leading zero-price rows เท่านั้น (ไม่ strip constant-price rows)
    # เพราะการ strip constant-price จะทำลาย EMA ของเหรียญปกติที่มี low-vol period
    if (df["close"] == 0).any():
        first_nonzero_mask = df["close"] > 0
        if first_nonzero_mask.any():
            df = df.loc[first_nonzero_mask.idxmax():].copy()

    close, high, low = df["close"], df["high"], df["low"]
    df["EMA_50"]  = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LONG,  adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_raw = 100 - (100 / (1 + rs))
    # Conditional fillna:
    #   avg_gain > 0, avg_loss = 0 → all gains, truly overbought → RSI = 100
    #   avg_gain = 0, avg_loss = 0 → no price movement (frozen) → RSI = 50 (neutral)
    fill_values = np.where(avg_gain > 1e-10, 100.0, 50.0)
    df["RSI"] = rsi_raw.fillna(pd.Series(fill_values, index=rsi_raw.index))

    # Fix 3: RSI Sanity Clamp — เฉพาะ degenerate data เท่านั้น
    # threshold ใช้ n_unique <= 3 เป็น primary (จาก EIGEN log: unique=2/20)
    # cv < 0.005 เป็น secondary (ระมัดระวัง — steady uptrend มี cv ~0.02+)
    last_rsi = df["RSI"].iloc[-1]
    if last_rsi > 90:
        recent_close = df["close"].iloc[-20:]
        mean_c = recent_close.mean()
        cv_check = recent_close.std() / mean_c if mean_c > 0 else 0
        n_uniq = recent_close.nunique()
        # Clamp เฉพาะกรณี degenerate จริงๆ:
        # n_unique <= 3 = มีแค่ 2-3 ราคาใน 20 แท่ง (EIGEN: unique=2)
        # cv < 0.005 = variance น้อยกว่า 0.5% (ต่ำกว่า steady uptrend ~1-2%)
        if n_uniq <= 3 or cv_check < 0.005:
            df["RSI"] = df["RSI"].clip(upper=75)
            logger.warning(
                f"RSI Clamp: RSI={last_rsi:.1f} → ≤75 "
                f"(cv={cv_check:.4f}, unique={n_uniq}/20)"
            )

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    up_move, down_move = high - high.shift(1), low.shift(1) - low
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_safe = df["ATR"].replace(0, np.nan)
    plus_di  = 100 * (pd.Series(plus_dm,  index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    dx       = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)

    df["VOL_MA20"] = df["volumeto"].rolling(20).mean()
    return df

def confirm_reversal_candle(df: pd.DataFrame) -> dict:
    """
    [Advanced] ตรวจ Bullish Reversal Candle Patterns — 10 รูปแบบ พร้อม context filter
    Context filter: pattern ต้องอยู่ใกล้ key level (oversold RSI / EMA / FVG zone)
    จึงจะได้คะแนนเต็ม — ป้องกัน false signal กลางอากาศ

    Patterns (strength weight):
      3pt: Morning Star, Three White Soldiers, Bullish Engulfing (ที่ key level)
      2pt: Bullish Engulfing (ทั่วไป), Tweezer Bottom, Piercing Line
      1pt: Hammer, Inverted Hammer, Bullish Harami, Bullish Doji Star
    """
    result = {
        "patterns_found": [],
        "reversal_strength": 0,
        "reversal_label": "⬜ ไม่มี Candle ยืนยัน",
        "at_key_level": False,
    }
    if len(df) < 4:
        return result

    c0, c1, c2, c3 = df.iloc[-1], df.iloc[-2], df.iloc[-3], df.iloc[-4]
    body0  = abs(c0["close"] - c0["open"])
    body1  = abs(c1["close"] - c1["open"])
    body2  = abs(c2["close"] - c2["open"])
    avg_body = (df["close"] - df["open"]).abs().rolling(10).mean().iloc[-1]
    avg_body = avg_body if not pd.isna(avg_body) and avg_body > 0 else body0

    rsi_now = df["RSI"].iloc[-1] if "RSI" in df.columns else 50.0
    ema50   = df["EMA_50"].iloc[-1]  if "EMA_50"  in df.columns else None
    ema200  = df["EMA_200"].iloc[-1] if "EMA_200" in df.columns else None

    # Context: อยู่ใกล้ key level ไหม?
    at_key = (
        rsi_now <= 40
        or (ema50  is not None and c0["close"] >= ema50  * 0.98 and c0["close"] <= ema50  * 1.02)
        or (ema200 is not None and c0["close"] >= ema200 * 0.98 and c0["close"] <= ema200 * 1.02)
    )
    result["at_key_level"] = at_key
    score = 0
    found = []

    # --- 3pt patterns ---
    # Morning Star: big red, small body doji/star, big green closing above midpoint of red
    if (c2["close"] < c2["open"] and body2 > avg_body
            and body1 < body2 * 0.4
            and c0["close"] > c0["open"]
            and c0["close"] > (c2["open"] + c2["close"]) / 2):
        s = 3 + (1 if at_key else 0)
        score += s; found.append(f"Morning Star({s}pt)")

    # Three White Soldiers: 3 consecutive bullish candles, each closing near high
    if (c0["close"] > c0["open"] and c1["close"] > c1["open"] and c2["close"] > c2["open"]
            and c0["close"] > c1["close"] > c2["close"]
            and (c0["high"] - c0["close"]) < body0 * 0.3
            and (c1["high"] - c1["close"]) < body1 * 0.3
            and body0 > avg_body * 0.8 and body1 > avg_body * 0.8):
        s = 3 + (1 if at_key else 0)
        score += s; found.append(f"Three White Soldiers({s}pt)")

    # Bullish Engulfing at key level
    if (c1["close"] < c1["open"] and c0["close"] > c0["open"]
            and c0["open"] <= c1["close"] and c0["close"] >= c1["open"]):
        s = (3 if at_key else 2)
        score += s; found.append(f"Bullish Engulfing({s}pt)")

    # --- 2pt patterns ---
    # Tweezer Bottom: two candles with nearly identical lows (within 0.2%)
    if (abs(c0["low"] - c1["low"]) / max(c1["low"], 1e-10) < 0.002
            and c1["close"] < c1["open"] and c0["close"] > c0["open"]):
        score += 2; found.append("Tweezer Bottom(2pt)")

    # Piercing Line: red candle, then green opens below red's low and closes above midpoint
    if (c1["close"] < c1["open"]
            and c0["open"] < c1["close"]
            and c0["close"] > c0["open"]
            and c0["close"] > (c1["open"] + c1["close"]) / 2
            and c0["close"] < c1["open"]):
        score += 2; found.append("Piercing Line(2pt)")

    # --- 1pt patterns ---
    # Hammer: small body at top, lower wick >= 2x body, upper wick <= 0.3x body
    if (body0 > 0
            and (min(c0["open"], c0["close"]) - c0["low"]) >= 2 * body0
            and (c0["high"] - max(c0["open"], c0["close"])) <= 0.3 * body0):
        score += 1; found.append("Hammer(1pt)")

    # Inverted Hammer (after downtrend): long upper wick, small body near low
    if (body0 > 0
            and (c0["high"] - max(c0["open"], c0["close"])) >= 2 * body0
            and (min(c0["open"], c0["close"]) - c0["low"]) <= 0.3 * body0
            and c0["close"] > c0["open"]):
        score += 1; found.append("Inverted Hammer(1pt)")

    # Bullish Harami: large red candle contains small green candle
    if (c1["close"] < c1["open"] and body1 > avg_body
            and c0["close"] > c0["open"]
            and c0["open"] > c1["close"] and c0["close"] < c1["open"]
            and body0 < body1 * 0.5):
        score += 1; found.append("Bullish Harami(1pt)")

    # Bullish Doji Star: doji after downtrend (body < 10% of range)
    rng0 = c0["high"] - c0["low"]
    if rng0 > 0 and body0 / rng0 < 0.1 and c1["close"] < c1["open"]:
        score += 1; found.append("Doji Star(1pt)")

    result["patterns_found"] = found
    result["reversal_strength"] = score

    if score >= 5:
        result["reversal_label"] = f"🕯️ <b>Candle ยืนยันแข็งแกร่งมาก</b> ({', '.join(found)})"
    elif score >= 3:
        result["reversal_label"] = f"🕯️ <b>Candle ยืนยันแข็งแกร่ง</b> ({', '.join(found)})"
    elif score >= 1:
        result["reversal_label"] = f"🕯️ Candle ยืนยันปานกลาง ({', '.join(found)})"
    return result

def confirm_bearish_reversal_candle(df: pd.DataFrame) -> dict:
    """
    [Advanced] ตรวจ Bearish Reversal Candle Patterns — 10 รูปแบบ พร้อม context filter
    Context filter: pattern ต้องอยู่ใกล้ overbought zone / resistance ถึงจะได้คะแนนเต็ม

    Patterns (strength weight):
      3pt: Evening Star, Three Black Crows, Bearish Engulfing (ที่ resistance)
      2pt: Bearish Engulfing (ทั่วไป), Tweezer Top, Dark Cloud Cover
      1pt: Shooting Star, Hanging Man, Bearish Harami, Bearish Doji Star
    """
    result = {
        "patterns_found": [],
        "bearish_strength": 0,
        "bearish_label": "⬜ ไม่มี Candle เตือนกลับตัว",
        "at_resistance": False,
    }
    if len(df) < 4:
        return result

    c0, c1, c2, c3 = df.iloc[-1], df.iloc[-2], df.iloc[-3], df.iloc[-4]
    body0  = abs(c0["close"] - c0["open"])
    body1  = abs(c1["close"] - c1["open"])
    body2  = abs(c2["close"] - c2["open"])
    avg_body = (df["close"] - df["open"]).abs().rolling(10).mean().iloc[-1]
    avg_body = avg_body if not pd.isna(avg_body) and avg_body > 0 else body0

    rsi_now = df["RSI"].iloc[-1] if "RSI" in df.columns else 50.0
    ema50   = df["EMA_50"].iloc[-1]  if "EMA_50"  in df.columns else None
    ema200  = df["EMA_200"].iloc[-1] if "EMA_200" in df.columns else None

    # Context: อยู่ใกล้ resistance / overbought ไหม?
    at_res = (
        rsi_now >= 60
        or (ema50  is not None and c0["close"] >= ema50  * 0.99 and c0["close"] <= ema50  * 1.01)
    )
    result["at_resistance"] = at_res
    score = 0
    found = []

    # --- 3pt patterns ---
    # Evening Star: big green, small body, big red closing below midpoint of green
    if (c2["close"] > c2["open"] and body2 > avg_body
            and body1 < body2 * 0.4
            and c0["close"] < c0["open"]
            and c0["close"] < (c2["open"] + c2["close"]) / 2):
        s = 3 + (1 if at_res else 0)
        score += s; found.append(f"Evening Star({s}pt)")

    # Three Black Crows: 3 consecutive bearish candles, each closing near low
    if (c0["close"] < c0["open"] and c1["close"] < c1["open"] and c2["close"] < c2["open"]
            and c0["close"] < c1["close"] < c2["close"]
            and (c0["close"] - c0["low"]) < body0 * 0.3
            and (c1["close"] - c1["low"]) < body1 * 0.3
            and body0 > avg_body * 0.8 and body1 > avg_body * 0.8):
        s = 3 + (1 if at_res else 0)
        score += s; found.append(f"Three Black Crows({s}pt)")

    # Bearish Engulfing
    if (c1["close"] > c1["open"] and c0["close"] < c0["open"]
            and c0["open"] >= c1["close"] and c0["close"] <= c1["open"]):
        s = (3 if at_res else 2)
        score += s; found.append(f"Bearish Engulfing({s}pt)")

    # --- 2pt patterns ---
    # Tweezer Top: two candles with nearly identical highs
    if (abs(c0["high"] - c1["high"]) / max(c1["high"], 1e-10) < 0.002
            and c1["close"] > c1["open"] and c0["close"] < c0["open"]):
        score += 2; found.append("Tweezer Top(2pt)")

    # Dark Cloud Cover: green candle, then red opens above green's high and closes below midpoint
    if (c1["close"] > c1["open"]
            and c0["open"] > c1["high"]
            and c0["close"] < c0["open"]
            and c0["close"] < (c1["open"] + c1["close"]) / 2
            and c0["close"] > c1["open"]):
        score += 2; found.append("Dark Cloud Cover(2pt)")

    # --- 1pt patterns ---
    # Shooting Star: long upper wick, small body near low
    if (body0 > 0
            and (c0["high"] - max(c0["open"], c0["close"])) >= 2 * body0
            and (min(c0["open"], c0["close"]) - c0["low"]) <= 0.3 * body0):
        score += 1; found.append("Shooting Star(1pt)")

    # Hanging Man (after uptrend): hammer shape but bearish context
    if (body0 > 0
            and (min(c0["open"], c0["close"]) - c0["low"]) >= 2 * body0
            and (c0["high"] - max(c0["open"], c0["close"])) <= 0.3 * body0
            and c0["close"] < c0["open"]
            and c1["close"] > c1["open"]):
        score += 1; found.append("Hanging Man(1pt)")

    # Bearish Harami: large green candle contains small red candle
    if (c1["close"] > c1["open"] and body1 > avg_body
            and c0["close"] < c0["open"]
            and c0["open"] < c1["close"] and c0["close"] > c1["open"]
            and body0 < body1 * 0.5):
        score += 1; found.append("Bearish Harami(1pt)")

    # Bearish Doji Star: doji after uptrend
    rng0 = c0["high"] - c0["low"]
    if rng0 > 0 and body0 / rng0 < 0.1 and c1["close"] > c1["open"]:
        score += 1; found.append("Bearish Doji Star(1pt)")

    result["patterns_found"] = found
    result["bearish_strength"] = score

    if score >= 5:
        result["bearish_label"] = f"🕯️ <b>Candle เตือนกลับตัวขาลงแข็งแกร่งมาก</b> ({', '.join(found)})"
    elif score >= 3:
        result["bearish_label"] = f"🕯️ <b>Candle เตือนกลับตัวขาลงแข็งแกร่ง</b> ({', '.join(found)})"
    elif score >= 1:
        result["bearish_label"] = f"🕯️ Candle เตือนกลับตัวขาลงปานกลาง ({', '.join(found)})"
    return result

def get_dynamic_atr_multiplier(tier: str, adx: float, atr_pct: float, df: pd.DataFrame | None = None) -> float:
    """
    [Advanced] Adaptive ATR Multiplier ที่ปรับตาม volatility regime จริงๆ

    เดิม: hardcoded threshold (adx>25 +20%, adx<15 -15%)
    ใหม่: ใช้ ATR percentile ของ coin นั้นเอง 90 วันย้อนหลัง เพื่อประเมิน
          ว่าความผันผวนปัจจุบัน "สูง/ต่ำ" เทียบกับตัวเอง (ไม่ใช่ค่าคงที่ข้ามเหรียญ)
          + smooth continuous scaling ไม่กระโดดเป็น step
    """
    base = {"major": 2.0, "mid": 2.5, "small": 3.0}.get(tier, 2.5)

    # --- ADX-based trend strength: smooth scaling แทน step ---
    # ADX 0-15: weak trend -> ลด multiplier (SL กว้างขึ้น)
    # ADX 15-25: normal
    # ADX 25+: strong trend -> เพิ่ม multiplier (เทรนด์ชัด TP ไกลขึ้น)
    adx_safe = float(adx) if not pd.isna(adx) else 20.0
    if adx_safe <= 15:
        adx_factor = 0.80 + (adx_safe / 15) * 0.15   # 0.80-0.95
    elif adx_safe <= 25:
        adx_factor = 0.95 + ((adx_safe - 15) / 10) * 0.10  # 0.95-1.05
    else:
        adx_factor = 1.05 + min((adx_safe - 25) / 25, 1.0) * 0.20  # 1.05-1.25
    base *= adx_factor

    # --- Volatility Regime: ATR percentile of this coin's own history ---
    if df is not None and "ATR" in df.columns and len(df) >= 30:
        try:
            atr_history = df["ATR"].dropna().iloc[-90:]  # up to 90 bars (~15 days 4H)
            if len(atr_history) >= 20:
                current_atr = atr_history.iloc[-1]
                pctile = float((atr_history < current_atr).mean())  # 0.0-1.0
                # Low vol regime (pctile < 0.3): tighter multiplier, price moves less
                # High vol regime (pctile > 0.7): wider multiplier, needs more room
                if pctile < 0.3:
                    base *= 0.85
                elif pctile > 0.7:
                    base *= 1.20
                elif pctile > 0.9:
                    base *= 1.35
        except Exception:
            pass
    else:
        # Fallback: use raw atr_pct thresholds (original logic)
        if atr_pct > 6.0:
            base *= 1.25
        elif atr_pct > 4.0:
            base *= 1.10

    return round(min(max(base, 1.5), 5.0), 2)

def get_correlation_adjusted_position(portfolio: float, risk_pct: float, sl_distance_pct: float, corr_btc: float, active_signals_count: int) -> float:
    base_risk = portfolio * (risk_pct / 100)
    if corr_btc > 0.85: base_risk *= 0.70
    elif corr_btc > 0.70: base_risk *= 0.85
    max_total_risk = portfolio * (MAX_TOTAL_RISK_PCT / 100)
    if active_signals_count > 0:
        base_risk = min(base_risk, max_total_risk / (active_signals_count + 1))
    return min(base_risk / max(sl_distance_pct, 0.01), portfolio * 0.25)

def get_mtf_rsi_alignment(df_4h: pd.DataFrame, df_1d: pd.DataFrame,
                           df_1h: pd.DataFrame | None = None) -> dict:
    """
    [Approach A] Multi-Timeframe RSI — 4 Timeframes: 1H + 4H + 1D + 1W
    1H ใช้สำหรับ entry timing confirmation (ไม่ใช่ primary gate)
    """
    result = {
        "aligned_oversold": False, "aligned_overbought": False,
        "rsi_4h": None, "rsi_1d": None, "rsi_1w": None, "rsi_1h": None,
        "slope_4h": None, "slope_1d": None,
        "mtf_label": "", "confluence_score": 0,
        "mtf_conflict": False,
        # 1H entry timing
        "timing_score": 0, "timing_label": "",
        "is_1h_bouncing": False,
    }
    if df_4h is None or len(df_4h) < 5:
        return result

    # --- 4H RSI + slope ---
    rsi_4h   = float(df_4h["RSI"].iloc[-1]) if "RSI" in df_4h.columns else 50.0
    slope_4h = float(df_4h["RSI"].diff().iloc[-3:].mean()) if "RSI" in df_4h.columns else 0.0
    result.update({"rsi_4h": round(rsi_4h, 2), "slope_4h": round(slope_4h, 3)})

    # --- 1H RSI entry timing (Approach A) ---
    h1_info = get_1h_rsi_info(df_1h)
    result.update({
        "rsi_1h":         h1_info.get("rsi_1h"),
        "timing_score":   h1_info.get("timing_score", 0),
        "timing_label":   h1_info.get("timing_label", ""),
        "is_1h_bouncing": h1_info.get("is_bouncing", False),
    })

    # --- 1D RSI + slope ---
    rsi_1d, slope_1d = 50.0, 0.0
    if df_1d is not None and len(df_1d) >= 5:
        rs_1d = (df_1d["close"].diff().clip(lower=0).ewm(com=13, adjust=False).mean()
                 / (-df_1d["close"].diff().clip(upper=0)).ewm(com=13, adjust=False).mean().replace(0, np.nan))
        rsi_1d_series = 100 - 100 / (1 + rs_1d)
        rsi_1d   = float(rsi_1d_series.iloc[-1])
        slope_1d = float(rsi_1d_series.diff().iloc[-3:].mean())
        result.update({"rsi_1d": round(rsi_1d, 2), "slope_1d": round(slope_1d, 3)})

    # --- 1W RSI (resample from 1D) ---
    rsi_1w = 50.0
    if df_1d is not None and len(df_1d) >= 14:
        try:
            df_w = df_1d.resample("W").agg({"close": "last"}).dropna()
            if len(df_w) >= 14:
                rs_w = (df_w["close"].diff().clip(lower=0).ewm(com=13, adjust=False).mean()
                        / (-df_w["close"].diff().clip(upper=0)).ewm(com=13, adjust=False).mean().replace(0, np.nan))
                rsi_1w = float((100 - 100 / (1 + rs_w)).iloc[-1])
                result["rsi_1w"] = round(rsi_1w, 2)
        except Exception:
            pass

    # --- Confluence Score (4H + 1D + 1W oversold) ---
    def _rsi_oversold_score(rsi: float, slope: float) -> int:
        s = 0
        if rsi <= 20:           s += 4
        elif rsi <= RSI_OVERSOLD: s += 3
        elif rsi <= 40:         s += 1
        if slope > 0:           s += 1
        return s

    def _rsi_overbought_score(rsi: float) -> int:
        if rsi >= 80:               return 3
        elif rsi >= RSI_OVERBOUGHT: return 2
        elif rsi >= 60:             return 1
        return 0

    os_4h = _rsi_oversold_score(rsi_4h, slope_4h)
    os_1d = _rsi_oversold_score(rsi_1d, slope_1d)
    os_1w = _rsi_oversold_score(rsi_1w, 0)
    total_os = os_4h + os_1d + os_1w

    ob_4h = _rsi_overbought_score(rsi_4h)
    ob_1d = _rsi_overbought_score(rsi_1d)
    ob_1w = _rsi_overbought_score(rsi_1w)
    total_ob = ob_4h + ob_1d + ob_1w

    result["confluence_score"] = min(total_os, 10)

    # Conflict detection
    if rsi_4h <= 40 and rsi_1w >= RSI_OVERBOUGHT:
        result["mtf_conflict"] = True
    if rsi_4h >= RSI_OVERBOUGHT and rsi_1w <= 40:
        result["mtf_conflict"] = True

    conflict_note = " ⚠️ MTF Conflict" if result["mtf_conflict"] else ""

    # 1H timing suffix
    h1_rsi_str = f"/1H:{h1_info['rsi_1h']:.1f}" if h1_info.get("rsi_1h") else ""
    bounce_note = " ⚡1H Bounce!" if h1_info.get("is_bouncing") else ""

    rsi_str   = f"4H:{rsi_4h:.1f}/1D:{rsi_1d:.1f}/1W:{rsi_1w:.1f}{h1_rsi_str}"
    slope_str = f"slope4H:{slope_4h:+.1f}"

    if total_os >= 9:
        result.update({"aligned_oversold": True,
                        "mtf_label": f"💎 MTF Oversold แข็งแกร่งมาก ({rsi_str}){bounce_note}{conflict_note}"})
    elif total_os >= 5:
        result.update({"aligned_oversold": True,
                        "mtf_label": f"🔵 MTF Oversold ({rsi_str}) | {slope_str}{bounce_note}{conflict_note}"})
    elif total_os >= 2:
        result["mtf_label"] = f"🟡 MTF RSI อ่อนแรง ({rsi_str}){bounce_note}{conflict_note}"
    elif total_ob >= 6:
        result.update({"aligned_overbought": True,
                        "mtf_label": f"🔴 MTF Overbought ({rsi_str}){conflict_note}"})
    else:
        result["mtf_label"] = f"⚪ MTF Neutral ({rsi_str}){conflict_note}"

    return result


def calculate_signal_score(rsi, bounce_info, candle_info, vol_confirmed, in_fibo_zone, in_ob_zone,
                            in_fvg_zone, weekly_ctx, mtf_info, adx, is_divergence, trend_info,
                            onchain_info, funding_info=None, oi_info=None, btcd_filter=None,
                            div_info=None, bounce_score=None, signal_type: str = "") -> tuple[int, str]:
    """
    [Advanced v2] Weighted Signal Scoring — แก้ไข 4 root causes ที่ทำให้พลาดสัญญาณ:

    Fix 1: RSI scoring ใน bull regime (40-65) ไม่ให้ 0 คะแนน
    Fix 2: Continuation candle scoring — big bullish body ก็นับคะแนนได้
    Fix 3: CoinGecko volume=0 ไม่ทำให้เสียคะแนน (fail-open)
    Fix 4: Divergence-alone ได้คะแนนขึ้นถึง 50 เพื่อไม่พลาด high-quality div signals
    Fix 5: MTF cap เพิ่มจาก 12 → 16 pts
    Fix 6: Signal-type-aware scoring — Momentum/Pullback signals ได้ weight ต่างกัน
    """
    market_regime = "neutral"
    if trend_info:
        ts = trend_info.get("trend_strength", "sideways")
        if ts in ("strong_up", "moderate_up"):      market_regime = "bull"
        elif ts in ("strong_down", "moderate_down"): market_regime = "bear"
        else:                                        market_regime = "sideways"

    # ตรวจประเภท signal เพื่อปรับ weight (Fix 6)
    is_momentum_type = any(kw in signal_type for kw in
                           ["Momentum", "Pullback in Uptrend", "EMA50 Bounce"])
    is_dip_type      = any(kw in signal_type for kw in
                           ["Dip & Rebound", "Divergence", "DEEP REVERSAL", "Deep Support"])

    # --- Regime + Signal-type adjusted weights ---
    if market_regime == "bull":
        if is_momentum_type:
            # Momentum/Pullback: structure + volume + ADX มีน้ำหนักสูงกว่า RSI oversold
            w_rsi=0.8; w_bounce=0.9; w_candle=0.8; w_vol=1.3; w_level=1.0; w_div=0.8; w_mtf=1.0
        else:
            # Dip & Rebound: bounce + level + RSI สำคัญ
            w_rsi=1.0; w_bounce=1.2; w_candle=1.1; w_vol=1.2; w_level=1.1; w_div=1.0; w_mtf=1.1
    elif market_regime == "bear":
        # Bear: ต้องการ confirmation แรงกว่า
        w_rsi=0.8; w_bounce=0.8; w_candle=0.9; w_vol=1.3; w_level=1.2; w_div=1.3; w_mtf=1.2
    else:  # sideways
        w_rsi=0.9; w_bounce=0.9; w_candle=1.0; w_vol=1.4; w_level=1.3; w_div=1.1; w_mtf=1.0

    score = 0.0

    # ─────────────────────────────────────────────────────────────
    # 1. RSI Scoring — Fix 1: bull regime RSI 40-65 ไม่ให้ 0
    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    # 1. RSI Scoring — ปรับให้สอดคล้องกับ RSI ceiling ใหม่ของ
    # Momentum/Pullback/EMA50 Bounce (30-50 แทน 40-65)
    # ตามหลักการ: RSI ยิ่งต่ำ (ใกล้ oversold) ยิ่งเป็นเหตุผลซื้อที่หนักแน่น
    # ─────────────────────────────────────────────────────────────
    if market_regime == "bull" and 30 <= rsi <= 50:
        # RSI อยู่ใน healthy pullback zone ที่ใกล้ oversold มากขึ้น
        rsi_base = 16 if rsi <= 40 else 11
    else:
        rsi_base = (25 if rsi <= RSI_OVERSOLD else
                    15 if rsi <= 40 else
                    8  if rsi <= 50 else 0)
    score += rsi_base * w_rsi

    # ─────────────────────────────────────────────────────────────
    # 2. RSI Bounce Quality
    # ─────────────────────────────────────────────────────────────
    b_score   = bounce_info.get("bounce_score", 0) if bounce_info else 0
    b_quality = bounce_info.get("quality", "none") if bounce_info else "none"
    bounce_pts = {"strong": 20, "moderate": 12, "weak": 5, "none": 0}.get(b_quality, 0)
    bounce_pts = max(bounce_pts, min(b_score * 2, 22))
    score += bounce_pts * w_bounce

    # ─────────────────────────────────────────────────────────────
    # 3. Candle Patterns — Fix 2: Continuation candles for momentum
    # ─────────────────────────────────────────────────────────────
    candle_raw = 0
    if candle_info:
        reversal_str = candle_info.get("reversal_strength", 0)
        candle_raw = min(reversal_str * 6, 18)
        if candle_info.get("at_key_level"):
            candle_raw = min(candle_raw + 4, 22)

    # Fix 2: Momentum signals ไม่มี reversal candle แต่มี continuation candle
    # ให้คะแนนแทนจาก trend_info (bullish bars streak)
    if is_momentum_type and candle_raw == 0 and trend_info:
        consistency = trend_info.get("trend_consistency", 0.0)
        if consistency >= 0.70:   candle_raw = 10  # strong trend continuation
        elif consistency >= 0.60: candle_raw = 6
    score += candle_raw * w_candle

    # ─────────────────────────────────────────────────────────────
    # 4. Key Levels (Fibo / OB / FVG)
    # ─────────────────────────────────────────────────────────────
    level_score = 0
    if in_fibo_zone: level_score += 8
    if in_ob_zone:   level_score += 7
    if in_fvg_zone:  level_score += 5
    levels_hit = sum([in_fibo_zone, in_ob_zone, in_fvg_zone])
    if levels_hit >= 2: level_score += 5
    if levels_hit == 3: level_score += 5
    score += level_score * w_level

    # ─────────────────────────────────────────────────────────────
    # 5. Divergence — Fix 4: ensure strong divergence alone = 50+
    # Regular Bullish Div: price LL, RSI HL → high-quality reversal signal
    #   str=1 → 16pts, str=2 → 22pts, str=3 → 30pts
    # Hidden Bullish Div: price HL, RSI LL → trend continuation
    #   str=1 → 10pts, str=2 → 14pts
    # Divergence-alone bonus: +12 ถ้ามี Regular Div str>=2 แต่ไม่มี factor อื่น
    #   (RSI + Div str=3 + MTF + bonus ≈ 14+30+2.8+10 = 56 → ผ่าน 50)
    # ─────────────────────────────────────────────────────────────
    div_pts = 0
    has_strong_div = False
    if div_info:
        if div_info.get("regular_bullish"):
            str_val = div_info.get("regular_bullish_strength", 1)
            div_pts += 10 + str_val * 6   # str=1→16, str=2→22, str=3→28
            if str_val >= 2:
                has_strong_div = True
        if div_info.get("hidden_bullish"):
            str_val = div_info.get("hidden_bullish_strength", 1)
            div_pts += 8 + str_val * 2    # str=1→10, str=2→12
        score += min(div_pts, 32) * w_div  # cap raised from 26 → 32
    elif is_divergence:
        score += 12 * w_div

    # Divergence-alone bonus: strong divergence without key level + bounce
    # still deserves to surface — divergence is one of the most reliable signals.
    # +12 (not +10) เพื่อให้ div str=2 + RSI pullback + no vol ยังผ่าน 50 ได้
    if has_strong_div and level_score == 0 and bounce_pts == 0:
        score += 12   # unconditional bonus, not weighted

    # ─────────────────────────────────────────────────────────────
    # 6. Volume — Fix 3: volume=0 (CoinGecko) = neutral not penalty
    # ─────────────────────────────────────────────────────────────
    if vol_confirmed:
        score += 10 * w_vol
    # vol_confirmed=False could be CoinGecko (no volume data) or real low vol
    # is_volume_confirmed() returns True for vol=0 (fail-open) so this is safe

    # ─────────────────────────────────────────────────────────────
    # 7. MTF Alignment — Fix 5: raise cap from 12 → 16
    # ─────────────────────────────────────────────────────────────
    mtf_raw = min(mtf_info.get("confluence_score", 0) * 2.5, 16) if mtf_info else 0
    if mtf_info and mtf_info.get("mtf_conflict"):
        mtf_raw -= 8
    score += mtf_raw * w_mtf

    # ─────────────────────────────────────────────────────────────
    # 8. Weekly Context
    # ─────────────────────────────────────────────────────────────
    if weekly_ctx:
        if weekly_ctx.get("weekly_bullish_div"):    score += 6
        elif (weekly_ctx.get("rsi_weekly") or 50) <= 35: score += 4
        if weekly_ctx.get("wyckoff_phase") == "accumulation": score += 5
        if (weekly_ctx.get("nearest_fibo_pct") or 99) <= 1.0: score += 3

    # ─────────────────────────────────────────────────────────────
    # 9. ADX / Trend momentum
    # ─────────────────────────────────────────────────────────────
    adx_safe = float(adx) if not pd.isna(adx) else 20.0
    if adx_safe > 30:   score += 6
    elif adx_safe > 25: score += 4
    elif adx_safe < 15: score -= 6
    if trend_info:
        er = trend_info.get("efficiency_ratio", 0)
        if er >= 0.6 and market_regime == "bull": score += 4
        if market_regime == "bear":               score -= 5

    # ─────────────────────────────────────────────────────────────
    # 10. On-Chain
    # ─────────────────────────────────────────────────────────────
    if onchain_info and onchain_info.get("has_data"):
        oc_trend = onchain_info.get("active_addresses_trend", 0)
        if oc_trend >= 15:    score += 15
        elif oc_trend >= 5:   score += 5
        elif oc_trend <= -10: score -= 15

    # ─────────────────────────────────────────────────────────────
    # 11. Funding Rate
    # ─────────────────────────────────────────────────────────────
    if funding_info and funding_info.get("has_data"):
        if funding_info.get("warn_long"): score -= 5
        fr = funding_info.get("funding_rate", 0)
        if fr is not None and fr < 0:     score += 3

    # ─────────────────────────────────────────────────────────────
    # 12. OI
    # ─────────────────────────────────────────────────────────────
    if oi_info and oi_info.get("has_data") and oi_info.get("weak_conviction"):
        score -= 8

    # ─────────────────────────────────────────────────────────────
    # 13. BTC Dominance
    # ─────────────────────────────────────────────────────────────
    if btcd_filter and btcd_filter.get("score_delta"):
        score += btcd_filter["score_delta"]

    # ─────────────────────────────────────────────────────────────
    # 14. Trend Structure (สำหรับ Momentum/Pullback signals)
    # ─────────────────────────────────────────────────────────────
    if trend_info:
        hh          = trend_info.get("higher_high", False)
        hl          = trend_info.get("higher_low", False)
        er          = trend_info.get("efficiency_ratio", 0.0)
        consistency = trend_info.get("trend_consistency", 0.0)

        if hh and hl and market_regime == "bull": score += 10
        elif hh or hl:                             score += 4

        if er >= 0.6:   score += 8
        elif er >= 0.4: score += 4

        if consistency >= 0.65:   score += 5
        elif consistency >= 0.55: score += 2

    # ─────────────────────────────────────────────────────────────
    # 15. Confluence Interaction Bonus
    # ─────────────────────────────────────────────────────────────
    confluence_factors = sum([
        rsi <= RSI_OVERSOLD,
        b_quality in ("strong", "moderate"),
        in_fibo_zone or in_ob_zone or in_fvg_zone,
        bool(div_info and div_info.get("any_bullish")) or is_divergence,
        vol_confirmed,
        mtf_info.get("aligned_oversold", False) if mtf_info else False,
        bool(trend_info and trend_info.get("higher_high") and trend_info.get("higher_low")),
        bool(trend_info and trend_info.get("efficiency_ratio", 0) >= 0.5),
    ])
    if confluence_factors >= 5:   score += 10
    elif confluence_factors >= 4: score += 5
    elif confluence_factors >= 3: score += 2

    score = max(0, min(100, int(round(score))))
    grade = "🔥 A+" if score >= 75 else "✅ A" if score >= 60 else "🟡 B" if score >= MINIMUM_SIGNAL_SCORE else "⬜ C"
    regime_tag = {"bull": "📈Bull", "bear": "📉Bear", "sideways": "↔️Side"}.get(market_regime, "")
    sig_tag = " 🔄Cont" if is_momentum_type else ""
    return score, f"{grade} | Score: {score}/100 | {regime_tag}{sig_tag}"


def calculate_exit_score(rsi, bearish_candle_info, vol_confirmed, mtf_info, adx, is_bear_div, trend_info, onchain_info, price, ema50, ema200, near_overbought_target: bool, df: pd.DataFrame | None = None, div_info: dict | None = None) -> tuple[int, str, list]:
    """
    [Advanced] คำนวณคะแนนเตือน 'ควรพิจารณาแบ่งขาย/ปิดสถานะ Long' (0-100)

    ปรับปรุงจากเดิม:
    1. RSI tiers ละเอียดขึ้น — RSI 78+ ได้คะแนนสูงทันที ไม่ต้องพึ่ง factor อื่น
    2. RSI Velocity (ความเร็ว RSI) — RSI พุ่งขึ้นเร็วเกินไป = exhaustion สูง
    3. Partial sell recommendation — บอก % ที่ควรขายตามระดับความรุนแรง
    4. RSI Extended Zone (>80, >85) — alert ระดับสูงสุด แจ้งทันทีแม้ไม่มี factor อื่น
    5. Price vs Bollinger-like extension — ราคาขยายออกจาก EMA200 มากเกินไป
    """
    reasons = []
    score = 0

    # ------------------------------------------------------------------
    # 1. RSI Tiered Scoring (ปรับใหม่ — RSI สูงควรได้คะแนนสูงทันที)
    # ------------------------------------------------------------------
    if rsi >= 85:
        score += 50
        reasons.append(f"🔴 RSI Extreme Overbought ({rsi:.1f}) — แรงซื้อหมดแล้ว แบ่งขายได้เลย")
    elif rsi >= 80:
        score += 40
        reasons.append(f"🔴 RSI Overbought รุนแรง ({rsi:.1f}) — ควรแบ่งขาย 30-50%")
    elif rsi >= RSI_OVERBOUGHT:  # 70
        score += 28
        reasons.append(f"🟠 RSI Overbought ({rsi:.1f}) — พิจารณาแบ่งขาย 20-30%")
    elif rsi >= 65:
        score += 15
        reasons.append(f"🟡 RSI สูง ({rsi:.1f}) — เฝ้าระวัง")
    elif rsi >= 60:
        score += 8
        reasons.append(f"RSI เริ่มสูง ({rsi:.1f})")

    # ------------------------------------------------------------------
    # 2. RSI Velocity — [Fix L] ตรวจทั้งขาขึ้น (exhaustion) และขาลง (momentum shift)
    # ------------------------------------------------------------------
    if df is not None and "RSI" in df.columns and len(df) >= 6:
        try:
            rsi_series = df["RSI"].iloc[-6:]
            rsi_now    = float(rsi_series.iloc[-1])
            rsi_3b_ago = float(rsi_series.iloc[-4])
            rsi_5b_ago = float(rsi_series.iloc[-6])
            rsi_peak   = float(rsi_series.max())
            rsi_peak_idx = rsi_series.argmax()

            # Fix L-a: RSI พุ่งขึ้นเร็ว = exhaustion (upward velocity)
            rsi_rise_3 = rsi_now - rsi_3b_ago
            rsi_rise_5 = rsi_now - rsi_5b_ago
            if rsi_rise_3 >= 15:
                score += 15
                reasons.append(f"⚡ RSI พุ่งขึ้น {rsi_rise_3:.1f} pts ใน 3 แท่ง (Exhaustion)")
            elif rsi_rise_5 >= 20:
                score += 10
                reasons.append(f"⚡ RSI พุ่งขึ้น {rsi_rise_5:.1f} pts ใน 5 แท่ง")

            # Fix L-b: RSI หักหัวลงจากยอด (downward velocity from peak) = momentum shift
            # เงื่อนไข: RSI เคยสูง (peak >= 65) และตอนนี้กำลังลงจากยอดนั้น
            if rsi_peak >= 65 and rsi_peak_idx < len(rsi_series) - 1:
                rsi_drop_from_peak = rsi_peak - rsi_now
                if rsi_drop_from_peak >= 8:
                    score += 12
                    reasons.append(
                        f"📉 RSI หักหัวลงจากยอด {rsi_peak:.1f} → {rsi_now:.1f} "
                        f"(ลด {rsi_drop_from_peak:.1f} pts) — Momentum Shift"
                    )
                elif rsi_drop_from_peak >= 5:
                    score += 6
                    reasons.append(f"📉 RSI เริ่มหักหัวจากยอด ({rsi_peak:.1f}→{rsi_now:.1f})")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3. Bearish reversal candle
    # ------------------------------------------------------------------
    bstr = bearish_candle_info.get("bearish_strength", 0)
    if bstr > 0:
        score += min(bstr * 8, 20)
        reasons.append(bearish_candle_info["bearish_label"].replace("🕯️ ", "").replace("<b>", "").replace("</b>", ""))

    # ------------------------------------------------------------------
    # 4. Bearish Divergence — [Fix K] use strength from div_info if available
    # ------------------------------------------------------------------
    if div_info and div_info.get("regular_bearish"):
        bear_str = div_info.get("regular_bearish_strength", 1)
        bear_pts = 12 + bear_str * 5   # str=1→17, str=2→22, str=3→27
        score += min(bear_pts, 28)
        reasons.append(
            f"RSI Regular Bearish Divergence (strength={bear_str}) — "
            f"ราคา HH แต่ RSI LH (โมเมนตัมอ่อนแรง)"
        )
    if div_info and div_info.get("hidden_bearish"):
        hb_str = div_info.get("hidden_bearish_strength", 1)
        score += 8 + hb_str * 3        # str=1→11, str=2→14
        reasons.append(f"RSI Hidden Bearish Divergence (strength={hb_str}) — เทรนด์ลงต่อ")
    elif is_bear_div and not (div_info and div_info.get("any_bearish")):
        # Fallback: old bool-only path
        score += 20
        reasons.append("RSI Bearish Divergence (โมเมนตัมอ่อนแรง)")

    # ------------------------------------------------------------------
    # 5. MTF Overbought Alignment (4H + 1D + 1W)
    # ------------------------------------------------------------------
    if mtf_info.get("aligned_overbought"):
        score += 18
        reasons.append(f"MTF Overbought (4H:{mtf_info.get('rsi_4h','?')}/1D:{mtf_info.get('rsi_1d','?')}/1W:{mtf_info.get('rsi_1w','?')})")
    elif mtf_info.get("rsi_4h", 0) and mtf_info.get("rsi_1d", 0):
        if (mtf_info["rsi_4h"] or 0) >= 70 and (mtf_info["rsi_1d"] or 0) >= 65:
            score += 10
            reasons.append(f"MTF RSI สูง (4H:{mtf_info['rsi_4h']}/1D:{mtf_info['rsi_1d']})")

    # ------------------------------------------------------------------
    # 6. Volume Exhaustion — mutual exclusion to avoid double-counting
    # ------------------------------------------------------------------
    vol_climax_detected = False
    # Volume Climax: spike สูงมากแล้วราคาไม่ไปต่อ = distribution (ตรวจก่อน)
    if df is not None and "volumeto" in df.columns and "VOL_MA20" in df.columns:
        try:
            vol_now = df["volumeto"].iloc[-1]
            vol_ma  = df["VOL_MA20"].iloc[-1]
            body    = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
            rng     = df["high"].iloc[-1] - df["low"].iloc[-1]
            if not pd.isna(vol_ma) and vol_ma > 0 and vol_now > vol_ma * 2.5:
                if rng > 0 and body / rng < 0.4:
                    # Climax = ใหญ่กว่า "ไม่ยืนยัน" ธรรมดา → ใช้แค่ climax score
                    score += 15
                    reasons.append("Volume Climax + Small Body (Distribution สัญญาณ — แรงซื้อหมด)")
                    vol_climax_detected = True
        except Exception:
            pass

    # ถ้าไม่ใช่ climax แต่ volume ไม่ยืนยัน → ใช้ low-volume penalty แทน
    # (mutual exclusion: ไม่บวกทั้งสอง)
    if not vol_climax_detected and not vol_confirmed:
        score += 8
        reasons.append("Volume ไม่ยืนยัน (อาจหมดแรงซื้อ)")

    # ------------------------------------------------------------------
    # 7. ADX Weakening
    # ------------------------------------------------------------------
    if adx < 15:
        score += 7
        reasons.append("ADX ต่ำ (เทรนด์อ่อนแรง)")
    elif df is not None and "ADX" in df.columns and len(df) >= 5:
        # ADX กำลังลดลง = trend losing steam
        try:
            adx_now  = float(df["ADX"].iloc[-1])
            adx_prev = float(df["ADX"].iloc[-4])
            if adx_now < adx_prev - 5 and adx_now < 25:
                score += 8
                reasons.append(f"ADX ลดลง ({adx_prev:.1f}→{adx_now:.1f}) — เทรนด์กำลังอ่อนแรง")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 8. Trend Structure Reversal
    # ------------------------------------------------------------------
    ts = trend_info.get("trend_strength", "sideways")
    if ts in ("strong_down", "moderate_down"):
        score += 20
        reasons.append(f"แนวโน้มกลับเป็นขาลง ({trend_info.get('trend_label','')})")
    elif ts == "sideways":
        score += 5
        reasons.append("แนวโน้มเริ่ม sideway")

    # ------------------------------------------------------------------
    # 9. Price vs EMA levels
    # ------------------------------------------------------------------
    if price < ema50 and price > ema200:
        score += 15
        reasons.append("ราคาหลุด EMA50 (โมเมนตัมอ่อนลง)")
    elif price < ema200:
        score += 28
        reasons.append("ราคาหลุด EMA200 (เปลี่ยนแนวโน้มหลัก)")

    # Price Extended too far from EMA200 (parabolic move = high reversal risk)
    if ema200 > 0 and price > 0:
        ext_pct = ((price - ema200) / ema200) * 100
        if ext_pct >= 40:
            score += 15
            reasons.append(f"ราคาห่างจาก EMA200 มาก ({ext_pct:.1f}%) — เสี่ยงกลับตัว Parabolic")
        elif ext_pct >= 25:
            score += 8
            reasons.append(f"ราคาห่างจาก EMA200 ({ext_pct:.1f}%)")

    # ------------------------------------------------------------------
    # 10. On-Chain Deterioration
    # ------------------------------------------------------------------
    if onchain_info.get("has_data") and onchain_info.get("active_addresses_trend", 0) <= -10:
        score += 10
        reasons.append("On-Chain หดตัวหนัก (คนเริ่มออกจากเครือข่าย)")

    # ------------------------------------------------------------------
    # 11. Near Estimated RSI-70 Target
    # ------------------------------------------------------------------
    if near_overbought_target:
        score += 8
        reasons.append("ราคาเข้าใกล้เป้าประมาณ RSI-70")

    # ------------------------------------------------------------------
    # 12. Partial Sell Recommendation (based on RSI level)
    # ------------------------------------------------------------------
    score = max(0, min(100, score))

    if rsi >= 85:
        sell_rec = "🔴 แนะนำขาย 50-70% ทันที"
    elif rsi >= 80:
        sell_rec = "🟠 แนะนำขาย 30-50%"
    elif rsi >= RSI_OVERBOUGHT:
        sell_rec = "🟡 แนะนำขาย 20-30% / เลื่อน SL ขึ้น"
    elif score >= 70:
        sell_rec = "🟠 แนะนำขาย 20-30%"
    elif score >= MINIMUM_EXIT_SCORE:
        sell_rec = "🟡 แนะนำเลื่อน SL / ขาย 10-20%"
    else:
        sell_rec = ""

    if sell_rec:
        reasons.append(sell_rec)

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------
    if rsi >= 80 or score >= 70:
        label = f"🚨 <b>สัญญาณเตือนแรง — แบ่งขายได้เลย</b> | Exit Score: {score}/100"
    elif score >= MINIMUM_EXIT_SCORE:
        label = f"⚠️ ควรพิจารณาลดสถานะ/เลื่อน SL | Exit Score: {score}/100"
    else:
        label = f"🟢 ยังถือต่อได้ | Exit Score: {score}/100"

    return score, label, reasons

def analyze_weekly_context(df_1d: pd.DataFrame) -> dict:
    """
    [Advanced] Weekly Context Analysis พร้อม:
    - Fibonacci Retracement ครบ 7 levels (23.6%, 38.2%, 50%, 61.8%, 78.6%, 88.6%)
    - Fibonacci Extensions (127.2%, 161.8%) สำหรับ TP targets
    - Weekly Structure: Higher High/Lower Low detection
    - Wyckoff Phase estimation (Accumulation / Markup / Distribution / Markdown)
    - Advanced Divergence ผ่าน weekly RSI pivots
    """
    result = {
        "rsi_weekly": None,
        "weekly_bullish_div": False,
        "weekly_status_label": "↔️ ไม่พบข้อมูล 1W",
        # Fibonacci Retracements
        "fibo_236": None, "fibo_382": None, "fibo_500": None,
        "fibo_618": None, "fibo_786": None, "fibo_886": None,
        # Fibonacci Extensions (จาก swing low)
        "fibo_ext_1272": None, "fibo_ext_1618": None,
        "liquidity_pool": None,
        "psycho_support": None,
        # Structure
        "weekly_structure": "unknown",  # "uptrend" | "downtrend" | "ranging"
        "higher_high": False,
        "lower_low": False,
        # Wyckoff
        "wyckoff_phase": "unknown",  # "accumulation" | "markup" | "distribution" | "markdown" | "unknown"
        "wyckoff_label": "",
        # Near which Fibo level?
        "nearest_fibo_level": None,
        "nearest_fibo_pct": None,
    }
    if df_1d is None or len(df_1d) < 35:
        return result
    try:
        df_w = df_1d.resample("W").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(df_w) < 15:
            return result

        window = min(52, len(df_w))
        w_slice = df_w.iloc[-window:]
        w_max = w_slice["high"].max()
        w_min = w_slice["low"].min()
        rng   = w_max - w_min

        if rng <= 0:
            return result

        # --- Fibonacci Retracements ---
        fibs_ret = {
            "fibo_236": w_max - 0.236 * rng,
            "fibo_382": w_max - 0.382 * rng,
            "fibo_500": w_max - 0.500 * rng,
            "fibo_618": w_max - 0.618 * rng,
            "fibo_786": w_max - 0.786 * rng,
            "fibo_886": w_max - 0.886 * rng,
        }
        result.update(fibs_ret)
        result["liquidity_pool"] = w_min

        # --- Fibonacci Extensions (from w_min upward) ---
        result["fibo_ext_1272"] = w_min + 1.272 * rng
        result["fibo_ext_1618"] = w_min + 1.618 * rng

        # --- Nearest Fibo to current price ---
        curr_price = df_w["close"].iloc[-1]
        all_fibs = {k: v for k, v in fibs_ret.items()}
        all_fibs["fibo_ext_1272"] = result["fibo_ext_1272"]
        all_fibs["fibo_ext_1618"] = result["fibo_ext_1618"]
        nearest_key = min(all_fibs, key=lambda k: abs(all_fibs[k] - curr_price))
        nearest_val = all_fibs[nearest_key]
        result["nearest_fibo_level"] = nearest_key
        result["nearest_fibo_pct"] = round(abs(curr_price - nearest_val) / curr_price * 100, 2)

        # --- Psychological support ---
        if curr_price > 0:
            mag = 10 ** math.floor(math.log10(curr_price))
            unit = mag if curr_price >= mag * 2 else mag / 2
            result["psycho_support"] = math.floor(curr_price / unit) * unit

        # --- Weekly RSI ---
        rs = (df_w["close"].diff().clip(lower=0).ewm(com=13, adjust=False).mean()
              / (-df_w["close"].diff().clip(upper=0)).ewm(com=13, adjust=False).mean().replace(0, np.nan))
        df_w["RSI"] = (100 - 100 / (1 + rs)).fillna(50)
        rsi_w = round(float(df_w["RSI"].iloc[-1]), 2)
        result["rsi_weekly"] = rsi_w

        # --- Weekly Structure: Higher High / Lower Low ---
        if len(df_w) >= 6:
            hh_now = df_w["high"].iloc[-3:].max()
            hh_prev = df_w["high"].iloc[-6:-3].max()
            ll_now = df_w["low"].iloc[-3:].min()
            ll_prev = df_w["low"].iloc[-6:-3].min()

            hh = hh_now > hh_prev
            ll = ll_now < ll_prev
            result["higher_high"] = hh
            result["lower_low"] = ll

            if hh and not ll:
                result["weekly_structure"] = "uptrend"
            elif ll and not hh:
                result["weekly_structure"] = "downtrend"
            elif hh and ll:
                result["weekly_structure"] = "expanding"
            else:
                result["weekly_structure"] = "ranging"

        # --- Wyckoff Phase Estimation ---
        close_w   = df_w["close"]
        vol_w     = df_w.get("volumeto", pd.Series(dtype=float)) if "volumeto" in df_w.columns else None
        rsi_slope = df_w["RSI"].diff().iloc[-4:].mean() if len(df_w) >= 4 else 0

        price_pos = (curr_price - w_min) / rng  # 0=at bottom, 1=at top

        if price_pos < 0.3 and rsi_w < 40:
            result["wyckoff_phase"] = "accumulation"
            result["wyckoff_label"] = "🏗️ Wyckoff: Accumulation (สะสมที่ฐาน)"
        elif price_pos >= 0.3 and rsi_slope > 0 and rsi_w < 70:
            result["wyckoff_phase"] = "markup"
            result["wyckoff_label"] = "📈 Wyckoff: Markup (ขาขึ้น)"
        elif price_pos > 0.7 and rsi_w >= 60:
            result["wyckoff_phase"] = "distribution"
            result["wyckoff_label"] = "🏛️ Wyckoff: Distribution (แจกจ่ายที่ยอด)"
        elif price_pos >= 0.3 and rsi_slope < 0 and rsi_w > 40:
            result["wyckoff_phase"] = "markdown"
            result["wyckoff_label"] = "📉 Wyckoff: Markdown (ขาลง)"
        else:
            result["wyckoff_label"] = "↔️ Wyckoff: ไม่ชัดเจน"

        # --- Weekly Bullish Divergence ---
        prev = df_w.iloc[-16:-3]
        if (len(prev) > 0
                and prev["RSI"].iloc[prev["low"].argmin()] <= 45
                and df_w["low"].iloc[-1] < prev["low"].min()
                and df_w["RSI"].iloc[-1] > prev["RSI"].iloc[prev["low"].argmin()]):
            result["weekly_bullish_div"] = True

        # --- Status label ---
        if result["weekly_bullish_div"]:
            result["weekly_status_label"] = f"👑 <b>Weekly Bullish Div!</b> (RSI: {rsi_w}) | {result['wyckoff_label']}"
        elif rsi_w <= RSI_OVERSOLD:
            result["weekly_status_label"] = f"🔥 <b>1W Oversold ({rsi_w})</b> | {result['wyckoff_label']}"
        elif rsi_w >= RSI_OVERBOUGHT:
            result["weekly_status_label"] = f"🔴 1W Overbought ({rsi_w}) | {result['wyckoff_label']}"
        else:
            result["weekly_status_label"] = f"↔️ 1W RSI: {rsi_w} | {result['wyckoff_label']}"

    except Exception as e:
        logger.debug(f"analyze_weekly_context error: {e}")
    return result

def analyze_trend_continuity(df: pd.DataFrame) -> dict:
    """
    [Advanced] Trend Continuity Analysis พร้อม:
    - Higher High / Higher Low structure detection (เทรนด์จริงๆ ไม่ใช่แค่ EMA slope)
    - Trend Consistency Score (ราคาชิดกับ EMA50 แค่ไหน)
    - Impulsive vs Corrective move detection
    - Efficiency Ratio (ราคาเดินทางตรงแค่ไหน เทียบกับ noise ทั้งหมด)
    """
    result = {
        "trend_strength": "sideways",
        "trend_label": "↔️ ไม่ชัดเจน",
        "higher_high": False,
        "higher_low": False,
        "lower_low": False,
        "lower_high": False,
        "efficiency_ratio": 0.0,
        "is_impulsive": False,
        "trend_consistency": 0.0,
    }
    if len(df) < 10:
        return result

    # --- EMA Slope ---
    ema50_now  = df["EMA_50"].iloc[-1]
    ema50_prev = df["EMA_50"].iloc[-6]
    ema200_now  = df["EMA_200"].iloc[-1]
    ema200_prev = df["EMA_200"].iloc[-6]
    slope50  = ((ema50_now - ema50_prev) / ema50_prev) * 100 if ema50_prev != 0 else 0
    slope200 = ((ema200_now - ema200_prev) / ema200_prev) * 100 if ema200_prev != 0 else 0

    # --- Price Streak ---
    diffs = df["close"].iloc[-20:].diff().iloc[1:].values[::-1]
    up_streak = next((i for i, v in enumerate(diffs) if v <= 0), len(diffs))
    dn_streak = next((i for i, v in enumerate(diffs) if v >= 0), len(diffs))

    # --- Higher High / Higher Low / Lower Low / Lower High Structure ---
    recent = df.iloc[-20:]
    pivot_h = _find_pivot_highs(recent["high"], left=2, right=2)
    pivot_l = _find_pivot_lows(recent["low"],   left=2, right=2)

    if len(pivot_h) >= 2:
        h1, h2 = recent["high"].iloc[pivot_h[-2]], recent["high"].iloc[pivot_h[-1]]
        result["higher_high"] = h2 > h1
        result["lower_high"]  = h2 < h1
    if len(pivot_l) >= 2:
        l1, l2 = recent["low"].iloc[pivot_l[-2]], recent["low"].iloc[pivot_l[-1]]
        result["higher_low"] = l2 > l1
        result["lower_low"]  = l2 < l1

    hh = result["higher_high"]
    hl = result["higher_low"]
    ll = result["lower_low"]
    lh = result["lower_high"]

    # --- Efficiency Ratio (Perry Kaufman) ---
    # ER = |net price change| / sum(|each bar change|)
    # 1.0 = perfectly directional, 0.0 = pure noise
    lookback_er = min(14, len(df) - 1)
    net_change = abs(df["close"].iloc[-1] - df["close"].iloc[-lookback_er])
    path_length = df["close"].iloc[-lookback_er:].diff().abs().sum()
    er = float(net_change / path_length) if path_length > 0 else 0.0
    result["efficiency_ratio"] = round(er, 3)
    result["is_impulsive"] = er >= 0.5

    # --- Trend Consistency: % of bars in last 20 that agree with the trend direction ---
    if slope50 > 0:
        consistent_bars = sum(1 for v in df["close"].iloc[-20:].diff().iloc[1:] if v > 0)
    else:
        consistent_bars = sum(1 for v in df["close"].iloc[-20:].diff().iloc[1:] if v < 0)
    result["trend_consistency"] = round(consistent_bars / 19, 2)

    # --- Classification ---
    struct_up   = hh and hl
    struct_down = ll and lh

    if slope50 > 0 and slope200 > 0 and up_streak >= 3 and struct_up:
        result.update({"trend_strength": "strong_up",
                        "trend_label": f"🚀 ขาขึ้นแข็งแกร่ง | HH+HL | ER:{er:.2f} | {up_streak} แท่ง"})
    elif slope50 > 0 and (up_streak >= 2 or struct_up):
        result.update({"trend_strength": "moderate_up",
                        "trend_label": f"📈 ขาขึ้นปานกลาง | ER:{er:.2f} | {up_streak} แท่ง"})
    elif slope50 <= 0 and slope200 <= 0 and dn_streak >= 3 and struct_down:
        result.update({"trend_strength": "strong_down",
                        "trend_label": f"🔻 ขาลงแข็งแกร่ง | LL+LH | ER:{er:.2f} | {dn_streak} แท่ง"})
    elif slope50 <= 0 and (dn_streak >= 2 or struct_down):
        result.update({"trend_strength": "moderate_down",
                        "trend_label": f"📉 ขาลงปานกลาง | ER:{er:.2f} | {dn_streak} แท่ง"})
    else:
        result.update({"trend_strength": "sideways",
                        "trend_label": f"↔️ Sideways | ER:{er:.2f} | Consistency:{result['trend_consistency']:.0%}"})
    return result

# ==========================================
# [Runner Extension] ADX Rising Check
# ==========================================
def is_adx_rising(df: pd.DataFrame, lookback: int = RUNNER_EXTEND_ADX_RISING_LOOKBACK) -> bool:
    """ตรวจว่า ADX กำลังเพิ่มขึ้นต่อเนื่องในช่วง lookback แท่งล่าสุดหรือไม่ (เทรนด์แข็งแกร่งขึ้น)"""
    if "ADX" not in df.columns or len(df) < lookback + 1:
        return False
    recent = df["ADX"].iloc[-(lookback + 1):]
    diffs = recent.diff().iloc[1:]
    return bool((diffs > 0).all())

# ==========================================
# [Entry Filter] 1D Trend Alignment (anti bag-holding)
# ==========================================
def check_1d_trend_alignment(df_1d: pd.DataFrame) -> dict:
    """
    ตรวจ daily trend: ราคาปิดล่าสุดต้องอยู่เหนือ EMA50 (1D) ด้วย ไม่ใช่แค่ 4H > EMA200
    ป้องกัน "dip ที่จริงๆ คือจุดเริ่มขาลง" บน timeframe ใหญ่
    คืน dict: {"has_data": bool, "aligned": bool, "ema50_1d": float|None, "label": str}
    """
    result = {"has_data": False, "aligned": True, "ema50_1d": None, "label": ""}
    if not REQUIRE_1D_EMA50_ALIGNMENT:
        return result
    if df_1d is None or len(df_1d) < EMA_SHORT_1D + 5:
        # Not enough daily data to judge — don't block (fail open)
        return result

    try:
        ema50_1d = df_1d["close"].ewm(span=EMA_SHORT_1D, adjust=False).mean().iloc[-1]
        price_1d = df_1d["close"].iloc[-1]
        result.update({"has_data": True, "ema50_1d": round(float(ema50_1d), 6)})
        if price_1d >= ema50_1d:
            result["aligned"] = True
            result["label"] = f"🟢 1D ราคา > EMA50(1D) — เทรนด์รายวันสอดคล้อง"
        else:
            result["aligned"] = False
            result["label"] = f"🔴 1D ราคา < EMA50(1D) — เทรนด์รายวันไม่สอดคล้อง (อาจเป็นกับดัก)"
    except Exception:
        pass

    return result

def analyze_rsi_bounce(df: pd.DataFrame) -> dict:
    """
    [Advanced] ตรวจ RSI Bounce Quality จากหลาย dimension:
    1. V-shape recovery: RSI ลงถึง oversold แล้วดีดแบบ V (ไม่แฉลบข้างนาน)
    2. Momentum acceleration: RSI ดีดเร็วขึ้นทุกแท่ง (แรงเพิ่ม)
    3. Volume confirmation: volume สูงขณะดีด (ยืนยันว่ามีแรงซื้อจริง)
    4. Depth of oversold: ยิ่งลึก oversold มาก ยิ่ง bounce มีน้ำหนัก
    5. Entry timing: บอก timing ที่เหมาะสม (early / confirmed / late)
    """
    result = {
        "quality": "none",
        "quality_label": "⬜ ไม่มีสัญญาณดีดกลับ",
        "entry_timing": "",
        "bounce_score": 0,
        "v_shape": False,
        "momentum_accel": False,
        "vol_on_bounce": False,
        "oversold_depth": 0.0,
    }
    if len(df) < 20:
        return result

    rsi_series = df["RSI"].iloc[-20:]
    rsi_curr   = rsi_series.iloc[-1]
    rsi_min    = rsi_series.min()
    min_idx    = rsi_series.argmin()  # position of the trough within the window

    if rsi_min > RSI_OVERSOLD:
        return result

    rsi_rise = rsi_curr - rsi_min
    bars_since_min = len(rsi_series) - 1 - min_idx
    result["oversold_depth"] = round(RSI_OVERSOLD - rsi_min, 1)

    score = 0

    # --- 1. Depth of oversold (ยิ่งลึก ยิ่งมีน้ำหนัก) ---
    if rsi_min <= 20:
        score += 3
    elif rsi_min <= 25:
        score += 2
    elif rsi_min <= RSI_OVERSOLD:
        score += 1

    # --- 2. RSI Rise magnitude ---
    if rsi_rise >= 8:
        score += 3
    elif rsi_rise >= 5:
        score += 2
    elif rsi_rise >= 3:
        score += 1

    # --- 3. V-shape: ดีดกลับภายใน 1-4 แท่งหลัง trough (ไม่ลาก sideways) ---
    if 1 <= bars_since_min <= 4 and rsi_rise >= 3:
        result["v_shape"] = True
        score += 2

    # --- 4. Momentum acceleration: RSI ดีดขึ้นเร็วขึ้นทุกแท่งใน 3 แท่งล่าสุด ---
    recent_rsi_diffs = rsi_series.diff().iloc[-4:].values[1:]  # 3 diffs
    if len(recent_rsi_diffs) == 3 and all(d > 0 for d in recent_rsi_diffs):
        if recent_rsi_diffs[2] > recent_rsi_diffs[1] > recent_rsi_diffs[0]:
            result["momentum_accel"] = True
            score += 2

    # --- 5. Volume confirmation during bounce ---
    vol_col = "volumeto"
    if vol_col in df.columns and "VOL_MA20" in df.columns:
        vol_now    = df[vol_col].iloc[-1]
        vol_ma     = df["VOL_MA20"].iloc[-1]
        vol_at_min = df[vol_col].iloc[-(bars_since_min + 1)] if bars_since_min < len(df) else 0
        if not pd.isna(vol_ma) and vol_ma > 0 and vol_now > vol_ma * 1.2:
            result["vol_on_bounce"] = True
            score += 2
        # Extra: volume spike AT the trough (capitulation candle) = strong reversal signal
        if not pd.isna(vol_ma) and vol_ma > 0 and vol_at_min > vol_ma * 1.5:
            score += 1

    # --- 6. RSI still below 50 (not overbought yet = room to run) ---
    if rsi_curr < 50:
        score += 1

    result["bounce_score"] = score

    # --- Entry timing ---
    if bars_since_min <= 1:
        result["entry_timing"] = "⚡ Early Entry (ดีดจาก trough ใหม่ๆ — risk สูง แต่ reward ดี)"
    elif bars_since_min <= 3:
        result["entry_timing"] = "✅ Confirmed Entry (ดีดยืนยันแล้ว — timing ดีที่สุด)"
    else:
        result["entry_timing"] = "⚠️ Late Entry (ดีดมาสักพักแล้ว — รอ pullback ก่อนอาจดีกว่า)"

    # --- Quality classification ---
    if score >= 9:
        result.update({"quality": "strong",
                        "quality_label": f"✅ <b>ดีดกลับแข็งแกร่งมาก</b> | RSI +{rsi_rise:.1f} | Score {score} | {result['entry_timing']}"})
    elif score >= 6:
        result.update({"quality": "strong",
                        "quality_label": f"✅ ดีดกลับแข็งแกร่ง | RSI +{rsi_rise:.1f} | Score {score} | {result['entry_timing']}"})
    elif score >= 3:
        result.update({"quality": "moderate",
                        "quality_label": f"🟡 ดีดกลับปานกลาง | RSI +{rsi_rise:.1f} | Score {score} | {result['entry_timing']}"})
    elif score >= 1:
        result.update({"quality": "weak",
                        "quality_label": f"🟠 ดีดกลับอ่อน | RSI +{rsi_rise:.1f} | Score {score}"})
    return result

def find_order_blocks(df: pd.DataFrame) -> dict:
    """
    ตรวจหา Order Block ทั้ง Bullish (Demand Zone) และ Bearish (Supply Zone)
    ตามแนวคิด Smart Money / ICT

    Bullish OB: แท่งแดง (bearish candle) ที่อยู่ก่อนการพุ่งขึ้นแรง
                ทำหน้าที่เป็น Demand Zone / แนวรับ
    Bearish OB: แท่งเขียว (bullish candle) ที่อยู่ก่อนการร่วงลงแรง
                ทำหน้าที่เป็น Supply Zone / แนวต้าน / กำแพงขาย

    Strength score (0-3) ของแต่ละ OB:
      +1 ถ้า body ของ OB candle ใหญ่กว่าค่าเฉลี่ย 2 เท่าขึ้นไป (institutional candle)
      +1 ถ้า volume ของ OB candle สูงกว่า MA20 volume (volume confirmation)
      +1 ถ้า OB ยังไม่ถูก "mitigated" (ราคาไม่เคยกลับมาแตะ OB zone อีก)
    """
    ob = {
        # Bullish OB (Demand Zone)
        "has_bullish_ob": False,
        "bullish_ob_price": None,        # low ของ OB candle (แนวรับล่าง)
        "bullish_ob_top": None,          # high ของ OB candle (แนวรับบน)
        "bullish_ob_strength": 0,        # 0-3
        "bullish_ob_label": "",

        # Bearish OB (Supply Zone)
        "has_bearish_ob": False,
        "bearish_ob_top": None,          # high ของ OB candle (แนวต้านบน)
        "bearish_ob_bottom": None,       # low ของ OB candle (แนวต้านล่าง)
        "bearish_ob_strength": 0,        # 0-3
        "bearish_ob_label": "",

        # Proximity alert (ราคาปัจจุบันเข้าใกล้ OB zone)
        "near_bearish_ob": False,        # ราคาอยู่ภายใน proximity ของ Bearish OB
        "near_bearish_ob_pct": None,     # ห่างจาก Bearish OB กี่ %
        "near_bullish_ob": False,
        "near_bullish_ob_pct": None,
    }

    if len(df) < 30:
        return ob

    avg_body = (df["close"] - df["open"]).abs().rolling(20).mean()
    vol_ma20 = df["volumeto"].rolling(20).mean()
    current_price = df["close"].iloc[-1]
    lookback = min(50, len(df) - 3)

    # -------------------------------------------------------
    # [Bullish OB] แท่งแดงก่อนการพุ่งขึ้นแรง (Demand Zone)
    # -------------------------------------------------------
    for i in range(2, lookback):
        candle = df.iloc[-i]
        # เงื่อนไข OB candle: เป็นแท่งแดง (bearish)
        if candle["close"] >= candle["open"]:
            continue

        # การพุ่งขึ้น: แท่งถัดไปต้องเป็น bullish แรง (body > 1.5x avg)
        next_c = df.iloc[-i + 1]
        avg_b = avg_body.iloc[-i + 1]
        move_up = (next_c["close"] - next_c["open"]) > (avg_b * 1.5 if not pd.isna(avg_b) else 0)
        if not move_up:
            continue

        # OB zone: low-high ของแท่งแดงนั้น
        ob_low, ob_high = candle["low"], candle["high"]

        # ตรวจว่า OB ยัง valid (ราคาอยู่เหนือ OB และไม่เคยกลับมา close ต่ำกว่า ob_low หลังจากนั้น)
        prices_after = df.iloc[-i + 1:]
        if (prices_after["close"] < ob_low).any():
            continue  # OB ถูก mitigated แล้ว — ข้าม

        # Strength scoring
        strength = 0
        candle_body = abs(candle["close"] - candle["open"])
        avg_b_ob = avg_body.iloc[-i]
        if not pd.isna(avg_b_ob) and avg_b_ob > 0 and candle_body > avg_b_ob * 2:
            strength += 1  # institutional-sized candle
        vol_ob = candle["volumeto"]
        vol_ma_ob = vol_ma20.iloc[-i]
        if vol_ob > 0 and not pd.isna(vol_ma_ob) and vol_ob > vol_ma_ob:
            strength += 1  # volume confirmation
        # ยังไม่ถูก mitigated (checked above) = +1
        strength += 1

        label_map = {3: "🔥 แข็งแกร่งมาก", 2: "🟢 ปานกลาง", 1: "🟡 อ่อน"}
        ob.update({
            "has_bullish_ob": True,
            "bullish_ob_price": ob_low,
            "bullish_ob_top": ob_high,
            "bullish_ob_strength": strength,
            "bullish_ob_label": label_map.get(strength, "⬜"),
        })

        # Proximity: ราคาปัจจุบันอยู่เหนือ ob_low ไม่เกิน 3%
        dist_pct = ((current_price - ob_high) / ob_high) * 100 if ob_high > 0 else None
        if dist_pct is not None and 0 <= dist_pct <= 3.0:
            ob["near_bullish_ob"] = True
            ob["near_bullish_ob_pct"] = round(dist_pct, 2)
        break  # ใช้ OB ล่าสุดที่ valid เพียงตัวเดียว

    # -------------------------------------------------------
    # [Bearish OB] แท่งเขียวก่อนการร่วงลงแรง (Supply Zone)
    # -------------------------------------------------------
    for i in range(2, lookback):
        candle = df.iloc[-i]
        # เงื่อนไข OB candle: เป็นแท่งเขียว (bullish)
        if candle["close"] <= candle["open"]:
            continue

        # การร่วงลง: แท่งถัดไปต้องเป็น bearish แรง (body > 1.5x avg)
        next_c = df.iloc[-i + 1]
        avg_b = avg_body.iloc[-i + 1]
        move_down = (next_c["open"] - next_c["close"]) > (avg_b * 1.5 if not pd.isna(avg_b) else 0)
        if not move_down:
            continue

        # OB zone: low-high ของแท่งเขียวนั้น
        ob_low, ob_high = candle["low"], candle["high"]

        # ตรวจว่า OB ยัง valid (ราคาอยู่ต่ำกว่า OB top และไม่เคยกลับมา close สูงกว่า ob_high หลังจากนั้น)
        prices_after = df.iloc[-i + 1:]
        if (prices_after["close"] > ob_high).any():
            continue  # Bearish OB ถูก mitigated (ราคาทะลุผ่านขึ้นไปแล้ว) — ข้าม

        # Strength scoring
        strength = 0
        candle_body = abs(candle["close"] - candle["open"])
        avg_b_ob = avg_body.iloc[-i]
        if not pd.isna(avg_b_ob) and avg_b_ob > 0 and candle_body > avg_b_ob * 2:
            strength += 1  # institutional-sized candle
        vol_ob = candle["volumeto"]
        vol_ma_ob = vol_ma20.iloc[-i]
        if vol_ob > 0 and not pd.isna(vol_ma_ob) and vol_ob > vol_ma_ob:
            strength += 1  # volume confirmation
        strength += 1  # ยังไม่ถูก mitigated

        label_map = {3: "🔥 แข็งแกร่งมาก", 2: "🟠 ปานกลาง", 1: "🟡 อ่อน"}
        ob.update({
            "has_bearish_ob": True,
            "bearish_ob_top": ob_high,
            "bearish_ob_bottom": ob_low,
            "bearish_ob_strength": strength,
            "bearish_ob_label": label_map.get(strength, "⬜"),
        })

        # Proximity: ราคาปัจจุบันอยู่ใต้ ob_high ไม่เกิน 5%
        dist_pct = ((ob_low - current_price) / current_price) * 100 if current_price > 0 else None
        if dist_pct is not None and 0 <= dist_pct <= 5.0:
            ob["near_bearish_ob"] = True
            ob["near_bearish_ob_pct"] = round(dist_pct, 2)
        break  # ใช้ OB ล่าสุดที่ valid เพียงตัวเดียว

    return ob

def get_bearish_ob_alert(ob_info: dict, price: float) -> dict:
    """
    สร้าง Supply Zone / Bearish OB alert สำหรับแสดงใน Telegram
    แบ่งเป็น 3 กรณี:
    1. ราคาเข้าใกล้ Supply Zone (near_bearish_ob) -> เตือนเป็นพิเศษ
    2. มี Bearish OB แต่ยังห่างอยู่ -> แสดงข้อมูลอ้างอิง
    3. ไม่มี Bearish OB -> N/A
    """
    result = {
        "has_alert": False,
        "alert_level": "none",  # "danger" | "caution" | "info"
        "supply_zone_label": "",
        "tp_ceiling_note": "",
    }

    if not ob_info.get("has_bearish_ob"):
        result["supply_zone_label"] = "⚪ ไม่พบ Supply Zone / Bearish OB"
        return result

    ob_top   = ob_info["bearish_ob_top"]
    ob_bot   = ob_info["bearish_ob_bottom"]
    strength = ob_info["bearish_ob_strength"]
    slabel   = ob_info["bearish_ob_label"]
    dist_pct = ((ob_bot - price) / price) * 100 if price > 0 else None

    result["has_alert"] = True

    if ob_info.get("near_bearish_ob"):
        # ราคาเข้าใกล้มาก (<= 5% ใต้ OB bottom)
        result["alert_level"] = "danger" if strength >= 2 else "caution"
        dist_str = f"{ob_info['near_bearish_ob_pct']:.1f}% ใต้ Supply Zone"
        result["supply_zone_label"] = (
            f"🚨 <b>ราคาใกล้ Supply Zone!</b>\n"
            f"   🔴 Bearish OB Zone: ${format_price(ob_bot)} – ${format_price(ob_top)}\n"
            f"   ห่างอีก {dist_str} | ความแข็งแกร่ง: {slabel}\n"
            f"   ⚠️ แนวต้านขนาดใหญ่ — ระมัดระวังแรงขาย/ลดสถานะบางส่วน"
        )
        result["tp_ceiling_note"] = (
            f"🏴 TP อาจถูก Supply Zone กดที่ ${format_price(ob_bot)} "
            f"(พิจารณาตั้ง TP1 ใต้กำแพง)"
        )
    else:
        # มี OB แต่ยังห่างอยู่
        dist_str = f"{dist_pct:.1f}%" if dist_pct is not None else "N/A"
        result["alert_level"] = "info"
        result["supply_zone_label"] = (
            f"🟠 Supply Zone อ้างอิง: ${format_price(ob_bot)} – ${format_price(ob_top)} "
            f"(ห่าง {dist_str}) | {slabel}"
        )

    return result

def get_bearish_ob_as_tp_ceiling(ob_info: dict, tp1: float, tp2: float) -> dict:
    """
    ถ้ามี Bearish OB ที่ valid และ strength >= 2 อยู่ระหว่าง TP1 กับ TP2:
    ให้ปรับ TP1 ลงมาไว้ใต้ Supply Zone เล็กน้อย (ก่อนถึงแนวต้าน)
    เพื่อป้องกันการ "ติดกำแพง" และเสีย gain กลับ

    คืน dict: {"adjusted": bool, "tp1": float, "note": str}
    """
    result = {"adjusted": False, "tp1": tp1, "note": ""}

    if not ob_info.get("has_bearish_ob"):
        return result
    if ob_info.get("bearish_ob_strength", 0) < 2:
        return result

    ob_bot = ob_info["bearish_ob_bottom"]
    if ob_bot is None:
        return result

    # ถ้า Supply Zone อยู่ระหว่าง TP1 และ TP2 -> ดึง TP1 มาอยู่ใต้ OB bottom 0.5%
    if tp1 < ob_bot < tp2:
        new_tp1 = ob_bot * 0.995  # ต่ำกว่า OB bottom 0.5% เพื่อจอง profit ก่อนโดน supply
        result.update({
            "adjusted": True,
            "tp1": new_tp1,
            "note": (
                f"⚙️ ปรับ TP1 ลงเป็น ${format_price(new_tp1)} "
                f"(ใต้ Supply Zone ${format_price(ob_bot)} 0.5%) "
                f"เพื่อหลีกเลี่ยงแนวต้าน"
            ),
        })

    return result

def find_fair_value_gaps(df: pd.DataFrame) -> dict:
    """
    [Advanced] หา Fair Value Gaps (FVG / Imbalance zones) ทั้งหมดที่ยังไม่ถูก fill
    แยก Bullish FVG (support) และ Bearish FVG (resistance)
    จัดอันดับตาม size + อายุ (FVG ขนาดใหญ่ + เพิ่งเกิด = น้ำหนักสูงสุด)

    Bullish FVG: low[i] > high[i-2] (gap up — support zone เมื่อราคาย้อนกลับมา)
    Bearish FVG: high[i] < low[i-2] (gap down — resistance zone เมื่อราคา rally)
    """
    result = {
        "has_fvg_support": False,
        "fvg_top": None, "fvg_bottom": None,
        "fvg_size_pct": None,
        "fvg_age_bars": None,
        # Bearish FVG (resistance)
        "has_fvg_resistance": False,
        "fvg_res_top": None, "fvg_res_bottom": None,
        "fvg_res_size_pct": None,
        # All unmitigated FVGs for context
        "all_bullish_fvgs": [],
        "all_bearish_fvgs": [],
    }
    if len(df) < 6:
        return result

    current_price = df["close"].iloc[-1]
    all_bull, all_bear = [], []

    for i in range(2, min(len(df) - 1, 80)):
        high_im2 = df["high"].iloc[-i - 2]
        low_im2  = df["low"].iloc[-i - 2]
        mid_body = df["close"].iloc[-i - 1] > df["open"].iloc[-i - 1]  # middle candle bullish?
        low_i    = df["low"].iloc[-i]
        high_i   = df["high"].iloc[-i]

        # --- Bullish FVG: low[i] > high[i-2] ---
        if low_i > high_im2 and mid_body:
            gap_size_pct = ((low_i - high_im2) / high_im2) * 100
            if gap_size_pct >= 0.15:
                # Check not mitigated: price never closed below fvg_bottom after the gap
                prices_after = df["close"].iloc[-i + 1:]
                if not (prices_after < high_im2).any() and current_price > high_im2:
                    all_bull.append({
                        "top": low_i, "bottom": high_im2,
                        "size_pct": round(gap_size_pct, 3),
                        "age_bars": i,
                        "score": gap_size_pct / max(i, 1),  # bigger + newer = higher score
                    })

        # --- Bearish FVG: high[i] < low[i-2] ---
        if high_i < low_im2 and not mid_body:
            gap_size_pct = ((low_im2 - high_i) / high_i) * 100
            if gap_size_pct >= 0.15:
                prices_after = df["close"].iloc[-i + 1:]
                if not (prices_after > low_im2).any() and current_price < low_im2:
                    all_bear.append({
                        "top": low_im2, "bottom": high_i,
                        "size_pct": round(gap_size_pct, 3),
                        "age_bars": i,
                        "score": gap_size_pct / max(i, 1),
                    })

    # Sort by score descending, best FVG first
    all_bull.sort(key=lambda x: -x["score"])
    all_bear.sort(key=lambda x: -x["score"])
    result["all_bullish_fvgs"] = all_bull
    result["all_bearish_fvgs"] = all_bear

    # Best Bullish FVG (support)
    if all_bull:
        best = all_bull[0]
        result.update({
            "has_fvg_support": True,
            "fvg_top": best["top"], "fvg_bottom": best["bottom"],
            "fvg_size_pct": best["size_pct"],
            "fvg_age_bars": best["age_bars"],
        })

    # Best Bearish FVG (resistance)
    if all_bear:
        best = all_bear[0]
        result.update({
            "has_fvg_resistance": True,
            "fvg_res_top": best["top"], "fvg_res_bottom": best["bottom"],
            "fvg_res_size_pct": best["size_pct"],
        })

    return result

def estimate_price_for_target_rsi(df: pd.DataFrame, target_rsi: float = 70.0, bars_ahead: int = 3) -> float | None:
    """
    [Advanced] ประมาณ Price Target สำหรับ RSI ที่ต้องการ
    - เดิม: ประมาณแค่ 1 แท่งข้างหน้า (ไม่แม่นยำ)
    - ใหม่: simulate หลายแท่งข้างหน้า (bars_ahead) ด้วย avg gain/loss ปัจจุบัน
            คืนค่า price ที่ RSI จะถึง target_rsi ภายใน bars_ahead แท่ง
            พร้อม confidence ว่าการประมาณน่าเชื่อถือแค่ไหน
    """
    if len(df) < 20 or target_rsi >= 100.0:
        return None
    try:
        delta    = df["close"].diff()
        avg_gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean().iloc[-1]
        avg_loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean().iloc[-1]
        curr_price = df["close"].iloc[-1]

        if avg_loss <= 0:
            return None

        # Simulate forward: assume avg_gain and avg_loss evolve as EMA
        sim_gain, sim_loss = avg_gain, avg_loss
        sim_price = curr_price

        for _ in range(bars_ahead):
            rs = sim_gain / max(sim_loss, 1e-10)
            rsi = 100 - 100 / (1 + rs)
            if rsi >= target_rsi:
                return round(sim_price, 8)
            # Each bar: price moves by (avg_gain - avg_loss)
            price_move = (sim_gain * RSI_PERIOD - sim_loss * (RSI_PERIOD - 1)) / RSI_PERIOD
            sim_price  = sim_price + max(price_move, 0)
            # Update EMAs with assumed all-gain bars
            sim_gain = (sim_gain * (RSI_PERIOD - 1) + price_move) / RSI_PERIOD
            sim_loss = (sim_loss * (RSI_PERIOD - 1)) / RSI_PERIOD

        return round(sim_price, 8)
    except Exception:
        return None

def check_time_stop(df: pd.DataFrame, lookback_bars: int = TIME_STOP_BARS_4H) -> dict:
    """
    [Advanced] Time Stop Analysis พร้อม Efficiency Ratio + Opportunity Cost

    เดิม: เช็คแค่ว่า price range แคบกว่า ATR×3
    ใหม่:
    1. Efficiency Ratio (Perry Kaufman): ยิ่งต่ำ = ราคายิ่งวน ไม่ไปไหน
    2. Volume trend during consolidation: volume ลดลงขณะ sideway = ตลาดหมดความสนใจ
    3. Opportunity cost estimate: กี่ ATR ที่ "เสีย" ไปโดยไม่ได้กำไร
    4. Staleness score (0-100): รวมทุก factor ให้ตัดสินใจง่ายขึ้น
    """
    result = {
        "is_stale": False,
        "time_stop_label": "",
        "efficiency_ratio": None,
        "vol_declining": False,
        "staleness_score": 0,
    }
    if len(df) < lookback_bars + 5 or "ATR" not in df.columns:
        return result

    recent  = df.iloc[-lookback_bars:]
    atr_now = df["ATR"].iloc[-1]
    if pd.isna(atr_now) or atr_now <= 0:
        return result

    # --- Efficiency Ratio ---
    net_change  = abs(recent["close"].iloc[-1] - recent["close"].iloc[0])
    path_length = recent["close"].diff().abs().sum()
    er = float(net_change / path_length) if path_length > 0 else 0.0
    result["efficiency_ratio"] = round(er, 3)

    # --- Price range vs ATR ---
    price_range  = recent["high"].max() - recent["low"].min()
    range_to_atr = price_range / atr_now

    # --- Volume declining during consolidation ---
    if "volumeto" in df.columns and "VOL_MA20" in df.columns:
        vol_recent = df["volumeto"].iloc[-lookback_bars:]
        vol_slope  = vol_recent.diff().mean()
        result["vol_declining"] = bool(vol_slope < 0 and vol_recent.mean() < df["VOL_MA20"].iloc[-1] * 0.8)

    # --- Staleness Score ---
    score = 0
    if er < 0.2:     score += 40   # very choppy
    elif er < 0.35:  score += 25
    if range_to_atr < 3: score += 30
    elif range_to_atr < 5: score += 15
    if result["vol_declining"]: score += 20
    result["staleness_score"] = min(score, 100)

    days_approx = lookback_bars * 4 / 24
    opportunity_cost_atr = round(range_to_atr, 1)

    if score >= 50:
        result["is_stale"] = True
        result["time_stop_label"] = (
            f"⏳ Time-Stop: ราคาวน sideway ~{days_approx:.0f} วัน "
            f"| ER:{er:.2f} | Range={opportunity_cost_atr}×ATR "
            f"| Staleness:{score}/100 — พิจารณาปิดเพื่อลด Opportunity Cost"
        )
    elif score >= 30:
        result["time_stop_label"] = (
            f"🕐 เริ่มแกว่ง sideway ({days_approx:.0f} วัน) | ER:{er:.2f} | Score:{score}/100"
        )
    return result

def _find_pivot_lows(series: pd.Series, left: int = 3, right: int = 3) -> list[int]:
    """หา Swing Low pivots จริงๆ: แต่ละ pivot ต้องต่ำกว่าแท่งข้างๆ left/right แท่ง"""
    pivots = []
    for i in range(left, len(series) - right):
        if all(series.iloc[i] <= series.iloc[i - j] for j in range(1, left + 1)) and \
           all(series.iloc[i] <= series.iloc[i + j] for j in range(1, right + 1)):
            pivots.append(i)
    return pivots

def _find_pivot_highs(series: pd.Series, left: int = 3, right: int = 3) -> list[int]:
    """หา Swing High pivots จริงๆ: แต่ละ pivot ต้องสูงกว่าแท่งข้างๆ left/right แท่ง"""
    pivots = []
    for i in range(left, len(series) - right):
        if all(series.iloc[i] >= series.iloc[i - j] for j in range(1, left + 1)) and \
           all(series.iloc[i] >= series.iloc[i + j] for j in range(1, right + 1)):
            pivots.append(i)
    return pivots

def check_bullish_divergence(df: pd.DataFrame) -> bool:
    """Wrapper ที่ยังคง return bool สำหรับ backward compatibility — ใช้ check_divergence_advanced แทนสำหรับรายละเอียด"""
    result = check_divergence_advanced(df)
    return result["regular_bullish"] or result["hidden_bullish"]

def check_bearish_divergence(df: pd.DataFrame) -> bool:
    """Wrapper ที่ยังคง return bool สำหรับ backward compatibility"""
    result = check_divergence_advanced(df)
    return result["regular_bearish"] or result["hidden_bearish"]

def check_divergence_advanced(df: pd.DataFrame) -> dict:
    """
    [Advanced] ตรวจ Divergence ทั้ง 4 ประเภทพร้อม Strength Score:

    Regular Bullish:  ราคา Lower Low, RSI Higher Low  -> กลับตัวขึ้น (weakening downtrend)
    Regular Bearish:  ราคา Higher High, RSI Lower High -> กลับตัวลง (weakening uptrend)
    Hidden Bullish:   ราคา Higher Low, RSI Lower Low   -> เทรนด์ขึ้นต่อ (pullback ใน uptrend)
    Hidden Bearish:   ราคา Lower High, RSI Higher High -> เทรนด์ลงต่อ (rally ใน downtrend)

    Strength factors:
    - ระยะห่าง RSI ระหว่าง 2 pivots (ยิ่งห่างยิ่งชัด)
    - จำนวน pivot ที่ confirm (2+ pivot pairs = stronger)
    - RSI level ของ pivot (oversold/overbought level เพิ่มน้ำหนัก)
    """
    result = {
        "regular_bullish": False,  "regular_bullish_strength": 0,
        "regular_bearish": False,  "regular_bearish_strength": 0,
        "hidden_bullish":  False,  "hidden_bullish_strength":  0,
        "hidden_bearish":  False,  "hidden_bearish_strength":  0,
        "divergence_label": "",
        "any_bullish": False,
        "any_bearish": False,
    }
    if len(df) < 20:
        return result

    lookback = df.iloc[-50:].copy()
    price_lows  = lookback["low"]
    price_highs = lookback["high"]
    rsi_series  = lookback["RSI"] if "RSI" in lookback.columns else None
    if rsi_series is None:
        return result

    pivot_lows  = _find_pivot_lows(price_lows,  left=3, right=2)
    pivot_highs = _find_pivot_highs(price_highs, left=3, right=2)

    labels = []

    # -------------------------------------------------------
    # BULLISH DIVERGENCES (ใช้ pivot lows)
    # -------------------------------------------------------
    if len(pivot_lows) >= 2:
        # เปรียบเทียบ 2 pivot lows ล่าสุด
        p1_idx, p2_idx = pivot_lows[-2], pivot_lows[-1]
        p1_price, p2_price = price_lows.iloc[p1_idx], price_lows.iloc[p2_idx]
        p1_rsi,   p2_rsi   = rsi_series.iloc[p1_idx],  rsi_series.iloc[p2_idx]

        rsi_diff = abs(p2_rsi - p1_rsi)

        # Regular Bullish: price LL, RSI HL
        if p2_price < p1_price and p2_rsi > p1_rsi:
            strength = 1
            if rsi_diff >= 5:  strength += 1
            if rsi_diff >= 10: strength += 1
            if p1_rsi <= RSI_OVERSOLD or p2_rsi <= RSI_OVERSOLD: strength += 1
            result["regular_bullish"] = True
            result["regular_bullish_strength"] = strength
            labels.append(f"📈 Regular Bullish Div (str={strength})")

        # Hidden Bullish: price HL, RSI LL
        if p2_price > p1_price and p2_rsi < p1_rsi:
            strength = 1
            if rsi_diff >= 5:  strength += 1
            if p1_rsi <= 45:   strength += 1
            result["hidden_bullish"] = True
            result["hidden_bullish_strength"] = strength
            labels.append(f"📈 Hidden Bullish Div (str={strength})")

    # -------------------------------------------------------
    # BEARISH DIVERGENCES (ใช้ pivot highs)
    # -------------------------------------------------------
    if len(pivot_highs) >= 2:
        p1_idx, p2_idx = pivot_highs[-2], pivot_highs[-1]
        p1_price, p2_price = price_highs.iloc[p1_idx], price_highs.iloc[p2_idx]
        p1_rsi,   p2_rsi   = rsi_series.iloc[p1_idx],  rsi_series.iloc[p2_idx]

        rsi_diff = abs(p2_rsi - p1_rsi)

        # Regular Bearish: price HH, RSI LH
        if p2_price > p1_price and p2_rsi < p1_rsi:
            strength = 1
            if rsi_diff >= 5:  strength += 1
            if rsi_diff >= 10: strength += 1
            if p1_rsi >= RSI_OVERBOUGHT or p2_rsi >= RSI_OVERBOUGHT: strength += 1
            result["regular_bearish"] = True
            result["regular_bearish_strength"] = strength
            labels.append(f"📉 Regular Bearish Div (str={strength})")

        # Hidden Bearish: price LH, RSI HH
        if p2_price < p1_price and p2_rsi > p1_rsi:
            strength = 1
            if rsi_diff >= 5:  strength += 1
            if p1_rsi >= 55:   strength += 1
            result["hidden_bearish"] = True
            result["hidden_bearish_strength"] = strength
            labels.append(f"📉 Hidden Bearish Div (str={strength})")

    result["any_bullish"] = result["regular_bullish"] or result["hidden_bullish"]
    result["any_bearish"] = result["regular_bearish"] or result["hidden_bearish"]
    result["divergence_label"] = " | ".join(labels) if labels else ""
    return result

def is_volume_confirmed(row: pd.Series) -> bool:
    """
    [Advanced] ตรวจ Volume Confirmation พร้อม Climax Detection
    - Volume = 0 (เช่น CoinGecko OHLC ไม่มี volume) -> pass through (True)
    - Volume Climax (spike สูง > 3x MA): อาจเป็น exhaustion ไม่ใช่ confirmation
      -> คืน False เพื่อไม่นับเป็น "volume ยืนยัน" ในทิศทางเดิม
    - ปกติ: volume > MA20 = confirmed
    """
    vol    = row.get("volumeto", 0)
    vol_ma = row.get("VOL_MA20", np.nan)

    if vol == 0.0:
        return True  # ไม่มีข้อมูล volume -> ไม่บล็อก
    if pd.isna(vol_ma) or vol_ma <= 0:
        return False

    ratio = vol / vol_ma
    if ratio > 3.0:
        # Volume Climax: spike สูงผิดปกติ
        # ถ้าแท่งนั้นเป็น doji / small body = exhaustion/reversal candle
        # ไม่นับเป็น "confirmed" ในทิศทางเดิม
        body = abs(row.get("close", 0) - row.get("open", 0))
        rng  = row.get("high", 0) - row.get("low", 0)
        if rng > 0 and body / rng < 0.3:
            return False  # climax + doji = exhaustion signal

    return ratio >= 1.0

def format_price(price: float) -> str:
    """
    [Fix N] แก้ไข decimal ranges ให้ครอบคลุมทุก price tier:
    < 0.000001  → 10 decimal (e.g. SHIB 0.0000089)
    < 0.0001    → 8 decimal
    < 0.001     → 6 decimal
    < 0.01      → 5 decimal  ← ช่องว่างที่หายไปเดิม
    < 1         → 4 decimal
    < 10        → 3 decimal  ← ช่องว่างที่หายไปเดิม (e.g. $1.234)
    >= 10       → 2 decimal
    """
    if price is None or pd.isna(price):
        return "N/A"
    if price < 0.000001:  return f"{price:.10f}"
    if price < 0.0001:    return f"{price:.8f}"
    if price < 0.001:     return f"{price:.6f}"
    if price < 0.01:      return f"{price:.5f}"
    if price < 1:         return f"{price:.4f}"
    if price < 10:        return f"{price:.3f}"
    return f"{price:.2f}"

# ==========================================
# [#1] ATR-Based TP/SL Calculation
# ==========================================
def calculate_atr_based_tp_sl(price: float, atr: float, dyn_mult: float, tier: str,
                               ob_info: dict, fvg_info: dict, ema200: float, adx: float = 20.0) -> dict:
    """
    คำนวณ TP1/TP2 จาก ATR (คูณด้วย ATR_TPx_MULTIPLIER) แทนเปอร์เซนต์ fixed
    SL ยังอ้างอิงโครงสร้างราคา (OB/FVG/EMA200) เหมือนเดิม แต่บังคับระยะห่างขั้นต่ำ
    จาก ATR และ MIN_SL_DISTANCE_PCT เพื่อไม่ให้ SL แน่นเกินไปจน noise ปกติโดนเขี่ยออก

    [Volatility-Adjusted SL] ถ้า ADX < CHOPPY_ADX_THRESHOLD (ตลาด sideway/choppy)
    ขยายระยะ SL ออกอีก CHOPPY_SL_WIDEN_MULTIPLIER เท่า เพื่อลดโอกาสโดน whipsaw
    ก่อนที่ trend จริงจะเกิด (position size จะเล็กลงอัตโนมัติเพื่อรักษา $ risk เท่าเดิม)

    Fallback: ถ้า ATR เป็น NaN/0 ใช้เปอร์เซนต์เดิมจาก TP_TIERS
    """
    if pd.isna(atr) or atr <= 0:
        # Fallback to fixed percentage (legacy behavior)
        tp1_pct, tp2_pct, sl_buf = TP_TIERS[tier]["tp1"], TP_TIERS[tier]["tp2"], TP_TIERS[tier]["sl_buffer"]
        tp1_val, tp2_val = price * (1 + tp1_pct), price * (1 + tp2_pct)
        sl_ref = ob_info["bullish_ob_price"] if ob_info["has_bullish_ob"] else fvg_info["fvg_bottom"] if fvg_info["has_fvg_support"] else ema200
        sl_val = sl_ref * (1 - sl_buf) if price > sl_ref else price * (1 - sl_buf)
        return {"tp1": tp1_val, "tp2": tp2_val, "sl": sl_val, "method": "fixed_pct_fallback", "choppy_widened": False}

    # ATR-based TP, scaled by the same dynamic multiplier used for trailing/exit logic
    tp1_val = price + (atr * ATR_TP1_MULTIPLIER * dyn_mult / 2.0)
    tp2_val = price + (atr * ATR_TP2_MULTIPLIER * dyn_mult / 2.0)

    # SL: structure-based reference (OB/FVG/EMA200), but never closer than
    # max(ATR * 1, price * MIN_SL_DISTANCE_PCT) to avoid getting stopped by noise
    sl_buf = TP_TIERS[tier]["sl_buffer"]
    sl_ref = ob_info["bullish_ob_price"] if ob_info["has_bullish_ob"] else fvg_info["fvg_bottom"] if fvg_info["has_fvg_support"] else ema200
    structure_sl = sl_ref * (1 - sl_buf) if price > sl_ref else price * (1 - sl_buf)

    atr_floor_mult = 1.0
    choppy_widened = False
    if not pd.isna(adx) and adx < CHOPPY_ADX_THRESHOLD:
        atr_floor_mult = CHOPPY_SL_WIDEN_MULTIPLIER
        choppy_widened = True

    min_sl_distance = max(atr * atr_floor_mult, price * MIN_SL_DISTANCE_PCT * (CHOPPY_SL_WIDEN_MULTIPLIER if choppy_widened else 1.0))
    min_allowed_sl = price - min_sl_distance
    sl_val = min(structure_sl, min_allowed_sl)  # whichever is FURTHER from price (lower)

    return {"tp1": tp1_val, "tp2": tp2_val, "sl": sl_val, "method": "atr_based", "choppy_widened": choppy_widened}

# ==========================================
# Market Scanner
# ==========================================
def scan_market(positions: dict):
    global _cache_oi_run
    _cache_oi_run = {}  # reset per-run OI cache

    buy_signals, exit_watch_signals, coin_trends_summary, position_updates = [], [], [], []
    supply_zone_alerts = []  # standalone Bearish OB alerts (not in exit_watch or buy signal)
    watch_reversal_signals = []  # [New] positions ที่ RSI oversold มาก — รอ confirm ไม่ trigger action
    stale_coins = []          # coins where data was patched or skipped due to staleness
    bullish_coins, bearish_coins = 0, 0
    # Weight existing open risk by remaining position size (post-partial-TP
    # positions only carry their runner's remaining_size_pct of risk).
    active_signal_count = sum(p.get("remaining_size_pct", 1.0) for p in positions.values())

    # [Re-entry Cooldown] load + will be mutated/saved by caller
    cooldowns = load_cooldowns()

    logger.info("Phase 1: Bulk fetch 4H (A+B+C)...")
    all_4h = bulk_fetch_4h(WATCHLIST)
    logger.info("Phase 2: Bulk fetch 1D (A+B+C)...")
    all_1d = bulk_fetch_1d(WATCHLIST)
    logger.info("Phase 2.5: Bulk fetch 1H (Entry Timing - Approach A)...")
    all_1h = bulk_fetch_1h(WATCHLIST)
    logger.info("Phase 3: Bulk fetch On-chain (1M)...")
    all_onchain = bulk_fetch_onchain(WATCHLIST)
    logger.info("Phase 4: Bulk fetch Funding Rate (Binance Futures)...")
    all_funding = bulk_fetch_funding(WATCHLIST)
    logger.info("Phase 5: Fetch BTC Dominance (CoinGecko Global)...")
    btcd_info = get_btc_dominance_info()
    logger.info("Phase 6: Fetch Macro Economic Calendar (FMP)...")
    macro_info = get_macro_filter_info()
    logger.info("Phase 7: Bulk fetch 24h Volume (Binance Spot)...")
    all_24h_volume = bulk_fetch_24h_volume(WATCHLIST)

    btc_df = all_4h.get("BTC")
    market_regime = "Unknown ⚪"
    if btc_df is not None:
        btc_df = calculate_indicators(btc_df)
        market_regime = "Bull Market 🟢" if btc_df["close"].iloc[-1] > btc_df["EMA_200"].iloc[-1] else "Bear Market 🔴"

    # Pre-calculate BTC returns series for correlation (use % change, not raw price)
    btc_returns = None
    if btc_df is not None:
        btc_returns = btc_df["close"].pct_change().dropna()

    # [Portfolio] Count existing high-BTC-correlation positions (computed
    # fresh each run from current 4H data, since correlation drifts over time)
    existing_high_corr_count = 0

    for coin in COINS:
        df, df_daily = all_4h.get(coin), all_1d.get(coin)
        onchain_info = all_onchain.get(coin, {"has_data": False, "onchain_label": "⚪ N/A"})
        funding_info = get_funding_filter_info(coin, all_funding)
        btcd_filter = get_btc_dominance_filter_for_coin(coin, btcd_info)
        volume_info = get_volume_filter_info(coin, all_24h_volume)

        if df is None or len(df) < EMA_LONG + 10:
            if coin in positions:
                stale_coins.append(f"{coin}(ไม่มีข้อมูล-มี position เปิดอยู่⚠️)")
            continue

        # Check staleness
        hours_old = _check_df_staleness(df, coin, "scan_market")
        if hours_old > DATA_STALENESS_WARN_HOURS_4H:
            stale_coins.append(f"{coin}({hours_old:.0f}h)")

        # Frozen guard ก่อนคำนวณ indicators
        if _is_df_frozen(df, coin):
            logger.error(f"❌ {coin}: frozen data ใน scan_market — skip")
            if coin in positions:
                stale_coins.append(f"{coin}(frozen-มี position ⚠️)")
            else:
                stale_coins.append(f"{coin}(frozen-skip)")
            continue

        weekly_ctx = analyze_weekly_context(df_daily)
        df = calculate_indicators(df)
        row = df.iloc[-1]

        price, rsi, ema50, ema200 = row["close"], row["RSI"], row["EMA_50"], row["EMA_200"]
        atr, adx, vol_confirmed = row["ATR"], row["ADX"], is_volume_confirmed(row)
        adx_rising = is_adx_rising(df)

        # ── Frozen/Corrupt Data Guard ────────────────────────────────────
        # ตรวจหลาย indicator พร้อมกัน:
        # RSI > 95 = impossible ในตลาดจริง
        # Frozen data guard — ใช้ _is_df_frozen ที่ consistent กับ get_historical_data
        # Note: _parse_binance_klines ก็ตรวจ frozen แล้ว แต่ตรวจอีกครั้งหลัง calculate_indicators
        # เพราะ indicator calculation อาจ surface frozen data ได้ชัดขึ้น (RSI 98+)
        is_frozen = _is_df_frozen(df, coin)
        if rsi > 90 and not is_frozen:
            # RSI > 90 แต่ frozen check ผ่าน — log diagnostic เพื่อ debug
            recent_20 = df["close"].iloc[-20:]
            cv_diag = recent_20.std() / recent_20.mean() if recent_20.mean() != 0 else 0
            logger.warning(
                f"⚠️ {coin}: RSI={rsi:.1f} สูงผิดปกติ | "
                f"close_last={recent_20.iloc[-1]:.6f} | "
                f"cv={cv_diag:.6f} | unique={recent_20.nunique()}/20 | "
                f"price_range={recent_20.max()-recent_20.min():.8f}"
            )

        if is_frozen:
            logger.warning(f"⚠️ {coin}: RSI={rsi:.1f}, frozen data detected — trying fresh sources")
            # ลองดึงข้อมูลใหม่จาก Gate.io และ KuCoin โดยตรง (bypass cache)
            fresh_df = None
            for fetch_fn in [_fetch_4h_from_gateio, _fetch_4h_from_kucoin,
                             _fetch_realtime_price_coingecko]:
                if fetch_fn == _fetch_realtime_price_coingecko:
                    rt = fetch_fn(coin)
                    if rt and abs(rt - price) / price > 0.001:
                        df = _patch_df_with_realtime_price(df, coin, rt)
                        df = calculate_indicators(df)
                        row = df.iloc[-1]
                        price, rsi = row["close"], row["RSI"]
                        stale_coins.append(f"{coin}(rt-patched)")
                        logger.info(f"✅ {coin}: real-time patched → ${rt:.6f}, RSI={rsi:.1f}")
                        is_frozen = rsi > 95
                    break
                else:
                    fresh_df = fetch_fn(coin)
                    if fresh_df is not None and len(fresh_df) >= EMA_LONG + 10:
                        df = calculate_indicators(fresh_df)
                        row = df.iloc[-1]
                        price, rsi, ema50, ema200 = row["close"], row["RSI"], row["EMA_50"], row["EMA_200"]
                        atr, adx = row["ATR"], row["ADX"]
                        is_frozen = rsi > 95
                        if not is_frozen:
                            source = "Gate.io" if fetch_fn == _fetch_4h_from_gateio else "KuCoin"
                            logger.info(f"✅ {coin}: fresh data from {source}, RSI={rsi:.1f}")
                            stale_coins.append(f"{coin}({source})")
                            break

            if is_frozen:
                logger.error(f"❌ {coin}: ยังคง frozen หลังลองทุก source — skip coin")
                stale_coins.append(f"{coin}(frozen-skip)")
                continue  # ข้าม coin นี้ทั้งหมด

        vol_confirmed = is_volume_confirmed(row)
        adx_rising = is_adx_rising(df)
        # ─────────────────────────────────────────────────────────────────

        corr_btc = 0.5
        if btc_returns is not None and coin != "BTC":
            coin_returns = df["close"].pct_change().dropna()
            aligned = pd.concat([coin_returns, btc_returns], axis=1, join="inner").dropna()
            if len(aligned) >= 30:
                corr_val = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if not pd.isna(corr_val):
                    corr_btc = corr_val
        elif coin == "BTC":
            corr_btc = 1.0

        # Tally existing open positions that are highly correlated to BTC
        if coin in positions and corr_btc >= HIGH_CORR_BTC_THRESHOLD:
            existing_high_corr_count += 1

        # --- Monitor existing open position for this coin (if any) ---
        if coin in positions:
            pos = positions[coin]
            status = check_position_status(pos, price, atr, adx_rising=adx_rising, current_adx=adx)
            if status["action"] in ("close_sl", "close_tp2", "close_runner_sl", "time_stop", "give_back_warn"):
                position_updates.append({
                    "coin": coin, "price": format_price(price),
                    "action": status["action"], "notes": status["notes"],
                    "entry_price": format_price(pos["entry_price"]),
                })
                if status["action"] in ("close_sl", "close_tp2", "close_runner_sl"):
                    close_position(positions, coin, status["action"], exit_price=price, cooldowns=cooldowns)
                # time_stop / give_back_warn: leave position open but surface
                # the warning; user decides manually whether to close.
                elif status["action"] != "time_stop":
                    # give_back_warn: also apply any trailing SL update computed
                    if status["new_sl"] != pos["sl"]:
                        pos["sl"] = status["new_sl"]
            elif status["action"] in ("partial_tp1", "partial_tp1_5"):
                position_updates.append({
                    "coin": coin, "price": format_price(price),
                    "action": status["action"], "notes": status["notes"],
                    "entry_price": format_price(pos["entry_price"]),
                })
                size_pct = (PARTIAL_TP1_CLOSE_PCT * 100) if status["action"] == "partial_tp1" else (TP1_5_CLOSE_PCT * 100)
                log_partial_close(pos, coin, exit_price=price, event=status["action"], size_closed_pct=round(size_pct, 1))
                pos["sl"] = status["new_sl"]
                # position stays open (as the runner) — not closed/deleted
            else:
                if status["new_sl"] != pos["sl"]:
                    position_updates.append({
                        "coin": coin, "price": format_price(price),
                        "action": "update_sl", "notes": status["notes"],
                        "entry_price": format_price(pos["entry_price"]),
                    })
                    pos["sl"] = status["new_sl"]
                elif status["notes"]:
                    position_updates.append({
                        "coin": coin, "price": format_price(price),
                        "action": "info", "notes": status["notes"],
                        "entry_price": format_price(pos["entry_price"]),
                    })

        tier = COIN_TIER.get(coin, "mid")
        atr_pct = (atr / price) * 100 if price > 0 and not pd.isna(atr) else 0.0
        dyn_mult = get_dynamic_atr_multiplier(tier, adx, atr_pct, df=df)

        trend_info, bounce_info = analyze_trend_continuity(df), analyze_rsi_bounce(df)
        ob_info, fvg_info = find_order_blocks(df), find_fair_value_gaps(df)
        df_1h_coin = all_1h.get(coin)
        candle_info, mtf_info = confirm_reversal_candle(df), get_mtf_rsi_alignment(df, df_daily, df_1h=df_1h_coin)
        bearish_candle_info = confirm_bearish_reversal_candle(df)
        # Advanced divergence — single call, all 4 types + strength
        div_info     = check_divergence_advanced(df)
        is_div       = div_info["any_bullish"]
        is_bear_div  = div_info["any_bearish"]
        time_stop_info = check_time_stop(df)
        trend_1d = check_1d_trend_alignment(df_daily)
        # [Bearish OB] Supply Zone alert — computed for every coin regardless
        # of signal/position, so it surfaces in both exit-watch and buy-signal
        bearish_ob_alert = get_bearish_ob_alert(ob_info, price)

        in_fibo = weekly_ctx["fibo_618"] is not None and price <= weekly_ctx["fibo_618"] * 1.02
        in_ob = ob_info["has_bullish_ob"] and price <= ob_info["bullish_ob_top"] * 1.03
        in_fvg = fvg_info["has_fvg_support"] and fvg_info["fvg_bottom"] * 0.99 <= price <= fvg_info["fvg_top"]

        signal_type = ""
        signal_reason = ""  # เหตุผลสั้นๆ สำหรับ log
        bias_warning = ""

        if price > ema200:
            bullish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🟢 ขาขึ้น | ${format_price(price)} | RSI: {rsi:.1f}\n"
                f"  └ {trend_info['trend_label']}"
            )

            ts = trend_info.get("trend_strength", "sideways")
            er = trend_info.get("efficiency_ratio", 0.0)
            hh = trend_info.get("higher_high", False)
            hl = trend_info.get("higher_low", False)

            # ─────────────────────────────────────────────────────────────
            # Signal A: Dip & Rebound (เดิม — ปรับ RSI ceiling)
            # เหรียญ pullback มาแตะ support แล้วดีดกลับ
            # [Adjusted] RSI ceiling ลดจาก 65 → 55 ให้ใกล้ oversold มากขึ้น
            # ─────────────────────────────────────────────────────────────
            if (in_fibo or in_ob or in_fvg) and price > (ema50 * 0.97) and rsi <= 55:
                if bounce_info["quality"] in ["strong", "moderate"]:
                    signal_type = "Institution Dip & Rebound 📉"
                    signal_reason = f"pullback ที่ key level, RSI {rsi:.1f}"

            # ─────────────────────────────────────────────────────────────
            # Signal B: Bullish Divergence (เดิม)
            # ─────────────────────────────────────────────────────────────
            if is_div and not signal_type:
                signal_type = "Confluence Bullish Divergence 📈"
                signal_reason = div_info.get("divergence_label", "bullish divergence")

            # ─────────────────────────────────────────────────────────────
            # Signal C: Pullback in Uptrend
            # เหรียญ uptrend แข็ง (HH+HL) แล้ว RSI ดึงกลับมาพักใกล้ oversold
            # แล้วดีดกลับ — เหมาะกับ "ช้อนซื้อใน pullback ของ trend"
            # [Adjusted] RSI ceiling ลดจาก 58 → 45 ให้ใกล้ oversold มากขึ้น
            # ตามหลักการ RSI ต่ำ = เหตุผลซื้อ แม้จะเป็นสัญญาณแบบ pullback ก็ตาม
            # ─────────────────────────────────────────────────────────────
            if (not signal_type
                    and ts in ("strong_up", "moderate_up")
                    and hh and hl                          # structure ยืนยัน uptrend
                    and 30 <= rsi <= 45                    # ใกล้ oversold มากขึ้น (เดิม 38-58)
                    and price > ema50 * 0.95               # ราคายังไม่หลุด EMA50 มากเกิน
                    and bounce_info["quality"] in ("strong", "moderate", "weak")):
                signal_type = "Pullback in Uptrend 🔄"
                signal_reason = f"HH+HL structure, RSI pullback {rsi:.1f}, bounce {bounce_info['quality']}"

            # ─────────────────────────────────────────────────────────────
            # Signal D: Momentum Continuation / Breakout
            # เหรียญพุ่งขึ้นแรง (ER สูง, ADX แข็ง) แต่ RSI ยังไม่ overbought
            # [Adjusted] RSI ceiling ลดจาก 70 → 50 ให้ใกล้ oversold มากขึ้น
            # ป้องกันการซื้อตอน RSI สูง ซึ่งขัดกับหลักการ RSI สูง=ขาย
            # ─────────────────────────────────────────────────────────────
            if (not signal_type
                    and ts == "strong_up"
                    and er >= 0.5                          # ราคาเดินทางตรง (impulsive)
                    and adx >= 25                          # trend แข็งแกร่ง
                    and rsi <= 50                           # ลดจาก 70 → 50 (ไม่เข้าใกล้ overbought)
                    and price > ema50                      # ราคาอยู่เหนือ EMA50
                    and vol_confirmed):                    # volume ยืนยัน
                signal_type = "Momentum Continuation 🚀"
                signal_reason = f"ER:{er:.2f}, ADX:{adx:.1f}, RSI:{rsi:.1f}"

            # ─────────────────────────────────────────────────────────────
            # Signal E: EMA50 Bounce
            # ราคา pullback มาแตะ EMA50 แล้วดีดกลับ — classic mid-trend reentry
            # [Adjusted] RSI ceiling ลดจาก 60 → 45 ให้ใกล้ oversold มากขึ้น
            # ─────────────────────────────────────────────────────────────
            if (not signal_type
                    and ts in ("strong_up", "moderate_up")
                    and ema50 > 0
                    and price >= ema50 * 0.98 and price <= ema50 * 1.03  # อยู่แถว EMA50
                    and rsi <= 45                           # ลดจาก 60 → 45
                    and bounce_info["quality"] in ("strong", "moderate")):
                signal_type = "EMA50 Bounce 🛡️"
                signal_reason = f"ราคากลับมา EMA50 ({format_price(ema50)}), RSI {rsi:.1f}"

            # --- Exit watch (only meaningful for coins in an uptrend, i.e. likely held as long) ---
            near_ob_target = False
            est_target = estimate_price_for_target_rsi(df)
            if est_target is not None and price > 0:
                near_ob_target = (est_target - price) / price <= 0.02  # within 2% of estimated RSI-70 price

            # [Fix G] Standalone TP Target Reached notification
            # ถ้าราคาถึงหรือเกิน Dynamic TP estimate (RSI-70 price) → แจ้งแยกชัดเจน
            if est_target is not None and price >= est_target and coin in positions:
                pnl_pct = round(((price - positions[coin]["entry_price"]) / positions[coin]["entry_price"]) * 100, 2)
                tp_alert = {
                    "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                    "exit_score": 85, "rsi_hard_alert": rsi >= RSI_EXIT_HARD_OVERRIDE,
                    "ext_from_ema200_pct": round(((price - ema200) / ema200) * 100, 1) if ema200 > 0 else 0.0,
                    "below_ema200_position": False,
                    "exit_label": f"🎯 <b>Dynamic TP Target ถูกแตะแล้ว!</b> | PnL: +{pnl_pct:.2f}% | Exit Score: 85/100",
                    "exit_reasons": [
                        f"🎯 ราคา ({format_price(price)}) ถึง Dynamic TP ประเมิน RSI-70 ({format_price(est_target)})",
                        f"📈 PnL จาก Entry: +{pnl_pct:.2f}%",
                        f"🟠 แนะนำปิด 30-50% หรือเลื่อน SL ขึ้นมาล็อคกำไร",
                    ],
                    "trend_label": trend_info["trend_label"],
                    "time_stop_label": "",
                    "bearish_label": "",
                    "supply_zone_label": bearish_ob_alert.get("supply_zone_label", ""),
                }
                exit_watch_signals.append(tp_alert)
                signal_type = ""  # mutual exclusion
                logger.info(f"🎯 {coin}: Dynamic TP target reached at {format_price(price)} (+{pnl_pct:.2f}%)")

            exit_score, exit_label, exit_reasons = calculate_exit_score(
                rsi, bearish_candle_info, vol_confirmed, mtf_info, adx,
                is_bear_div, trend_info, onchain_info, price, ema50, ema200, near_ob_target,
                df=df, div_info=div_info,
            )
            # [Bearish OB] ถ้าราคาเข้าใกล้ Supply Zone -> เพิ่ม exit score + เพิ่ม reason
            if bearish_ob_alert.get("near_bearish_ob") and ob_info.get("bearish_ob_strength", 0) >= 2:
                boost = 15 if ob_info["bearish_ob_strength"] == 3 else 8
                exit_score = min(100, exit_score + boost)
                exit_reasons.append(f"ราคาใกล้ Supply Zone / Bearish OB ({ob_info['bearish_ob_label']})")
                exit_label = (
                    f"🚨 <b>สัญญาณเตือนแรง</b> | Exit Score: {exit_score}/100"
                    if exit_score >= 70
                    else f"⚠️ ควรพิจารณาลดสถานะ/เลื่อน SL | Exit Score: {exit_score}/100"
                )

            # [Hard Override] RSI >= RSI_EXIT_HARD_OVERRIDE (78) -> force เข้า Exit Watch เสมอ
            # [Soft Override] RSI >= RSI_EXIT_WARN_THRESHOLD (72) -> ลด effective threshold เป็น 25
            rsi_hard_alert = rsi >= RSI_EXIT_HARD_OVERRIDE
            effective_exit_threshold = MINIMUM_EXIT_SCORE

            # [RSI Prerequisite Gate]
            # Exit Watch สำหรับ coin ที่ไม่มี open position:
            # ต้องการ RSI >= 60 เป็นขั้นต่ำ เพราะ RSI < 60 + Sideways
            # ไม่ใช่สัญญาณ "ควรขาย" ที่แท้จริง — เป็นแค่ noise จาก ADX/Volume/EMA
            # (coin ที่มี open position ผ่านได้เสมอผ่าน Fix H ที่เรียก calculate_exit_score แยก)
            if rsi < 60 and coin not in positions:
                # ยก threshold สูงขึ้นเป็น 65 สำหรับ RSI < 60 (ต้องมี factor แรงกว่าปกติ)
                effective_exit_threshold = 65

            if rsi >= RSI_EXIT_HARD_OVERRIDE:
                effective_exit_threshold = 0  # force ผ่าน
                if exit_score < MINIMUM_EXIT_SCORE:
                    exit_score = max(exit_score, MINIMUM_EXIT_SCORE)
                    if not any("RSI" in r and str(int(rsi)) in r for r in exit_reasons[:3]):
                        exit_reasons.insert(0, f"🔴 RSI {rsi:.1f} — สูงเกินเกณฑ์ (Hard Override)")
                    exit_label = f"🚨 <b>RSI {rsi:.1f} สูงมาก — แบ่งขายได้เลย</b> | Exit Score: {exit_score}/100"
            elif rsi >= RSI_EXIT_WARN_THRESHOLD:
                effective_exit_threshold = min(effective_exit_threshold, 25)  # lower threshold

            if exit_score >= effective_exit_threshold:
                ext_pct = round(((price - ema200) / ema200) * 100, 1) if ema200 > 0 else 0.0
                exit_watch_signals.append({
                    "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                    "exit_score": exit_score, "exit_label": exit_label,
                    "exit_reasons": exit_reasons,
                    "trend_label": trend_info["trend_label"],
                    "time_stop_label": time_stop_info["time_stop_label"],
                    "bearish_label": bearish_candle_info["bearish_label"],
                    "supply_zone_label": bearish_ob_alert.get("supply_zone_label", ""),
                    "rsi_hard_alert": rsi_hard_alert,
                    "ext_from_ema200_pct": ext_pct,
                })
                # [Mutual Exclusion] ถ้าเหรียญอยู่ใน exit watch แล้ว
                # ไม่ควรให้สัญญาณซื้อในรอบเดียวกัน — ลบ signal_type ออก
                # เพื่อไม่ให้ Telegram แสดงทั้ง "ซื้อ" และ "ขาย" พร้อมกัน
                if signal_type:
                    logger.info(f"🔄 {coin}: ล้าง signal_type '{signal_type}' "
                                f"เพราะอยู่ใน Exit Watch (RSI:{rsi:.1f}, exit_score:{exit_score})")
                    signal_type = ""

            elif bearish_ob_alert.get("near_bearish_ob") and bearish_ob_alert.get("alert_level") in ("danger", "caution"):
                exit_watch_coins = {s["coin"] for s in exit_watch_signals}
                if coin not in exit_watch_coins:
                    supply_zone_alerts.append({
                        "coin": coin,
                        "price": format_price(price),
                        "supply_zone_label": bearish_ob_alert["supply_zone_label"],
                    })

        else:
            bearish_coins += 1
            ts = trend_info.get("trend_strength", "sideways")
            if ts == "strong_down":
                bias_warning = "⚠️ <b>เทรนด์ขาลงแข็งแกร่ง — หลีกเลี่ยงการเปิด Long ใหม่</b>"
            elif ts == "moderate_down":
                bias_warning = "⚠️ เทรนด์ขาลง — รอสัญญาณกลับตัวที่ชัดเจนก่อนเข้า Long"
            else:
                bias_warning = "🔻 ราคาต่ำกว่า EMA200 — ระมัดระวัง"

            coin_trends_summary.append(
                f"• {coin}: 🔴 ขาลง | ${format_price(price)} | RSI: {rsi:.1f}\n"
                f"  └ {trend_info['trend_label']} | {bias_warning}"
            )

            # ─────────────────────────────────────────────────────────────
            # [Fix H] Exit Watch สำหรับ positions ที่ราคาหลุด EMA200
            # ถ้ามี open position บน coin นี้ ต้องตรวจ exit_score เสมอ
            # ไม่ว่าราคาจะอยู่เหนือหรือต่ำกว่า EMA200
            #
            # [Approach 2 + ATR Proximity] Live Price Check + ATR-based threshold:
            # ตรวจราคาจริง ณ ขณะนี้ (ไม่ใช่ close ของแท่ง 4H ที่ยังไม่ปิด)
            # และจัดระดับความรุนแรงตามระยะห่างจาก EMA200 เทียบกับ ATR
            # (แทนที่จะใช้ % คงที่ ซึ่งไม่ fair ระหว่างเหรียญ volatility ต่างกัน)
            #   distance < 0.75x ATR  → "ใกล้เส้น" ลด severity, ให้ live price ตัดสิน
            #   distance >= 0.75x ATR → "หลุดจริงชัดเจน" คง severity เดิม
            # ─────────────────────────────────────────────────────────────
            if coin in positions:
                near_ob_target = False
                est_target = estimate_price_for_target_rsi(df)
                if est_target is not None and price > 0:
                    near_ob_target = (est_target - price) / price <= 0.02

                exit_score_b, exit_label_b, exit_reasons_b = calculate_exit_score(
                    rsi, bearish_candle_info, vol_confirmed, mtf_info, adx,
                    is_bear_div, trend_info, onchain_info, price, ema50, ema200, near_ob_target,
                    df=df, div_info=div_info,
                )

                # ── ATR-based proximity classification ──
                ema200_distance = abs(price - ema200)
                atr_safe = atr if atr and atr > 0 else (price * 0.01)  # fallback 1% ถ้า ATR ไม่มี
                proximity_threshold = atr_safe * EMA200_PROXIMITY_ATR_MULTIPLIER
                atr_multiple = ema200_distance / atr_safe if atr_safe > 0 else 0.0
                is_near_ema200_line = ema200_distance < proximity_threshold

                is_below_ema200_closed = price < ema200
                is_below_ema200_live   = is_below_ema200_closed  # default ถ้า live check ล้มเหลว
                live_price = None

                if is_below_ema200_closed:
                    live_price = _fetch_live_price_binance(coin)
                    if live_price is not None and live_price > 0:
                        is_below_ema200_live = live_price < ema200
                        if not is_below_ema200_live:
                            logger.info(
                                f"✅ {coin}: closed candle ({format_price(price)}) < EMA200 "
                                f"แต่ live price ({format_price(live_price)}) > EMA200 — ไม่ trigger severe alert"
                            )

                # ── RSI Oversold → Watch for Reversal (แยกออกจาก Exit Watch ทั้งหมด) ──
                # ไม่ trigger ทั้ง buy หรือ sell action — แค่แจ้งให้จับตาดู
                # รอ confirmation (RSI ดีดขึ้น / bounce ยืนยัน) ก่อนตัดสินใจ
                is_rsi_oversold_override = is_below_ema200_closed and is_below_ema200_live and rsi <= RSI_OVERSOLD

                if is_rsi_oversold_override:
                    watch_reversal_signals.append({
                        "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                        "ema200": format_price(ema200),
                        "atr_multiple": round(atr_multiple, 2),
                        "live_price": format_price(live_price) if live_price else None,
                        "trend_label": trend_info["trend_label"],
                        "note": (
                            f"ราคาหลุด EMA200 ชัดเจน ({atr_multiple:.2f}x ATR) แต่ RSI={rsi:.1f} "
                            f"อยู่ในโซน Oversold (≤{RSI_OVERSOLD}) — ระบบไม่แนะนำทั้งซื้อเพิ่มหรือขาย "
                            f"จนกว่าจะมีสัญญาณยืนยันทิศทางชัดเจนกว่านี้"
                        ),
                    })
                    # ไม่ trigger Exit Watch สำหรับ coin นี้ในรอบนี้ — แต่ยังให้ logic อื่น
                    # (เช่น Fix A ด้านล่าง) ทำงานต่อตามปกติ ไม่ใช้ continue เพื่อไม่ skip
                    # ส่วนอื่นของ loop ที่อาจจำเป็น (เช่น trend summary ที่ append ไปแล้วด้านบน)
                    skip_exit_watch_for_coin = True
                else:
                    skip_exit_watch_for_coin = False

                if not skip_exit_watch_for_coin and is_below_ema200_closed and is_below_ema200_live:
                    if is_near_ema200_line:
                        # ราคาหลุดจริงแต่ระยะใกล้เส้นมาก (< 0.75x ATR) — ลด severity
                        # เพราะอาจเป็นแค่ noise ปกติของแท่งเดียว ไม่ใช่ trend break จริง
                        exit_score_b = min(100, exit_score_b + 10)  # บวกน้อยกว่าเดิม (25→10)
                        exit_reasons_b.insert(0,
                            f"🟡 ราคาใกล้เส้น EMA200 ({format_price(ema200)}) — ห่างเพียง "
                            f"{atr_multiple:.2f}x ATR (< {EMA200_PROXIMITY_ATR_MULTIPLIER}x) "
                            f"อาจเป็น noise ปกติ ไม่ใช่ trend break ชัดเจน"
                        )
                        exit_label_b = (
                            f"⚠️ <b>ราคาใกล้เส้น EMA200 — เฝ้าระวัง</b> | Exit Score: {exit_score_b}/100"
                        )
                    else:
                        # หลุดชัดเจน ห่างเกิน 0.75x ATR และ RSI ไม่ oversold — severity เต็ม
                        exit_score_b = min(100, exit_score_b + 25)
                        exit_reasons_b.insert(0,
                            f"🔴 ราคาหลุด EMA200 ({format_price(ema200)}) ชัดเจน "
                            f"({atr_multiple:.2f}x ATR) — สถานะ Long เสี่ยงสูง"
                        )
                        exit_label_b = (
                            f"🚨 <b>ราคาหลุด EMA200 — พิจารณาปิดสถานะ Long</b> | Exit Score: {exit_score_b}/100"
                        )
                elif is_below_ema200_closed and not is_below_ema200_live:
                    # closed candle ต่ำกว่า EMA200 แต่ live price กลับมายืนเหนือแล้ว
                    exit_reasons_b.append(
                        f"ℹ️ Live price (${format_price(live_price)}) กลับมายืนเหนือ EMA200 แล้ว "
                        f"— รอแท่งถัดไปยืนยัน (closed candle ${format_price(price)} ยังต่ำกว่า)"
                    )

                rsi_hard_b = rsi >= RSI_EXIT_HARD_OVERRIDE
                eff_threshold_b = 0 if rsi_hard_b else (25 if rsi >= RSI_EXIT_WARN_THRESHOLD else MINIMUM_EXIT_SCORE)

                # force-include เฉพาะเมื่อหลุดชัดเจน (ไม่ใช่ near-line) และไม่ใช่ RSI oversold case
                force_include = is_below_ema200_live and not is_near_ema200_line and not skip_exit_watch_for_coin
                if not skip_exit_watch_for_coin and (exit_score_b >= eff_threshold_b or force_include):
                    ext_pct_b = round(((price - ema200) / ema200) * 100, 1) if ema200 > 0 else 0.0
                    exit_watch_signals.append({
                        "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                        "exit_score": exit_score_b, "exit_label": exit_label_b,
                        "exit_reasons": exit_reasons_b,
                        "trend_label": trend_info["trend_label"],
                        "time_stop_label": time_stop_info["time_stop_label"],
                        "bearish_label": bearish_candle_info["bearish_label"],
                        "supply_zone_label": bearish_ob_alert.get("supply_zone_label", ""),
                        "rsi_hard_alert": rsi_hard_b,
                        "ext_from_ema200_pct": ext_pct_b,
                        "below_ema200_position": is_below_ema200_live and not is_near_ema200_line,
                        "live_price": format_price(live_price) if live_price else None,
                        "atr_multiple": round(atr_multiple, 2),
                        "is_near_ema200_line": is_near_ema200_line,
                    })

            # ─────────────────────────────────────────────────────────────
            # [Fix A] Extreme Oversold below EMA200 (RSI < 25)
            # ไม่ต้องรอ weekly fibo_786 — RSI ต่ำสุดระดับนี้หายากมาก
            # ─────────────────────────────────────────────────────────────
            if rsi < 25 and not signal_type:
                if bounce_info["quality"] in ("strong", "moderate") or is_div:
                    signal_type = "⚡ Extreme Oversold Reversal 🔄"
                    signal_reason = f"RSI extreme {rsi:.1f} + {'div' if is_div else bounce_info['quality']+' bounce'}"

            if weekly_ctx.get("fibo_786") and price <= weekly_ctx["fibo_786"] * 1.02:
                if is_div and not signal_type:
                    signal_type = "🚨 DEEP REVERSAL + Bullish Div 🐳"
                    signal_reason = "deep reversal + bullish divergence"
                elif bounce_info["quality"] == "strong" and not signal_type:
                    signal_type = "🛡️ Deep Support Strong Bounce 📉"
                    signal_reason = "strong bounce at weekly 78.6% fibo"

        if signal_type and coin not in positions:
            logger.info(f"💡 {coin}: signal candidate '{signal_type}' ({signal_reason})")

            # [#5] Macro News Filter: block ALL new entries near high-impact USD events
            if macro_info.get("block_new_entries"):
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — Macro Event ใกล้เวลานี้")
                continue

            # [#3] Funding Rate hard filter: block brand-new longs if funding too hot
            # Exception: Momentum Continuation signals ใน strong uptrend
            # ยอมให้ผ่านได้แม้ funding สูง เพราะ cost of missing the move > funding cost
            # แต่ต้องไม่เกิน 2x threshold (FUNDING_RATE_MAX_LONG * 2)
            if funding_info.get("block_long"):
                fr = funding_info.get("funding_rate", 0) or 0
                is_momentum_signal = "Momentum" in signal_type or "Pullback" in signal_type
                if is_momentum_signal and fr < FUNDING_RATE_MAX_LONG * 2:
                    # warn แต่ไม่ block — บันทึก note ใน advisory แทน
                    logger.info(f"⚠️ {coin}: funding สูง ({fr*100:.3f}%) แต่ผ่านได้เพราะ Momentum signal")
                else:
                    logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — {funding_info['funding_label']}")
                    continue

            # [Entry Filter] 24h Volume floor — illiquid alts get skipped entirely
            if not volume_info.get("passes", True):
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — {volume_info['label']}")
                continue

            # [Entry Filter] 1D EMA50 trend alignment — daily trend must agree
            # Exception: Extreme Oversold (RSI<=25) หรือ Strong Divergence ได้รับการยกเว้น
            # เหตุผล: ช่วง market-wide correction ราคา 4H oversold มักเกิดพร้อมกับ
            # 1D ยังไม่ทันกลับตัวเหนือ EMA50 — ถ้า block แบบไม่มีข้อยกเว้นเลย
            # จะพลาดจังหวะช้อนซื้อที่ดีที่สุดของระบบ (deep oversold reversal)
            is_extreme_oversold_exception = (
                rsi <= RSI_OVERSOLD
                or (div_info.get("regular_bullish") and div_info.get("regular_bullish_strength", 0) >= 2)
                or "Extreme Oversold" in signal_type
                or "DEEP REVERSAL" in signal_type
                or "Deep Support" in signal_type
            )
            if not trend_1d.get("aligned", True):
                if is_extreme_oversold_exception:
                    logger.info(
                        f"⚠️ {coin}: 1D trend ไม่ aligned แต่ผ่านได้เพราะ Extreme Oversold/Strong Div "
                        f"(RSI:{rsi:.1f}) — {trend_1d['label']}"
                    )
                else:
                    logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — {trend_1d['label']}")
                    continue

            # [Entry Filter] Re-entry cooldown after a recent SL loss
            in_cd, cd_hours_left = is_in_cooldown(coin, cooldowns)
            if in_cd:
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — Re-entry Cooldown (เหลือ {cd_hours_left:.1f} ชม.)")
                continue

            # [Portfolio] Max concurrent positions cap
            # ใช้ len(positions) + new_positions_this_run เพื่อนับ position ที่เพิ่งเปิดในรอบนี้ด้วย
            # (positions dict อัปเดต in-place ทุกครั้งที่ open_position ถูกเรียก)
            if len(positions) >= MAX_CONCURRENT_POSITIONS:
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — เปิด position ครบ {MAX_CONCURRENT_POSITIONS} แล้ว (Max Concurrent)")
                continue

            # [Portfolio] Correlation-based rejection: too many high-BTC-corr positions already
            if corr_btc >= HIGH_CORR_BTC_THRESHOLD and existing_high_corr_count >= MAX_HIGH_CORR_POSITIONS:
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — มี position ที่ correlation สูงกับ BTC ครบ {MAX_HIGH_CORR_POSITIONS} แล้ว")
                continue

            # [#3] Open Interest filter: only fetched for candidates that already
            # passed the basic signal_type + funding gate (saves API calls)
            oi_info = get_oi_filter_info(coin, df)

            sig_score, score_label = calculate_signal_score(
                rsi, bounce_info, candle_info, vol_confirmed,
                in_fibo, in_ob, in_fvg, weekly_ctx, mtf_info, adx, is_div, trend_info, onchain_info,
                funding_info=funding_info, oi_info=oi_info, btcd_filter=btcd_filter,
                div_info=div_info, signal_type=signal_type,
            )

            # [Approach A] 1H timing bonus: บวกคะแนนเมื่อ 1H RSI กำลัง bounce
            # timing_score 0-10 → max +8 pts (ไม่ให้เกินเพื่อป้องกัน false positive)
            timing_score = mtf_info.get("timing_score", 0)
            if timing_score > 0:
                timing_bonus = min(timing_score * 0.8, 8)
                sig_score = min(100, sig_score + int(timing_bonus))
                if timing_bonus >= 4:
                    score_label = score_label.replace("| Score:", f"| +{int(timing_bonus)}pts 1H⚡ | Score:")


            if sig_score >= MINIMUM_SIGNAL_SCORE:
                # [#1 + Choppy-SL] ATR-based TP/SL, widened in low-ADX regimes
                tpsl = calculate_atr_based_tp_sl(price, atr, dyn_mult, tier, ob_info, fvg_info, ema200, adx=adx)
                tp1_val, tp2_val, sl_val = tpsl["tp1"], tpsl["tp2"], tpsl["sl"]

                # [Bearish OB] ปรับ TP1 ลงถ้ามี Supply Zone แข็งแกร่งอยู่ระหว่าง TP1-TP2
                tp_ceiling = get_bearish_ob_as_tp_ceiling(ob_info, tp1_val, tp2_val)
                if tp_ceiling["adjusted"]:
                    tp1_val = tp_ceiling["tp1"]

                sl_dist = max((price - sl_val) / price, 0.01)

                pos_size = get_correlation_adjusted_position(PORTFOLIO_USDT, RISK_PER_TRADE_PCT, sl_dist, corr_btc, active_signal_count)
                active_signal_count += 1
                if corr_btc >= HIGH_CORR_BTC_THRESHOLD:
                    existing_high_corr_count += 1

                # Bearish bias warning for downside reversal setups (signal entered below EMA200)
                downside_warning = ""
                if price <= ema200 and bias_warning:
                    downside_warning = "\n" + bias_warning

                # Funding/OI/BTC.D/Macro/1D-trend/Volume/SupplyZone advisory notes
                advisory_lines = []
                if funding_info.get("has_data"):
                    advisory_lines.append(f"💸 {funding_info['funding_label']}")
                if oi_info.get("has_data"):
                    advisory_lines.append(f"📊 {oi_info['oi_label']}")
                if btcd_filter.get("note"):
                    advisory_lines.append(f"🌐 {btcd_filter['note']}")
                if macro_info.get("has_data") and macro_info.get("events"):
                    advisory_lines.append(f"📅 {macro_info['macro_label']}")
                if trend_1d.get("has_data"):
                    advisory_lines.append(f"📆 {trend_1d['label']}")
                if volume_info.get("has_data"):
                    advisory_lines.append(f"💧 24h Vol: ${volume_info['volume_24h']:,.0f}")
                if tpsl.get("choppy_widened"):
                    advisory_lines.append(f"🌀 ตลาด Choppy (ADX<{CHOPPY_ADX_THRESHOLD}) — ขยาย SL x{CHOPPY_SL_WIDEN_MULTIPLIER} แล้ว")
                # Supply Zone advisory (always show if OB exists)
                advisory_lines.append(bearish_ob_alert["supply_zone_label"])
                if tp_ceiling["adjusted"]:
                    advisory_lines.append(tp_ceiling["note"])
                advisory_block = ("\n" + "\n".join(advisory_lines)) if advisory_lines else ""

                tp1_5_price = tp1_val + (tp2_val - tp1_val) * (ATR_TP1_5_MULTIPLIER - ATR_TP1_MULTIPLIER) / max(ATR_TP2_MULTIPLIER - ATR_TP1_MULTIPLIER, 0.0001)

                buy_signals.append({
                    "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                    "type": signal_type, "score_label": score_label,
                    "tp1": f"${format_price(tp1_val)}",
                    "tp1_5": f"${format_price(tp1_5_price)}",
                    "tp2": f"${format_price(tp2_val)}",
                    "dynamic_tp": f"${format_price(estimate_price_for_target_rsi(df) or price * 1.1)}",
                    "sl": f"${format_price(sl_val)}", "pos_size": f"${pos_size:.2f}",
                    "sl_risk_pct": f"{sl_dist*100:.1f}%", "vol_confirmed": vol_confirmed,
                    "mtf_label": mtf_info.get("mtf_label", ""), "reversal_label": candle_info["reversal_label"],
                    "trend_label": trend_info["trend_label"], "bounce_label": bounce_info["quality_label"],
                    "onchain_label": onchain_info["onchain_label"],
                    "corr_btc": round(corr_btc, 2),
                    "time_stop_label": time_stop_info["time_stop_label"],
                    "downside_warning": downside_warning,
                    "tp_method": tpsl["method"],
                    "advisory_block": advisory_block,
                    "divergence_label": div_info.get("divergence_label", ""),
                    "wyckoff_label": weekly_ctx.get("wyckoff_label", ""),
                    "er_label": f"ER:{trend_info.get('efficiency_ratio', 0):.2f}" if trend_info.get("efficiency_ratio") is not None else "",
                    "timing_label": mtf_info.get("timing_label", ""),
                    "rsi_1h": mtf_info.get("rsi_1h"),
                    "is_1h_bouncing": mtf_info.get("is_1h_bouncing", False),
                    "staleness": time_stop_info.get("staleness_score", 0),
                    "partial_tp_note": (
                        f"📐 Partial TP: ปิด {PARTIAL_TP1_CLOSE_PCT*100:.0f}% ที่ TP1, "
                        f"ปิดอีก {TP1_5_CLOSE_PCT*100:.0f}% ที่ TP1.5, "
                        f"เหลือ {100-PARTIAL_TP1_CLOSE_PCT*100-TP1_5_CLOSE_PCT*100:.0f}% เป็น Runner "
                        f"(Trail ATR x{RUNNER_TRAIL_ATR_MULTIPLIER}, Extension x{RUNNER_EXTEND_ATR_MULTIPLIER} ถ้าเทรนด์แข็งแกร่ง)"
                    ),
                })

                # Record the new position so future runs track SL/TP/trailing/partial
                open_position(positions, coin, price, sl_val, tp1_val, tp2_val, tier, atr_at_entry=atr)


    # Persist re-entry cooldowns (mutated in-place during the loop)
    save_cooldowns(cooldowns)

    # [Fix J] Portfolio-level overbought meta-alert
    # ถ้า exit_watch signals >= 3 ตัว (หรือ >= ครึ่งของ positions) → แจ้ง portfolio-wide warning
    portfolio_alert = ""
    if len(exit_watch_signals) >= 3:
        high_score_exits = [s for s in exit_watch_signals if s["exit_score"] >= 60]
        if len(high_score_exits) >= 2:
            coins_str = ", ".join(s["coin"] for s in high_score_exits)
            portfolio_alert = (
                f"\n🚨 <b>Portfolio-Wide Warning:</b> {len(high_score_exits)} เหรียญ "
                f"({coins_str}) มีสัญญาณอ่อนแรงพร้อมกัน — พิจารณาลด exposure โดยรวม"
            )

    stale_line = ""
    if stale_coins:
        stale_line = (
            f"\n⚠️ <b>Data Staleness Warning:</b> {', '.join(stale_coins)} "
            f"— ข้อมูลอาจไม่ตรงกับตลาดจริง (CoinGecko OHLC delay) "
            f"ราคาจะถูก patch ด้วย real-time price อัตโนมัติถ้าเป็นไปได้"
        )

    summary_msg = (
        f"📊 <b>[Market Summary – v9-Final]</b>\n"
        f"ดัชนีหลัก (BTC): <b>{market_regime}</b>\n"
        f"🌐 {btcd_info.get('btcd_label', '⚪ N/A BTC.D')}\n"
        f"📅 {macro_info.get('macro_label', '')}\n"
        f"📈 ขาขึ้น: {bullish_coins} | 📉 ขาลง: {bearish_coins}\n"
        f"💼 Positions: {len(positions)}/{MAX_CONCURRENT_POSITIONS} | "
        f"High-BTC-Corr: {existing_high_corr_count}/{MAX_HIGH_CORR_POSITIONS}"
        f"{portfolio_alert}"
        f"{stale_line}\n\n"
        + "\n".join(coin_trends_summary)
    )
    return buy_signals, exit_watch_signals, position_updates, summary_msg, supply_zone_alerts, watch_reversal_signals

def build_messages(buy_list, exit_watch_list, position_updates, market_summary,
                    supply_zone_alerts=None, watch_reversal_list=None) -> list:
    blocks = [market_summary]

    # --- Position Updates Section (from positions.json) ---
    if position_updates:
        current = "📋 <b>[Position Updates – สถานะที่ติดตามอยู่]</b>"
        action_icons = {
            "close_sl": "❌ ปิดสถานะเต็มจำนวน (โดน SL ก่อน TP1)",
            "close_tp2": "🏁 ปิด Runner ที่เหลือ (ถึง TP2 เต็มเป้า)",
            "close_runner_sl": "🔚 ปิด Runner ที่เหลือ (โดน Trailing SL)",
            "partial_tp1": "🎯 Partial TP1 — ปิดบางส่วน + เริ่ม Runner",
            "partial_tp1_5": "🎯 Partial TP1.5 — ปิดเพิ่มอีกส่วนหนึ่ง",
            "give_back_warn": "⚠️ Give-Back Warning (ก่อน TP1)",
            "time_stop": "⏳ Time-Stop Warning",
            "update_sl": "🔧 ปรับ SL",
            "info": "ℹ️ อัปเดต",
            "sl_proximity": "⚠️ SL Proximity Warning",
        }
        for upd in position_updates:
            notes_txt = "\n".join(f"  • {n}" for n in upd["notes"])
            msg = (
                f"\n\n🪙 <b>{upd['coin']}</b> | เข้า: ${upd['entry_price']} → ปัจจุบัน: ${upd['price']}\n"
                f"{action_icons.get(upd['action'], upd['action'])}\n"
                f"{notes_txt}"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "📋 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)


    if buy_list:
        # NOTE: each fully-loaded signal block (with funding/OI/BTC.D/macro
        # advisories) is ~900-1100 chars. The 3500-char chunk limit always
        # splits BETWEEN whole signal blocks (never mid-message), so Telegram
        # messages stay readable even with all #3/#4/#5 advisories present.
        current = "🎯 <b>[Crypto Screener 4H – สัญญาณซื้อ]</b>"
        for opt in buy_list:
            vol = "🔊 Volume ยืนยัน" if opt["vol_confirmed"] else "🔇 Volume ต่ำ"
            time_stop_line = f"\n{opt['time_stop_label']}" if opt.get("time_stop_label") else ""
            tp_method_note = " (ATR-Based)" if opt.get("tp_method") == "atr_based" else " (Fixed % Fallback)"
            div_line = f"\n🔀 Divergence: {opt['divergence_label']}" if opt.get("divergence_label") else ""
            wyckoff_line = f"\n{opt['wyckoff_label']}" if opt.get("wyckoff_label") else ""
            er_line      = f" | {opt['er_label']}" if opt.get("er_label") else ""
            timing_line  = f"\n{opt['timing_label']}" if opt.get("timing_label") else ""
            msg = (
                f"\n\n🪙 <b>{opt['coin']}</b> | ราคา: ${opt['price']}\n"
                f"🚨 สัญญาณ: <b>{opt['type']}</b> | RSI 4H: {opt['rsi']}\n"
                f"{vol}\n"
                f"🏆 {opt['score_label']}\n"
                f"📡 MTF: {opt['mtf_label']}\n"
                f"{opt['reversal_label']}"
                f"{div_line}\n"
                f"📐 แนวโน้ม: {opt['trend_label']}{er_line}\n"
                f"🔄 RSI Bounce: {opt['bounce_label']}"
                f"{timing_line}\n"
                f"🔗 On-Chain (1M): {opt['onchain_label']}"
                f"{wyckoff_line}\n"
                f"🔁 Correlation กับ BTC (returns): {opt['corr_btc']}"
                f"{opt['advisory_block']}\n\n"
                f"💼 Position: <code>{opt['pos_size']}</code> | SL ระยะ: {opt['sl_risk_pct']}\n"
                f"❌ Hard SL: <code>{opt['sl']}</code>\n"
                f"🚀 TP1: <code>{opt['tp1']}</code> | TP1.5: <code>{opt['tp1_5']}</code> | TP2: <code>{opt['tp2']}</code>{tp_method_note}\n"
                f"🔥 Dynamic TP (RSI70 est.): <code>{opt['dynamic_tp']}</code>\n"
                f"{opt['partial_tp_note']}"
                f"{time_stop_line}"
                f"{opt['downside_warning']}"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "🎯 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)
    else:
        blocks.append("😴 <i>ตลาดนิ่ง: ไม่มีสัญญาณซื้อในรอบนี้</i>")

    # --- Watch for Reversal Section (RSI Oversold + EMA200 breach, no action) ---
    if watch_reversal_list:
        current = (
            "🔵 <b>[Watch for Reversal – จับตาดู ไม่ใช่สัญญาณซื้อ/ขาย]</b>\n"
            "<i>RSI Oversold มาก แม้ราคาหลุด EMA200 — รอ confirm ทิศทางก่อนตัดสินใจ</i>"
        )
        for opt in watch_reversal_list:
            live_line = f" | Live: ${opt['live_price']}" if opt.get("live_price") else ""
            msg = (
                f"\n\n🪙 <b>{opt['coin']}</b> | ราคา: ${opt['price']} | RSI: {opt['rsi']}{live_line} "
                f"| {opt['atr_multiple']}x ATR\n"
                f"📐 แนวโน้ม: {opt['trend_label']}\n"
                f"📝 {opt['note']}"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "🔵 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)

    # --- Exit Watch Section ---
    if exit_watch_list:
        current = "🚪 <b>[Exit Watch – พิจารณาแบ่งขาย / ปิดสถานะ / เลื่อน SL]</b>\n<i>เรียงตาม Exit Score สูงสุดก่อน | 🔴=RSI Alert, ⚠️=ควรระวัง</i>"
        for opt in sorted(exit_watch_list, key=lambda x: -x["exit_score"]):
            reasons_txt = "\n".join(f"  • {r}" for r in opt["exit_reasons"])
            time_stop_line = f"\n{opt['time_stop_label']}" if opt.get("time_stop_label") else ""
            supply_line = f"\n{opt['supply_zone_label']}" if opt.get("supply_zone_label") else ""
            ext_line = f" | ห่าง EMA200: +{opt['ext_from_ema200_pct']:.1f}%" if opt.get("ext_from_ema200_pct", 0) >= 15 else ""
            live_price_line = f" | Live: ${opt['live_price']}" if opt.get("live_price") else ""
            atr_line = f" | {opt['atr_multiple']}x ATR" if opt.get("atr_multiple") is not None else ""
            # Fix M: special headers for different alert types
            rsi_header = ""
            if opt.get("is_rsi_oversold_override"):
                rsi_header = (
                    f"🔵 <b>RSI Oversold ({opt['rsi']}) — ไม่แนะนำ Panic Sell แม้ราคาหลุด EMA200</b>\n"
                )
            elif opt.get("is_near_ema200_line"):
                rsi_header = f"🟡 <b>ราคาใกล้เส้น EMA200 — เฝ้าระวัง (ยังไม่ชัดเจน)</b>\n"
            elif opt.get("below_ema200_position"):
                rsi_header = f"🔴 <b>ราคาหลุด EMA200 — สถานะ Long เสี่ยงสูง!</b>\n"
            elif opt.get("rsi_hard_alert"):
                rsi_header = f"🔴 <b>RSI {opt['rsi']} — สูงเกินเกณฑ์ ควรแบ่งขายได้เลย</b>\n"
            msg = (
                f"\n\n🪙 <b>{opt['coin']}</b> | ราคา: ${opt['price']} | RSI: {opt['rsi']}{ext_line}{live_price_line}{atr_line}\n"
                f"{rsi_header}"
                f"{opt['exit_label']}\n"
                f"📐 แนวโน้ม: {opt['trend_label']}\n"
                f"สาเหตุ:\n{reasons_txt}"
                f"{supply_line}"
                f"{time_stop_line}"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "🚪 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)

    # --- Standalone Supply Zone / Bearish OB Alert ---
    # แจ้งเตือนแยกสำหรับเหรียญที่ราคาเข้าใกล้ Supply Zone แต่ยังไม่อยู่ใน exit_watch
    # (เช่น เหรียญที่ยังไม่มี position แต่กำลังเข้าใกล้แนวต้านสำคัญ)
    if supply_zone_alerts:
        current = "🔴 <b>[Supply Zone Alert – กำแพงต้านขนาดใหญ่]</b>\n<i>เหรียญที่ราคากำลังเข้าใกล้ Bearish Order Block / Supply Zone ที่มีนัยสำคัญ</i>"
        for alert in supply_zone_alerts:
            msg = (
                f"\n\n🪙 <b>{alert['coin']}</b> | ราคา: ${alert['price']}\n"
                f"{alert['supply_zone_label']}"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "🔴 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)

    return blocks

# ==========================================
# Sanity Check Utility: ATR-Based TP/SL Geometry
# ==========================================
def run_tp_sl_sanity_check() -> None:
    """
    คำนวณ TP1/TP2/SL (ATR-based) ของทุกเหรียญด้วยข้อมูลปัจจุบัน แล้วพิมพ์
    ระยะห่างเป็น % เทียบกับราคา เพื่อเทียบกับค่า fixed-% เดิม (TP_TIERS)
    ว่า geometry ใหม่สมเหตุสมผลหรือไม่ (ไม่แน่นไป/กว้างไปสำหรับแต่ละ tier)

    เรียกด้วย: python3 crypto_screener_v6.py --check-tp
    """
    logger.info("🔍 TP/SL Sanity Check: กำลังดึงข้อมูล 4H ทุกเหรียญ...")
    all_4h = bulk_fetch_4h(WATCHLIST)

    header = f"{'Coin':<7}{'Tier':<7}{'Price':>14}{'ATR':>12}{'ATR%':>7}  " \
             f"{'TP1%':>7}{'TP2%':>7}{'SL%':>7}{'Method':>16}  {'Old TP1%':>9}{'Old TP2%':>9}"
    print(header)
    print("-" * len(header))

    for coin in COINS:
        df = all_4h.get(coin)
        if df is None or len(df) < EMA_LONG + 10:
            print(f"{coin:<7} (insufficient data)")
            continue

        df = calculate_indicators(df)
        row = df.iloc[-1]
        price, atr, adx = row["close"], row["ATR"], row["ADX"]
        tier = COIN_TIER.get(coin, "mid")
        atr_pct = (atr / price) * 100 if price > 0 and not pd.isna(atr) else 0.0
        dyn_mult = get_dynamic_atr_multiplier(tier, adx, atr_pct, df=df)

        ob_info, fvg_info = find_order_blocks(df), find_fair_value_gaps(df)
        ema200 = row["EMA_200"]

        tpsl = calculate_atr_based_tp_sl(price, atr, dyn_mult, tier, ob_info, fvg_info, ema200, adx=adx)
        tp1_pct = ((tpsl["tp1"] - price) / price) * 100
        tp2_pct = ((tpsl["tp2"] - price) / price) * 100
        sl_pct  = ((price - tpsl["sl"]) / price) * 100

        old_tp1_pct = TP_TIERS[tier]["tp1"] * 100
        old_tp2_pct = TP_TIERS[tier]["tp2"] * 100

        print(
            f"{coin:<7}{tier:<7}{price:>14.4f}{atr:>12.4f}{atr_pct:>6.2f}%  "
            f"{tp1_pct:>6.2f}%{tp2_pct:>6.2f}%{sl_pct:>6.2f}%{tpsl['method']:>16}  "
            f"{old_tp1_pct:>8.2f}%{old_tp2_pct:>8.2f}%"
        )

    print("\nหมายเหตุ: 'Old TP1%/TP2%' คือค่า fixed-% เดิมจาก TP_TIERS (ก่อน v5 ATR-based)")
    print("ใช้เทียบดูว่า TP ใหม่ (ATR-based) แน่น/กว้างกว่าเดิมแค่ไหนในสภาวะตลาดปัจจุบัน")


if __name__ == "__main__":
    import sys

    if "--check-tp" in sys.argv:
        run_tp_sl_sanity_check()
        raise SystemExit(0)

    try:
        ip = api_session.get("https://ifconfig.me", timeout=5).text.strip()
        logger.info(f"🌐 Network IP: {ip} (Proxy Active: {bool(PROXIES)})")
    except Exception: pass

    positions = load_positions()
    buy_list, exit_watch_list, position_updates, market_summary, supply_zone_alerts, watch_reversal_list = scan_market(positions)
    send_telegram_messages(build_messages(buy_list, exit_watch_list, position_updates, market_summary, supply_zone_alerts, watch_reversal_list))
    save_positions(positions)
    logger.info("ระบบทำงานสมบูรณ์!")
