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

COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "FLOKI", "SHIB", "EIGEN", "OP", "DOGE", "NEAR",
    "TRX", "AVAX", "SUI",
]

# ==========================================
# Constants & Hyperparameters
# ==========================================
API_RATE_LIMIT_DELAY = 1.0
API_MAX_RETRIES      = 3
BINANCE_LIMIT        = 500
CC_HISTOHOUR_LIMIT   = 2000
CC_HISTODAY_LIMIT    = 500
CG_OHLC_DAYS_4H      = 90
CG_OHLC_DAYS_1D      = 365
CACHE_TTL_SECONDS    = 3600

# Time-based stop: if a setup hasn't progressed within this many 4H bars
# (approx. days), flag it as "stale" in the trend/exit narrative.
TIME_STOP_BARS_4H = 42   # ~7 days on 4H candles

# Position state persistence (JSON file committed back to repo via CI)
POSITIONS_FILE = os.getenv("POSITIONS_FILE", "positions.json")
# Time-based stop for an actual open position: if it hasn't hit TP1 or SL
# within this many hours, surface a "พิจารณาปิด" warning regardless of score.
POSITION_TIME_STOP_HOURS = 7 * 24  # 7 days
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
    "https://api.binance.us",
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api3.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
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
    total=API_MAX_RETRIES,
    backoff_factor=2.0,
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
_cache_onchain:   dict  = {}
_cache_funding:   dict  = {}
_cache_btcd:      dict  = {}
_cache_macro:     dict  = {}
_cache_ts_4h:     float = 0.0
_cache_ts_1d:     float = 0.0
_cache_ts_onchain: float = 0.0
_cache_ts_funding: float = 0.0
_cache_ts_btcd:    float = 0.0
_cache_ts_macro:   float = 0.0

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
    โครงสร้าง positions.json (v5):
    {
      "BTC": {
        "entry_price": 65000.0,
        "entry_time": "2026-06-10T12:00:00+00:00",
        "sl": 63000.0,
        "tp1": 70000.0,
        "tp2": 73000.0,
        "tier": "major",
        "atr_at_entry": 1234.5,
        "tp1_hit": false,
        "partial_closed": false,
        "remaining_size_pct": 1.0,
        "status": "open"
      },
      ...
    }
    """
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
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

def open_position(positions: dict, coin: str, entry_price: float, sl: float, tp1: float, tp2: float, tier: str, atr_at_entry: float) -> None:
    positions[coin] = {
        "entry_price": entry_price,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tier": tier,
        "atr_at_entry": atr_at_entry,
        "tp1_hit": False,
        "partial_closed": False,
        "remaining_size_pct": 1.0,
        "status": "open",
    }

def close_position(positions: dict, coin: str, reason: str) -> None:
    if coin in positions:
        positions[coin]["status"] = f"closed ({reason})"
        positions[coin]["closed_time"] = datetime.now(timezone.utc).isoformat()
        # Drop closed positions from the active file to keep it lean,
        # but keep a short note in the log for traceability.
        logger.info(f"📕 ปิด position {coin}: {reason}")
        del positions[coin]

def update_position_trailing_stop(position: dict, current_price: float, current_atr: float) -> tuple[float, list]:
    """
    ปรับ SL ตาม trailing logic:
    1. ถ้าราคาวิ่งไปแล้วครึ่งทางสู่ TP1 (และยังไม่ถึง TP1) -> เลื่อน SL ไป breakeven (entry price)
    2. ถ้าราคาทะลุ TP1 ไปแล้ว (เข้าสู่ Runner phase หลัง Partial TP) ->
       trail SL ด้วย ATR * RUNNER_TRAIL_ATR_MULTIPLIER ใต้ราคาปัจจุบัน
       (เลื่อนขึ้นได้เท่านั้น ไม่เลื่อนลง — trail แน่นกว่า phase ก่อน TP1)
    คืนค่า (new_sl, change_notes)
    """
    entry, sl, tp1 = position["entry_price"], position["sl"], position["tp1"]
    notes = []
    new_sl = sl

    halfway_to_tp1 = entry + (tp1 - entry) * BREAKEVEN_TRIGGER_PCT

    if position.get("tp1_hit", False):
        # Runner phase: trail tighter (RUNNER_TRAIL_ATR_MULTIPLIER), never move SL down
        if not pd.isna(current_atr) and current_atr > 0:
            trail_sl = current_price - (current_atr * RUNNER_TRAIL_ATR_MULTIPLIER)
            if trail_sl > new_sl:
                new_sl = trail_sl
                notes.append(f"📈 Runner: เลื่อน Trailing SL ขึ้นเป็น {format_price(new_sl)} (ATR x{RUNNER_TRAIL_ATR_MULTIPLIER})")
    elif current_price >= halfway_to_tp1 and new_sl < entry:
        # Move to breakeven once price has covered 50% of the distance to TP1
        new_sl = entry
        notes.append(f"🛡️ ราคาวิ่งเกินครึ่งทางสู่ TP1 — เลื่อน SL ไป Breakeven ({format_price(entry)})")

    return new_sl, notes

def check_position_status(position: dict, current_price: float, current_atr: float) -> dict:
    """
    ตรวจสอบ position ที่เปิดอยู่เทียบกับราคาปัจจุบัน:
    - แตะ SL แล้ว -> ต้องปิด (loss / breakeven / runner stop-out)
    - แตะ TP1 ครั้งแรก (ยังไม่ partial close) -> สั่ง Partial TP, เริ่ม Runner phase
    - แตะ TP2 (หรือ SL หลัง partial) -> ปิด runner ที่เหลือทั้งหมด
    - อัปเดต trailing SL ถ้ายังไม่ถึงเงื่อนไขปิด
    - ตรวจ time-based stop ถ้าเปิดมานานเกินกำหนดและยังไม่ไป TP1/SL
    คืน dict: {"action": ..., "new_sl": float, "notes": [...]}

    Possible actions:
      "close_sl"      -> full close, SL hit before TP1 (full loss/breakeven)
      "partial_tp1"   -> TP1 hit for the first time: close PARTIAL_TP1_CLOSE_PCT,
                          remainder becomes the "runner"
      "close_tp2"     -> runner hit TP2: close remainder fully
      "close_runner_sl" -> runner trailing stop hit after TP1: close remainder
      "update"        -> trailing SL adjusted, position stays open
      "time_stop"     -> still open too long without reaching TP1
    """
    result = {"action": "update", "new_sl": position["sl"], "notes": []}

    sl, tp1, tp2 = position["sl"], position["tp1"], position["tp2"]
    tp1_hit = position.get("tp1_hit", False)

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
        new_sl, trail_notes = update_position_trailing_stop(position, current_price, current_atr)
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

        # Runner trailing stop hit -> close remainder
        if current_price <= sl:
            result["action"] = "close_runner_sl"
            result["notes"].append(f"🔚 Runner โดน Trailing SL ({format_price(sl)}) — ปิดสถานะส่วนที่เหลือ")
            return result

    # --- Otherwise: update trailing SL (no closure this cycle) ---
    new_sl, trail_notes = update_position_trailing_stop(position, current_price, current_atr)
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

    return result


def _parse_binance_klines(data: list, coin: str, tf: str) -> pd.DataFrame | None:
    try:
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "volumeto", "trades", "taker_base", "taker_quote", "ignore",
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close", "volumeto"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close", "volumeto"]].dropna()
        logger.info(f"✅ {coin} {tf} Binance สำเร็จ ({len(df)} แท่ง)")
        return df
    except Exception as e:
        logger.warning(f"{coin} {tf} Binance parse error: {e}")
        return None

def _fetch_from_binance(symbol: str, interval: str, limit: int, coin: str, tf: str) -> pd.DataFrame | None:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}/api/v3/klines"
        try:
            resp = api_session.get(url, params=params, timeout=15)
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
        logger.info(f"✅ {coin} 4H CC สำเร็จ ({len(df_4h)} แท่ง 4H)")
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

def _fetch_4h_from_coingecko(coin: str) -> pd.DataFrame | None:
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
            time.sleep(15)
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

        logger.info(f"✅ {coin} 4H CoinGecko สำเร็จ ({len(df)} แท่ง)")
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
            time.sleep(15)
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

def get_historical_data(coin: str) -> pd.DataFrame | None:
    symbol   = f"{coin}USDT"
    min_bars = EMA_LONG + 10

    df = _fetch_from_binance(symbol, "4h", BINANCE_LIMIT, coin, "4H")
    if df is not None and len(df) >= min_bars: return df

    df = _fetch_4h_from_cryptocompare(coin)
    if df is not None and len(df) >= min_bars: return df

    return _fetch_4h_from_coingecko(coin)

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
    """
    symbol = f"{coin}USDT"
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
            return df
        except Exception as e:
            logger.warning(f"{coin} OI hist → {e}")
    return None

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
            time.sleep(15)
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
    url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    params = {"from": today, "to": today, "apikey": FMP_API_KEY}
    try:
        resp = api_session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Macro calendar fetch → HTTP {resp.status_code}")
            _cache_macro, _cache_ts_macro = result, now
            return result["events"]

        data = resp.json()
        if not isinstance(data, list):
            _cache_macro, _cache_ts_macro = result, now
            return result["events"]

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
        logger.warning(f"Macro calendar fetch error: {e}")

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
    close, high, low = df["close"], df["high"], df["low"]
    df["EMA_50"]  = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LONG,  adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rs    = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean() / loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean().replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

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
    result = {"has_bullish_engulfing": False, "has_hammer": False, "has_morning_star": False, "reversal_strength": 0, "reversal_label": "⬜ ไม่มี Candle ยืนยัน"}
    if len(df) < 3: return result
    c0, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    body0 = abs(c0["close"] - c0["open"])

    if c1["close"] < c1["open"] and c0["close"] > c0["open"] and c0["open"] <= c1["close"] and c0["close"] >= c1["open"]:
        result["has_bullish_engulfing"], result["reversal_strength"] = True, result["reversal_strength"] + 2
    if body0 > 0 and (min(c0["open"], c0["close"]) - c0["low"]) >= 2 * body0 and (c0["high"] - max(c0["open"], c0["close"])) <= 0.3 * body0 and c0["close"] > c0["open"]:
        result["has_hammer"], result["reversal_strength"] = True, result["reversal_strength"] + 1
    if c2["close"] < c2["open"] and abs(c2["close"] - c2["open"]) > 0 and abs(c1["close"] - c1["open"]) < abs(c2["close"] - c2["open"]) * 0.4 and c0["close"] > c0["open"] and c0["close"] > (c2["open"] + c2["close"]) / 2:
        result["has_morning_star"], result["reversal_strength"] = True, result["reversal_strength"] + 2

    if result["reversal_strength"] >= 3: result["reversal_label"] = "🕯️ <b>Candle ยืนยันแข็งแกร่ง</b>"
    elif result["reversal_strength"] >= 1: result["reversal_label"] = "🕯️ Candle ยืนยันปานกลาง"
    return result

def confirm_bearish_reversal_candle(df: pd.DataFrame) -> dict:
    """ตรวจ candle pattern กลับตัวขาลง (สำหรับ exit/short watch): Bearish Engulfing, Shooting Star, Evening Star"""
    result = {"has_bearish_engulfing": False, "has_shooting_star": False, "has_evening_star": False, "bearish_strength": 0, "bearish_label": "⬜ ไม่มี Candle เตือนกลับตัว"}
    if len(df) < 3: return result
    c0, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    body0 = abs(c0["close"] - c0["open"])

    # Bearish Engulfing: previous green candle fully engulfed by a red candle
    if c1["close"] > c1["open"] and c0["close"] < c0["open"] and c0["open"] >= c1["close"] and c0["close"] <= c1["open"]:
        result["has_bearish_engulfing"], result["bearish_strength"] = True, result["bearish_strength"] + 2
    # Shooting Star: small body near the low, long upper wick, after an up move
    if body0 > 0 and (c0["high"] - max(c0["open"], c0["close"])) >= 2 * body0 and (min(c0["open"], c0["close"]) - c0["low"]) <= 0.3 * body0 and c0["close"] < c0["open"]:
        result["has_shooting_star"], result["bearish_strength"] = True, result["bearish_strength"] + 1
    # Evening Star: big green, small body, big red closing below midpoint of first candle
    if c2["close"] > c2["open"] and abs(c2["close"] - c2["open"]) > 0 and abs(c1["close"] - c1["open"]) < abs(c2["close"] - c2["open"]) * 0.4 and c0["close"] < c0["open"] and c0["close"] < (c2["open"] + c2["close"]) / 2:
        result["has_evening_star"], result["bearish_strength"] = True, result["bearish_strength"] + 2

    if result["bearish_strength"] >= 3: result["bearish_label"] = "🕯️ <b>Candle เตือนกลับตัวขาลงแข็งแกร่ง</b>"
    elif result["bearish_strength"] >= 1: result["bearish_label"] = "🕯️ Candle เตือนกลับตัวขาลงปานกลาง"
    return result

def get_dynamic_atr_multiplier(tier: str, adx: float, atr_pct: float) -> float:
    base = {"major": 2.0, "mid": 2.5, "small": 3.0}.get(tier, 2.5)
    if adx > 25: base *= 1.20
    if atr_pct > 4.0: base *= 1.15
    if adx < 15: base *= 0.85
    return round(min(base, 4.0), 2)

def get_correlation_adjusted_position(portfolio: float, risk_pct: float, sl_distance_pct: float, corr_btc: float, active_signals_count: int) -> float:
    base_risk = portfolio * (risk_pct / 100)
    if corr_btc > 0.85: base_risk *= 0.70
    elif corr_btc > 0.70: base_risk *= 0.85
    max_total_risk = portfolio * (MAX_TOTAL_RISK_PCT / 100)
    if active_signals_count > 0:
        base_risk = min(base_risk, max_total_risk / (active_signals_count + 1))
    return min(base_risk / max(sl_distance_pct, 0.01), portfolio * 0.25)

def get_mtf_rsi_alignment(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    result = {"aligned_oversold": False, "aligned_overbought": False, "rsi_4h": None, "rsi_1d": None, "mtf_label": "", "confluence_score": 0}
    if df_4h is None or df_1d is None: return result
    rsi_4h = df_4h["RSI"].iloc[-1] if "RSI" in df_4h.columns else 50.0
    rs = df_1d["close"].diff().clip(lower=0).ewm(com=13, adjust=False).mean() / (-df_1d["close"].diff().clip(upper=0)).ewm(com=13, adjust=False).mean().replace(0, np.nan)
    rsi_1d = float((100 - 100 / (1 + rs)).iloc[-1]) if len(df_1d) > 0 else 50.0

    result.update({"rsi_4h": round(rsi_4h, 2), "rsi_1d": round(rsi_1d, 2)})
    score = (2 if rsi_4h <= RSI_OVERSOLD else 1 if rsi_4h <= 45 else 0) + (2 if rsi_1d <= RSI_OVERSOLD else 1 if rsi_1d <= 45 else 0)
    result["confluence_score"] = score

    if score >= 4: result.update({"aligned_oversold": True, "mtf_label": f"💎 MTF RSI Oversold (4H:{rsi_4h:.1f}/1D:{rsi_1d:.1f})"})
    elif score >= 2: result["mtf_label"] = f"🟡 MTF RSI อ่อนแรงปานกลาง (4H:{rsi_4h:.1f}/1D:{rsi_1d:.1f})"
    elif rsi_4h >= RSI_OVERBOUGHT and rsi_1d >= RSI_OVERBOUGHT: result.update({"aligned_overbought": True, "mtf_label": f"🔴 MTF RSI Overbought (4H:{rsi_4h:.1f}/1D:{rsi_1d:.1f})"})
    return result

def calculate_signal_score(rsi, bounce_info, candle_info, vol_confirmed, in_fibo_zone, in_ob_zone, in_fvg_zone, weekly_ctx, mtf_info, adx, is_divergence, trend_info, onchain_info, funding_info=None, oi_info=None, btcd_filter=None) -> tuple[int, str]:
    # 1. Technical Score
    score = (25 if rsi <= RSI_OVERSOLD else 15 if rsi <= 40 else 8 if rsi <= 50 else 0)
    score += {"strong": 20, "moderate": 12, "weak": 5, "none": 0}.get(bounce_info.get("quality", "none"), 0)
    score += min(candle_info.get("reversal_strength", 0) * 7, 15)
    if in_fibo_zone: score += 8
    if in_ob_zone: score += 7
    if in_fvg_zone: score += 5
    if is_divergence: score += 10
    if vol_confirmed: score += 10
    score += min(mtf_info.get("confluence_score", 0) * 2, 10)
    if weekly_ctx.get("weekly_bullish_div"): score += 5
    elif weekly_ctx.get("rsi_weekly") and weekly_ctx["rsi_weekly"] <= 35: score += 3
    if adx > 25: score += 5
    elif adx < 15: score -= 5
    ts = trend_info.get("trend_strength", "sideways")
    if ts == "strong_up": score += 5
    elif ts == "moderate_up": score += 2

    # 2. On-Chain Modifier Score (เพิ่มคะแนนหากบนเชนโต หักคะแนนหากหดตัว)
    if onchain_info.get("has_data"):
        trend = onchain_info.get("active_addresses_trend", 0)
        if trend >= 15:
            score += 15      # โบนัส: พื้นฐานแข็งแกร่งมาก
        elif trend >= 5:
            score += 5       # โบนัส: โตปกติ
        elif trend <= -10:
            score -= 15      # จุดอันตราย: ราคาลงหรือสวิง แต่คนหนีออกจากเชน

    # 3. [#3] Funding Rate Modifier (ลดคะแนนถ้า funding สูง = crowded long)
    if funding_info and funding_info.get("has_data"):
        if funding_info.get("warn_long"):
            score -= 5

    # 4. [#3] Open Interest Modifier (ลดคะแนนถ้า rally ดูเหมือน short-covering)
    if oi_info and oi_info.get("has_data") and oi_info.get("weak_conviction"):
        score -= 8

    # 5. [#4] BTC Dominance Modifier (alt_unfriendly -> หักคะแนน, alt_friendly -> บวก)
    if btcd_filter and btcd_filter.get("score_delta"):
        score += btcd_filter["score_delta"]

    score = max(0, min(100, score))
    grade = "🔥 A+" if score >= 70 else "✅ A" if score >= 55 else "🟡 B" if score >= MINIMUM_SIGNAL_SCORE else "⬜ C"
    return score, f"{grade} | Score: {score}/100"

def calculate_exit_score(rsi, bearish_candle_info, vol_confirmed, mtf_info, adx, is_bear_div, trend_info, onchain_info, price, ema50, ema200, near_overbought_target: bool) -> tuple[int, str, list]:
    """
    คำนวณคะแนนเตือน 'ควรพิจารณาปิดสถานะ Long' (0-100)
    ใช้สำหรับ coin ที่กำลังอยู่ในแนวโน้มขาขึ้น (price > EMA200) เพื่อเตือนสัญญาณอ่อนแรง/กลับตัว
    คะแนนสูง = สัญญาณเตือนแรง ควรพิจารณา TP/ลดสถานะ
    """
    reasons = []
    score = 0

    # RSI Overbought
    if rsi >= RSI_OVERBOUGHT:
        score += 25
        reasons.append(f"RSI Overbought ({rsi:.1f})")
    elif rsi >= 60:
        score += 10
        reasons.append(f"RSI สูง ({rsi:.1f})")

    # Bearish reversal candle
    bstr = bearish_candle_info.get("bearish_strength", 0)
    if bstr > 0:
        score += min(bstr * 8, 20)
        reasons.append(bearish_candle_info["bearish_label"].replace("🕯️ ", "").replace("<b>", "").replace("</b>", ""))

    # Bearish divergence (price higher high, RSI lower high)
    if is_bear_div:
        score += 20
        reasons.append("RSI Bearish Divergence")

    # MTF overbought alignment
    if mtf_info.get("aligned_overbought"):
        score += 15
        reasons.append("MTF RSI Overbought (4H+1D)")

    # Volume not confirming the move (potential exhaustion)
    if not vol_confirmed:
        score += 8
        reasons.append("Volume ไม่ยืนยัน (อาจหมดแรงซื้อ)")

    # ADX weakening trend strength
    if adx < 15:
        score += 7
        reasons.append("ADX ต่ำ (เทรนด์อ่อนแรง)")

    # Trend continuity flipping down
    ts = trend_info.get("trend_strength", "sideways")
    if ts in ("strong_down", "moderate_down"):
        score += 20
        reasons.append(f"แนวโน้มกลับเป็นขาลง ({trend_info.get('trend_label','')})")
    elif ts == "sideways":
        score += 5
        reasons.append("แนวโน้มเริ่ม sideway")

    # Price falling back below EMA50 while still above EMA200 (momentum loss)
    if price < ema50 and price > ema200:
        score += 15
        reasons.append("ราคาหลุด EMA50 (โมเมนตัมอ่อนลง)")
    elif price < ema200:
        score += 25
        reasons.append("ราคาหลุด EMA200 (เปลี่ยนแนวโน้มหลัก)")

    # On-chain deterioration
    if onchain_info.get("has_data") and onchain_info.get("active_addresses_trend", 0) <= -10:
        score += 10
        reasons.append("On-Chain หดตัวหนัก")

    # Near a statistically-estimated overbought price target
    if near_overbought_target:
        score += 10
        reasons.append("ราคาเข้าใกล้เป้า RSI Overbought ที่ประเมินไว้")

    score = max(0, min(100, score))
    if score >= 70: label = f"🚨 <b>สัญญาณเตือนแรง</b> | Exit Score: {score}/100"
    elif score >= MINIMUM_EXIT_SCORE: label = f"⚠️ ควรพิจารณาลดสถานะ/เลื่อน SL | Exit Score: {score}/100"
    else: label = f"🟢 ยังถือต่อได้ | Exit Score: {score}/100"
    return score, label, reasons

def analyze_weekly_context(df_1d: pd.DataFrame) -> dict:
    result = {"rsi_weekly": None, "weekly_bullish_div": False, "weekly_status_label": "↔️ ไม่พบข้อมูล 1W", "fibo_618": None, "fibo_786": None, "fibo_886": None, "liquidity_pool": None, "psycho_support": None}
    if df_1d is None or len(df_1d) < 35: return result
    try:
        df_w = df_1d.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if len(df_w) < 15: return result
        w_max, w_min = df_w.iloc[-52:]["high"].max() if len(df_w) >= 52 else df_w["high"].max(), df_w.iloc[-52:]["low"].min() if len(df_w) >= 52 else df_w["low"].min()
        result.update({"fibo_618": w_max - (0.618 * (w_max - w_min)), "fibo_786": w_max - (0.786 * (w_max - w_min)), "fibo_886": w_max - (0.886 * (w_max - w_min)), "liquidity_pool": w_min})

        curr_price = df_w["close"].iloc[-1]
        if curr_price > 0:
            mag = 10 ** math.floor(math.log10(curr_price))
            result["psycho_support"] = math.floor(curr_price / (mag if curr_price >= mag * 2 else mag / 2)) * (mag if curr_price >= mag * 2 else mag / 2)

        rs = df_w["close"].diff().clip(lower=0).ewm(com=13, adjust=False).mean() / (-df_w["close"].diff().clip(upper=0)).ewm(com=13, adjust=False).mean().replace(0, np.nan)
        df_w["RSI"] = (100 - (100 / (1 + rs))).fillna(100)
        result["rsi_weekly"] = round(df_w["RSI"].iloc[-1], 2)

        prev = df_w.iloc[-16:-3]
        if len(prev) > 0 and prev["RSI"].iloc[prev["low"].argmin()] <= 45 and df_w["low"].iloc[-1] < prev["low"].min() and df_w["RSI"].iloc[-1] > prev["RSI"].iloc[prev["low"].argmin()]:
            result["weekly_bullish_div"] = True

        if result["weekly_bullish_div"]: result["weekly_status_label"] = f"👑 <b>Weekly Bullish Div!</b> (RSI: {result['rsi_weekly']})"
        elif result["rsi_weekly"] <= RSI_OVERSOLD: result["weekly_status_label"] = f"🔥 <b>1W Oversold รุนแรง ({result['rsi_weekly']})</b>"
    except Exception: pass
    return result

def analyze_trend_continuity(df: pd.DataFrame) -> dict:
    result = {"trend_strength": "sideways", "trend_label": "↔️ ไม่ชัดเจน"}
    if len(df) < 7: return result
    ema50_now, ema50_prev = df["EMA_50"].iloc[-1], df["EMA_50"].iloc[-6]
    ema200_now, ema200_prev = df["EMA_200"].iloc[-1], df["EMA_200"].iloc[-6]
    slope50 = ((ema50_now - ema50_prev) / ema50_prev) * 100 if ema50_prev != 0 else 0
    slope200 = ((ema200_now - ema200_prev) / ema200_prev) * 100 if ema200_prev != 0 else 0

    diffs = df["close"].iloc[-20:].diff().iloc[1:].values[::-1]
    up_streak = next((i for i, v in enumerate(diffs) if v <= 0), len(diffs))
    dn_streak = next((i for i, v in enumerate(diffs) if v >= 0), len(diffs))

    if slope50 > 0 and slope200 > 0 and up_streak >= 3: result.update({"trend_strength": "strong_up", "trend_label": f"🚀 ขาขึ้นแข็งแกร่ง ({up_streak} แท่ง)"})
    elif slope50 > 0 and up_streak >= 1: result.update({"trend_strength": "moderate_up", "trend_label": f"📈 ขาขึ้นปานกลาง ({up_streak} แท่ง)"})
    elif slope50 <= 0 and slope200 <= 0 and dn_streak >= 3: result.update({"trend_strength": "strong_down", "trend_label": f"🔻 ขาลงแข็งแกร่ง ({dn_streak} แท่ง)"})
    elif slope50 <= 0 and dn_streak >= 1: result.update({"trend_strength": "moderate_down", "trend_label": f"📉 ขาลงปานกลาง ({dn_streak} แท่ง)"})
    return result

def analyze_rsi_bounce(df: pd.DataFrame) -> dict:
    result = {"quality": "none", "quality_label": "⬜ ไม่มีสัญญาณดีดกลับ", "entry_timing": ""}
    if len(df) < 20: return result
    rsi_series, rsi_curr = df["RSI"].iloc[-16:-1], df["RSI"].iloc[-1]
    rsi_min = rsi_series.min()
    if rsi_min > RSI_OVERSOLD: return result

    rsi_rise = rsi_curr - rsi_min
    consec = sum(1 for v in df["RSI"].iloc[-5:].diff().iloc[1:].values[::-1] if v > 0)
    score = (1 if rsi_rise >= 3.0 else 0) + (1 if consec >= 2 else 0) + (1 if rsi_curr < 50 or (df["RSI"].iloc[-5:] >= 45).any() else 0)

    if score == 3: result.update({"quality": "strong", "quality_label": f"✅ ดีดกลับแข็งแกร่ง (+{rsi_rise:.1f})"})
    elif score == 2: result.update({"quality": "moderate", "quality_label": f"🟡 ดีดกลับปานกลาง (+{rsi_rise:.1f})"})
    return result

def find_order_blocks(df: pd.DataFrame) -> dict:
    ob = {"has_bullish_ob": False, "bullish_ob_price": None, "has_bearish_ob": False, "bearish_ob_price": None}
    if len(df) < 25: return ob
    avg_body = (df["close"] - df["open"]).abs().rolling(20).mean().iloc[-1]
    curr_body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    recent_high, recent_low = df.iloc[-21:-3]["high"].max(), df.iloc[-21:-3]["low"].min()

    if df["close"].iloc[-1] > recent_high and curr_body > avg_body * 1.5:
        for i in range(2, 15):
            if df["close"].iloc[-i] < df["open"].iloc[-i] and not (df["low"].iloc[-i+1:] < df["low"].iloc[-i]).any():
                ob.update({"has_bullish_ob": True, "bullish_ob_price": df["low"].iloc[-i]})
                break
    return ob

def find_fair_value_gaps(df: pd.DataFrame) -> dict:
    fvg = {"has_fvg_support": False, "fvg_top": None, "fvg_bottom": None}
    if len(df) < 4: return fvg
    for i in range(len(df) - 1, 2, -1):
        if df["low"].iloc[i] > df["high"].iloc[i-2] and df["close"].iloc[i-1] > df["open"].iloc[i-1]:
            if ((df["low"].iloc[i] - df["high"].iloc[i-2]) / df["high"].iloc[i-2]) * 100 >= 0.2 and df["close"].iloc[-1] > df["high"].iloc[i-2]:
                fvg.update({"has_fvg_support": True, "fvg_top": df["low"].iloc[i], "fvg_bottom": df["high"].iloc[i-2]})
                break
    return fvg

def check_bullish_divergence(df: pd.DataFrame) -> bool:
    if len(df) < 17: return False
    prev = df.iloc[-16:-3]
    if len(prev) == 0: return False
    min_idx = prev["low"].argmin()
    return prev["RSI"].iloc[min_idx] <= 45 and df["low"].iloc[-1] < prev["low"].iloc[min_idx] and df["RSI"].iloc[-1] > prev["RSI"].iloc[min_idx]

def check_bearish_divergence(df: pd.DataFrame) -> bool:
    """ตรวจหา Bearish Divergence: ราคาทำ Higher High ใหม่ แต่ RSI ทำ Lower High (โมเมนตัมอ่อนแรง)"""
    if len(df) < 17: return False
    prev = df.iloc[-16:-3]
    if len(prev) == 0: return False
    max_idx = prev["high"].argmax()
    return prev["RSI"].iloc[max_idx] >= RSI_BEAR_DIV_MIN and df["high"].iloc[-1] > prev["high"].iloc[max_idx] and df["RSI"].iloc[-1] < prev["RSI"].iloc[max_idx]

def is_volume_confirmed(row: pd.Series) -> bool:
    if row.get("volumeto", 0) == 0.0:
        return True
    return not pd.isna(row.get("VOL_MA20")) and row["VOL_MA20"] > 0 and row["volumeto"] > row["VOL_MA20"]

def format_price(price: float) -> str:
    if price is None or pd.isna(price): return "N/A"
    return f"{price:.8f}" if price < 0.0001 else f"{price:.6f}" if price < 0.001 else f"{price:.4f}" if price < 1 else f"{price:.2f}"

def estimate_price_for_target_rsi(df: pd.DataFrame, target_rsi=70.0) -> float | None:
    if len(df) < 15 or target_rsi >= 100.0: return None
    try:
        delta = df["close"].diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean().iloc[-1]
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean().iloc[-1]
        curr_price = df["close"].iloc[-1]
        target_rs = target_rsi / (100.0 - target_rsi)
        next_avg_loss = (avg_loss * 13) / 14
        return curr_price + max(0, (target_rs * next_avg_loss * 14) - (avg_gain * 13))
    except Exception: return None

def check_time_stop(df: pd.DataFrame, lookback_bars: int = TIME_STOP_BARS_4H) -> dict:
    """
    ตรวจสอบว่าราคาปัจจุบัน 'แกว่งตัวในกรอบแคบ' (sideways) มานานเกินไปหรือไม่
    เทียบ range ของ N แท่งล่าสุด กับ ATR ปัจจุบัน — ถ้าราคาไม่ไปไหนเลยเทียบความผันผวนปกติ
    ถือว่าตลาด 'อืด' และควรพิจารณา time-based exit เพื่อลด opportunity cost
    """
    result = {"is_stale": False, "time_stop_label": ""}
    if len(df) < lookback_bars + 1 or "ATR" not in df.columns:
        return result
    recent = df.iloc[-lookback_bars:]
    price_range = recent["high"].max() - recent["low"].min()
    atr = df["ATR"].iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return result
    # ถ้า range ของ N แท่งล่าสุด แคบกว่า ATR ปกติ 3 เท่า แสดงว่าตลาดไม่ไปไหนนานเกินคาด
    if price_range < atr * 3:
        days_approx = lookback_bars * 4 / 24
        result["is_stale"] = True
        result["time_stop_label"] = f"⏳ ราคาแกว่ง sideway มา ~{days_approx:.0f} วัน โดยไม่ไปไหน — พิจารณา Time-Stop หากยังไม่ถึง TP"
    return result

# ==========================================
# [#1] ATR-Based TP/SL Calculation
# ==========================================
def calculate_atr_based_tp_sl(price: float, atr: float, dyn_mult: float, tier: str,
                               ob_info: dict, fvg_info: dict, ema200: float) -> dict:
    """
    คำนวณ TP1/TP2 จาก ATR (คูณด้วย ATR_TPx_MULTIPLIER) แทนเปอร์เซนต์ fixed
    SL ยังอ้างอิงโครงสร้างราคา (OB/FVG/EMA200) เหมือนเดิม แต่บังคับระยะห่างขั้นต่ำ
    จาก ATR และ MIN_SL_DISTANCE_PCT เพื่อไม่ให้ SL แน่นเกินไปจน noise ปกติโดนเขี่ยออก

    Fallback: ถ้า ATR เป็น NaN/0 ใช้เปอร์เซนต์เดิมจาก TP_TIERS
    """
    if pd.isna(atr) or atr <= 0:
        # Fallback to fixed percentage (legacy behavior)
        tp1_pct, tp2_pct, sl_buf = TP_TIERS[tier]["tp1"], TP_TIERS[tier]["tp2"], TP_TIERS[tier]["sl_buffer"]
        tp1_val, tp2_val = price * (1 + tp1_pct), price * (1 + tp2_pct)
        sl_ref = ob_info["bullish_ob_price"] if ob_info["has_bullish_ob"] else fvg_info["fvg_bottom"] if fvg_info["has_fvg_support"] else ema200
        sl_val = sl_ref * (1 - sl_buf) if price > sl_ref else price * (1 - sl_buf)
        return {"tp1": tp1_val, "tp2": tp2_val, "sl": sl_val, "method": "fixed_pct_fallback"}

    # ATR-based TP, scaled by the same dynamic multiplier used for trailing/exit logic
    tp1_val = price + (atr * ATR_TP1_MULTIPLIER * dyn_mult / 2.0)
    tp2_val = price + (atr * ATR_TP2_MULTIPLIER * dyn_mult / 2.0)

    # SL: structure-based reference (OB/FVG/EMA200), but never closer than
    # max(ATR * 1, price * MIN_SL_DISTANCE_PCT) to avoid getting stopped by noise
    sl_buf = TP_TIERS[tier]["sl_buffer"]
    sl_ref = ob_info["bullish_ob_price"] if ob_info["has_bullish_ob"] else fvg_info["fvg_bottom"] if fvg_info["has_fvg_support"] else ema200
    structure_sl = sl_ref * (1 - sl_buf) if price > sl_ref else price * (1 - sl_buf)

    min_sl_distance = max(atr * 1.0, price * MIN_SL_DISTANCE_PCT)
    min_allowed_sl = price - min_sl_distance
    sl_val = min(structure_sl, min_allowed_sl)  # whichever is FURTHER from price (lower)

    return {"tp1": tp1_val, "tp2": tp2_val, "sl": sl_val, "method": "atr_based"}

# ==========================================
# Market Scanner
# ==========================================
def scan_market(positions: dict):
    buy_signals, exit_watch_signals, coin_trends_summary, position_updates = [], [], [], []
    bullish_coins, bearish_coins = 0, 0
    active_signal_count = len(positions)  # account for already-open risk

    logger.info("Phase 1: Bulk fetch 4H (A+B+C)...")
    all_4h = bulk_fetch_4h(COINS)
    logger.info("Phase 2: Bulk fetch 1D (A+B+C)...")
    all_1d = bulk_fetch_1d(COINS)
    logger.info("Phase 3: Bulk fetch On-chain (1M)...")
    all_onchain = bulk_fetch_onchain(COINS)
    logger.info("Phase 4: Bulk fetch Funding Rate (Binance Futures)...")
    all_funding = bulk_fetch_funding(COINS)
    logger.info("Phase 5: Fetch BTC Dominance (CoinGecko Global)...")
    btcd_info = get_btc_dominance_info()
    logger.info("Phase 6: Fetch Macro Economic Calendar (FMP)...")
    macro_info = get_macro_filter_info()

    btc_df = all_4h.get("BTC")
    market_regime = "Unknown ⚪"
    if btc_df is not None:
        btc_df = calculate_indicators(btc_df)
        market_regime = "Bull Market 🟢" if btc_df["close"].iloc[-1] > btc_df["EMA_200"].iloc[-1] else "Bear Market 🔴"

    # Pre-calculate BTC returns series for correlation (use % change, not raw price)
    btc_returns = None
    if btc_df is not None:
        btc_returns = btc_df["close"].pct_change().dropna()

    for coin in COINS:
        df, df_daily = all_4h.get(coin), all_1d.get(coin)
        onchain_info = all_onchain.get(coin, {"has_data": False, "onchain_label": "⚪ N/A"})
        funding_info = get_funding_filter_info(coin, all_funding)
        btcd_filter = get_btc_dominance_filter_for_coin(coin, btcd_info)

        if df is None or len(df) < EMA_LONG + 10: continue

        weekly_ctx = analyze_weekly_context(df_daily)
        df = calculate_indicators(df)
        row = df.iloc[-1]

        price, rsi, ema50, ema200 = row["close"], row["RSI"], row["EMA_50"], row["EMA_200"]
        atr, adx, vol_confirmed = row["ATR"], row["ADX"], is_volume_confirmed(row)

        # --- Monitor existing open position for this coin (if any) ---
        if coin in positions:
            pos = positions[coin]
            status = check_position_status(pos, price, atr)
            if status["action"] in ("close_sl", "close_tp2", "close_runner_sl", "time_stop"):
                position_updates.append({
                    "coin": coin, "price": format_price(price),
                    "action": status["action"], "notes": status["notes"],
                    "entry_price": format_price(pos["entry_price"]),
                })
                if status["action"] in ("close_sl", "close_tp2", "close_runner_sl"):
                    close_position(positions, coin, status["action"])
                # time_stop: leave position open but surface the warning;
                # user decides manually whether to close.
            elif status["action"] == "partial_tp1":
                position_updates.append({
                    "coin": coin, "price": format_price(price),
                    "action": "partial_tp1", "notes": status["notes"],
                    "entry_price": format_price(pos["entry_price"]),
                })
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
        dyn_mult = get_dynamic_atr_multiplier(tier, adx, atr_pct)

        trend_info, bounce_info = analyze_trend_continuity(df), analyze_rsi_bounce(df)
        ob_info, fvg_info = find_order_blocks(df), find_fair_value_gaps(df)
        candle_info, mtf_info = confirm_reversal_candle(df), get_mtf_rsi_alignment(df, df_daily)
        bearish_candle_info = confirm_bearish_reversal_candle(df)
        is_div = check_bullish_divergence(df)
        is_bear_div = check_bearish_divergence(df)
        time_stop_info = check_time_stop(df)

        in_fibo = weekly_ctx["fibo_618"] is not None and price <= weekly_ctx["fibo_618"] * 1.02
        in_ob = ob_info["has_bullish_ob"] and price <= ob_info["bullish_ob_price"] * 1.03
        in_fvg = fvg_info["has_fvg_support"] and fvg_info["fvg_bottom"] * 0.99 <= price <= fvg_info["fvg_top"]

        # --- Correlation to BTC, calculated on RETURNS not raw price ---
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

        signal_type = ""
        bias_warning = ""

        if price > ema200:
            bullish_coins += 1
            coin_trends_summary.append(f"• {coin}: 🟢 ขาขึ้น | RSI: {rsi:.1f} | {trend_info['trend_label']}")

            # --- Long entry signals (existing logic) ---
            if (in_fibo or in_ob or in_fvg) and price > (ema50 * 0.98) and rsi <= 55:
                if bounce_info["quality"] in ["strong", "moderate"]: signal_type = "Institution Dip & Rebound 📉"
            if is_div and not signal_type: signal_type = "Confluence Bullish Divergence 📈"

            # --- Exit watch (only meaningful for coins in an uptrend, i.e. likely held as long) ---
            near_ob_target = False
            est_target = estimate_price_for_target_rsi(df)
            if est_target is not None and price > 0:
                near_ob_target = (est_target - price) / price <= 0.02  # within 2% of estimated RSI-70 price

            exit_score, exit_label, exit_reasons = calculate_exit_score(
                rsi, bearish_candle_info, vol_confirmed, mtf_info, adx,
                is_bear_div, trend_info, onchain_info, price, ema50, ema200, near_ob_target
            )
            if exit_score >= MINIMUM_EXIT_SCORE:
                exit_watch_signals.append({
                    "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                    "exit_score": exit_score, "exit_label": exit_label,
                    "exit_reasons": exit_reasons,
                    "trend_label": trend_info["trend_label"],
                    "time_stop_label": time_stop_info["time_stop_label"],
                    "bearish_label": bearish_candle_info["bearish_label"],
                })

        else:
            bearish_coins += 1
            # --- Downside bias warning labels (no short signals, per design choice) ---
            ts = trend_info.get("trend_strength", "sideways")
            if ts == "strong_down":
                bias_warning = "⚠️ <b>เทรนด์ขาลงแข็งแกร่ง — หลีกเลี่ยงการเปิด Long ใหม่</b>"
            elif ts == "moderate_down":
                bias_warning = "⚠️ เทรนด์ขาลง — รอสัญญาณกลับตัวที่ชัดเจนก่อนเข้า Long"
            else:
                bias_warning = "🔻 ราคาต่ำกว่า EMA200 — ระมัดระวัง"

            coin_trends_summary.append(f"• {coin}: 🔴 ขาลง | RSI: {rsi:.1f} | {trend_info['trend_label']} | {bias_warning}")

            if weekly_ctx.get("fibo_786") and price <= weekly_ctx["fibo_786"] * 1.02:
                if is_div: signal_type = "🚨 DEEP REVERSAL + Bullish Div 🐳"
                elif bounce_info["quality"] == "strong": signal_type = "🛡️ Deep Support Strong Bounce 📉"

        if signal_type and coin not in positions:

            # [#5] Macro News Filter: block ALL new entries near high-impact USD events
            if macro_info.get("block_new_entries"):
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — Macro Event ใกล้เวลานี้")
                continue

            # [#3] Funding Rate hard filter: block brand-new longs if funding too hot
            if funding_info.get("block_long"):
                logger.info(f"⛔ {coin}: signal '{signal_type}' ถูกระงับ — {funding_info['funding_label']}")
                continue

            # [#3] Open Interest filter: only fetched for candidates that already
            # passed the basic signal_type + funding gate (saves API calls)
            oi_info = get_oi_filter_info(coin, df)

            sig_score, score_label = calculate_signal_score(
                rsi, bounce_info, candle_info, vol_confirmed,
                in_fibo, in_ob, in_fvg, weekly_ctx, mtf_info, adx, is_div, trend_info, onchain_info,
                funding_info=funding_info, oi_info=oi_info, btcd_filter=btcd_filter,
            )

            if sig_score >= MINIMUM_SIGNAL_SCORE:
                # [#1] ATR-based TP/SL
                tpsl = calculate_atr_based_tp_sl(price, atr, dyn_mult, tier, ob_info, fvg_info, ema200)
                tp1_val, tp2_val, sl_val = tpsl["tp1"], tpsl["tp2"], tpsl["sl"]
                sl_dist = max((price - sl_val) / price, 0.01)

                pos_size = get_correlation_adjusted_position(PORTFOLIO_USDT, RISK_PER_TRADE_PCT, sl_dist, corr_btc, active_signal_count)
                active_signal_count += 1

                # Bearish bias warning for downside reversal setups (signal entered below EMA200)
                downside_warning = ""
                if price <= ema200 and bias_warning:
                    downside_warning = "\n" + bias_warning

                # Funding/OI/BTC.D/Macro advisory notes (non-blocking warnings)
                advisory_lines = []
                if funding_info.get("has_data"):
                    advisory_lines.append(f"💸 {funding_info['funding_label']}")
                if oi_info.get("has_data"):
                    advisory_lines.append(f"📊 {oi_info['oi_label']}")
                if btcd_filter.get("note"):
                    advisory_lines.append(f"🌐 {btcd_filter['note']}")
                if macro_info.get("has_data") and macro_info.get("events"):
                    advisory_lines.append(f"📅 {macro_info['macro_label']}")
                advisory_block = ("\n" + "\n".join(advisory_lines)) if advisory_lines else ""


                buy_signals.append({
                    "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                    "type": signal_type, "score_label": score_label,
                    "tp1": f"${format_price(tp1_val)}",
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
                    "partial_tp_note": f"📐 Partial TP: ปิด {PARTIAL_TP1_CLOSE_PCT*100:.0f}% ที่ TP1, เหลือ {100-PARTIAL_TP1_CLOSE_PCT*100:.0f}% เป็น Runner (Trail ATR x{RUNNER_TRAIL_ATR_MULTIPLIER})",
                })

                # Record the new position so future runs track SL/TP/trailing/partial
                open_position(positions, coin, price, sl_val, tp1_val, tp2_val, tier, atr_at_entry=atr)


    summary_msg = (
        f"📊 <b>[Market Summary – v6.0 ATR-TP/Partial/Runner + Funding/OI/BTC.D/Macro]</b>\n"
        f"ดัชนีหลัก (BTC): <b>{market_regime}</b>\n"
        f"🌐 {btcd_info.get('btcd_label', '⚪ N/A BTC.D')}\n"
        f"📅 {macro_info.get('macro_label', '')}\n"
        f"📈 ขาขึ้น: {bullish_coins} | 📉 ขาลง: {bearish_coins}\n\n"
        + "\n".join(coin_trends_summary)
    )
    return buy_signals, exit_watch_signals, position_updates, summary_msg

def build_messages(buy_list, exit_watch_list, position_updates, market_summary) -> list:
    blocks = [market_summary]

    # --- Position Updates Section (from positions.json) ---
    if position_updates:
        current = "📋 <b>[Position Updates – สถานะที่ติดตามอยู่]</b>"
        action_icons = {
            "close_sl": "❌ ปิดสถานะเต็มจำนวน (โดน SL ก่อน TP1)",
            "close_tp2": "🏁 ปิด Runner ที่เหลือ (ถึง TP2 เต็มเป้า)",
            "close_runner_sl": "🔚 ปิด Runner ที่เหลือ (โดน Trailing SL)",
            "partial_tp1": "🎯 Partial TP1 — ปิดบางส่วน + เริ่ม Runner",
            "time_stop": "⏳ Time-Stop Warning",
            "update_sl": "🔧 ปรับ SL",
            "info": "ℹ️ อัปเดต",
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
        current = "🎯 <b>[Crypto Screener 4H – สัญญาณซื้อ]</b>"
        for opt in buy_list:
            vol = "🔊 Volume ยืนยัน" if opt["vol_confirmed"] else "🔇 Volume ต่ำ"
            time_stop_line = f"\n{opt['time_stop_label']}" if opt.get("time_stop_label") else ""
            tp_method_note = " (ATR-Based)" if opt.get("tp_method") == "atr_based" else " (Fixed % Fallback)"
            msg = (
                f"\n\n🪙 <b>{opt['coin']}</b> | ราคา: ${opt['price']}\n"
                f"🚨 สัญญาณ: <b>{opt['type']}</b> | RSI: {opt['rsi']}\n"
                f"{vol}\n"
                f"🏆 {opt['score_label']}\n"
                f"📡 MTF: {opt['mtf_label']}\n"
                f"{opt['reversal_label']}\n"
                f"📐 แนวโน้ม: {opt['trend_label']}\n"
                f"🔄 RSI Bounce: {opt['bounce_label']}\n"
                f"🔗 On-Chain (1M): {opt['onchain_label']}\n"
                f"🔁 Correlation กับ BTC (returns): {opt['corr_btc']}"
                f"{opt['advisory_block']}\n\n"
                f"💼 Position: <code>{opt['pos_size']}</code> | SL ระยะ: {opt['sl_risk_pct']}\n"
                f"❌ Hard SL: <code>{opt['sl']}</code>\n"
                f"🚀 TP1: <code>{opt['tp1']}</code> | TP2: <code>{opt['tp2']}</code>{tp_method_note}\n"
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

    # --- Exit Watch Section ---
    if exit_watch_list:
        current = "🚪 <b>[Exit Watch – พิจารณาปิดสถานะ / เลื่อน SL]</b>\n<i>สำหรับเหรียญที่อยู่ในแนวโน้มขาขึ้นและเริ่มมีสัญญาณอ่อนแรง</i>"
        for opt in sorted(exit_watch_list, key=lambda x: -x["exit_score"]):
            reasons_txt = "\n".join(f"  • {r}" for r in opt["exit_reasons"])
            time_stop_line = f"\n{opt['time_stop_label']}" if opt.get("time_stop_label") else ""
            msg = (
                f"\n\n🪙 <b>{opt['coin']}</b> | ราคา: ${opt['price']} | RSI: {opt['rsi']}\n"
                f"{opt['exit_label']}\n"
                f"📐 แนวโน้ม: {opt['trend_label']}\n"
                f"สาเหตุ:\n{reasons_txt}"
                f"{time_stop_line}"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "🚪 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)

    return blocks

if __name__ == "__main__":
    try:
        ip = api_session.get("https://ifconfig.me", timeout=5).text.strip()
        logger.info(f"🌐 Network IP: {ip} (Proxy Active: {bool(PROXIES)})")
    except Exception: pass

    positions = load_positions()
    buy_list, exit_watch_list, position_updates, market_summary = scan_market(positions)
    send_telegram_messages(build_messages(buy_list, exit_watch_list, position_updates, market_summary))
    save_positions(positions)
    logger.info("ระบบทำงานสมบูรณ์!")
