import os
import time
import math
import logging
import requests
import pandas as pd
import numpy as np
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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
# ดึงค่าและจัดการช่องว่างทันทีเพื่อความปลอดภัย
CRYPTOCOMPARE_API_KEY = str(os.getenv("CRYPTOCOMPARE_API_KEY") or "").strip()

# ตรวจสอบเบื้องต้นใน GitHub Actions log ว่าตรวจพบ Key หรือไม่
if not CRYPTOCOMPARE_API_KEY:
    logger.warning("⚠️ ไม่พบ CRYPTOCOMPARE_API_KEY ใน Environment Variables (โปรดตรวจสอบ GitHub Secrets)")

PORTFOLIO_USDT      = 1500.0
RISK_PER_TRADE_PCT  = 2.0
MAX_TOTAL_RISK_PCT  = 6.0   # จำกัด total exposure ต่อรอบ

COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "FLOKI", "SHIB", "EIGEN", "OP", "DOGE", "NEAR",
    "TRX", "AVAX", "SUI"
]

# ==========================================
# Constants & Hyperparameters
# ==========================================
API_RATE_LIMIT_DELAY = 2.0   # ลดลงจาก 0.35 เพราะ bulk fetch ลด calls
API_MAX_RETRIES      = 3
API_RETRY_DELAY      = 2.0
HISTOHOUR_LIMIT      = 2000

CACHE_TTL_SECONDS    = 3600   # cache อายุ 1 ชั่วโมง

# --- Indicators ---
RSI_PERIOD    = 14
EMA_SHORT     = 50
EMA_LONG      = 200
RSI_OVERSOLD  = 32
RSI_OVERBOUGHT = 70
ATR_PERIOD    = 14

# --- RSI Recovery & Pullback ---
RSI_RECOVERY_THRESHOLD = 45
RSI_PULLBACK_THRESHOLD = 55
RSI_RECOVERY_LOOKBACK  = 5

# --- Divergence ---
RSI_BULL_DIV_MAX  = 45
RSI_BEAR_DIV_MIN  = 55
LOOKBACK_BARS     = 15
LOOKBACK_SKIP_BARS = 3

# --- Trend Continuity ---
TREND_SLOPE_BARS      = 5
TREND_MIN_CONSECUTIVE = 3

# --- RSI Bounce ---
RSI_BOUNCE_CONFIRM_BARS = 2
RSI_BOUNCE_MIN_RISE     = 3.0

# --- Order Block (SMC) & FVG ---
OB_LOOKBACK        = 20
OB_IMBALANCE_RATIO = 1.5
FVG_THRESHOLD_PCT  = 0.2

# --- Signal Score Threshold ---
MINIMUM_SIGNAL_SCORE = 50   # ส่ง alert เฉพาะ score >= ค่านี้

# --- On-Chain: เฉพาะ coin ที่ CryptoCompare รองรับจริง ---
ONCHAIN_SUPPORTED_COINS = {"BTC", "ETH", "BNB", "SOL"}

# --- Take Profit Tiers ---
TP_TIERS = {
    "major": {"tp1": 0.10, "tp2": 0.15, "sl_buffer": 0.025},
    "mid":   {"tp1": 0.15, "tp2": 0.20, "sl_buffer": 0.050},
    "small": {"tp1": 0.20, "tp2": 0.35, "sl_buffer": 0.080},
}

COIN_TIER = {
    "BTC": "major", "ETH": "major",
    "BNB": "mid",   "SOL": "mid",   "XRP": "mid",
    "ADA": "mid",   "NEAR": "mid",  "OP":  "mid",
    "TRX": "mid",   "AVAX": "mid",
    "FLOKI": "small","SHIB": "small","EIGEN": "small",
    "DOGE": "small","SUI": "small",
}

# ==========================================
# Global API Session
# ==========================================
api_session = requests.Session()
api_session.headers.update({
    "User-Agent": "CryptoScreenerBot/2.0 (Mozilla/5.0; Trading API System)",
    "Accept":     "application/json",
    "Authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"
})

retry_strategy = Retry(
    total=API_MAX_RETRIES,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
api_session.mount("https://", adapter)

# ==========================================
# In-Memory Cache (ลด API calls ซ้ำภายใน 1 ชั่วโมง)
# ==========================================
_cache_4h:   dict  = {}
_cache_1d:   dict  = {}
_cache_ts_4h: float = 0.0
_cache_ts_1d: float = 0.0

# ==========================================
# Telegram Integration
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
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Telegram ส่งสำเร็จ (ส่วน {idx}/{len(chunks)})")
            else:
                logger.warning(f"Telegram ส่งล้มเหลว (ส่วน {idx}): {resp.text}")
        except Exception as e:
            logger.error(f"Exception ขณะส่ง Telegram (ส่วน {idx}): {e}")
        if idx < len(chunks):
            time.sleep(0.5)

# ==========================================
# Data Fetching (ปรับปรุง: แนบ api_key ใน params ตรงๆ)
# ==========================================
def get_historical_data(coin: str) -> pd.DataFrame | None:
    """ดึง 4H OHLCV จาก CryptoCompare (histohour → resample 4H)"""
    url    = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": coin, 
        "tsym": "USD", 
        "limit": HISTOHOUR_LIMIT,
        "api_key": CRYPTOCOMPARE_API_KEY  # <--- แก้ไขจุดที่ 1
    }
    try:
        resp = api_session.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df_4h = df.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volumeto": "sum"}
            ).dropna()
            return df_4h
        else:
            logger.warning(f"{coin} 4H: {data.get('Message')}")
    except Exception as e:
        logger.warning(f"{coin} 4H fetch error: {e}")
    return None


def get_histoday_data(coin: str) -> pd.DataFrame | None:
    """ดึง 1D OHLCV จาก CryptoCompare"""
    url    = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": coin, 
        "tsym": "USD", 
        "limit": 2000,
        "api_key": CRYPTOCOMPARE_API_KEY  # <--- แก้ไขจุดที่ 2
    }
    try:
        resp = api_session.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            return df
        else:
            logger.warning(f"{coin} 1D: {data.get('Message')}")
    except Exception as e:
        logger.warning(f"{coin} 1D fetch error: {e}")
    return None

# ==========================================
# Bulk Cache Fetchers (ลด API calls ~60%)
# ==========================================
def bulk_fetch_4h(coins: list) -> dict:
    """ดึง 4H ทุก coin ครั้งเดียว + cache 1 ชั่วโมง"""
    global _cache_4h, _cache_ts_4h
    now = time.time()
    if _cache_4h and (now - _cache_ts_4h) < CACHE_TTL_SECONDS:
        logger.info("ใช้ Cache 4H (ไม่ยิง API ซ้ำ)")
        return _cache_4h
    result = {}
    for coin in coins:
        df = get_historical_data(coin)
        if df is not None:
            result[coin] = df
        time.sleep(API_RATE_LIMIT_DELAY)
    _cache_4h    = result
    _cache_ts_4h = now
    logger.info(f"Bulk fetch 4H สำเร็จ: {len(result)}/{len(coins)} coins")
    return result


def bulk_fetch_1d(coins: list) -> dict:
    """ดึง 1D ทุก coin ครั้งเดียว + cache 1 ชั่วโมง"""
    global _cache_1d, _cache_ts_1d
    now = time.time()
    if _cache_1d and (now - _cache_ts_1d) < CACHE_TTL_SECONDS:
        logger.info("ใช้ Cache 1D (ไม่ยิง API ซ้ำ)")
        return _cache_1d
    result = {}
    for coin in coins:
        df = get_histoday_data(coin)
        if df is not None:
            result[coin] = df
        time.sleep(API_RATE_LIMIT_DELAY)
    _cache_1d    = result
    _cache_ts_1d = now
    logger.info(f"Bulk fetch 1D สำเร็จ: {len(result)}/{len(coins)} coins")
    return result

# ==========================================
# Core Indicators
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    df["EMA_50"]  = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LONG,  adjust=False).mean()

    # RSI (Wilder's Smoothing)
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    # ADX
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_safe  = df["ATR"].replace(0, np.nan)
    plus_di   = 100 * (pd.Series(plus_dm,  index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    minus_di  = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    dx        = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)

    df["VOL_MA20"] = df["volumeto"].rolling(20).mean()
    return df

# ==========================================
# Candle Pattern Confirmation
# ==========================================
def confirm_reversal_candle(df: pd.DataFrame) -> dict:
    """ตรวจ Candle Pattern ยืนยันการกลับตัว ลด False Entry"""
    result = {
        "has_bullish_engulfing": False,
        "has_hammer":            False,
        "has_morning_star":      False,
        "reversal_strength":     0,
        "reversal_label":        "⬜ ไม่มี Candle ยืนยัน",
    }
    if len(df) < 3:
        return result

    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]

    body0       = abs(c0["close"] - c0["open"])
    body1       = abs(c1["close"] - c1["open"])
    body2       = abs(c2["close"] - c2["open"])
    lower_wick0 = min(c0["open"], c0["close"]) - c0["low"]
    upper_wick0 = c0["high"] - max(c0["open"], c0["close"])

    # Bullish Engulfing
    if (c1["close"] < c1["open"] and
            c0["close"] > c0["open"] and
            c0["open"] <= c1["close"] and
            c0["close"] >= c1["open"]):
        result["has_bullish_engulfing"] = True
        result["reversal_strength"] += 2

    # Hammer
    if (body0 > 0 and
            lower_wick0 >= 2 * body0 and
            upper_wick0 <= 0.3 * body0 and
            c0["close"] > c0["open"]):
        result["has_hammer"] = True
        result["reversal_strength"] += 1

    # Morning Star (3 bars)
    if (c2["close"] < c2["open"] and body2 > 0 and
            body1 < body2 * 0.4 and
            c0["close"] > c0["open"] and
            c0["close"] > (c2["open"] + c2["close"]) / 2):
        result["has_morning_star"] = True
        result["reversal_strength"] += 2

    if result["reversal_strength"] >= 3:
        result["reversal_label"] = "🕯️ <b>Candle ยืนยันแข็งแกร่ง (Engulfing / Morning Star)</b>"
    elif result["reversal_strength"] >= 1:
        result["reversal_label"] = "🕯️ Candle ยืนยันปานกลาง (Hammer)"

    return result

# ==========================================
# Dynamic ATR Multiplier
# ==========================================
def get_dynamic_atr_multiplier(tier: str, adx: float, atr_pct: float) -> float:
    base = {"major": 2.0, "mid": 2.5, "small": 3.0}.get(tier, 2.5)
    if adx > 25:
        base *= 1.20
    if atr_pct > 4.0:
        base *= 1.15
    if adx < 15:
        base *= 0.85
    return round(min(base, 4.0), 2)

# ==========================================
# Correlation-Adjusted Position Sizing
# ==========================================
def get_correlation_adjusted_position(
    portfolio: float,
    risk_pct: float,
    sl_distance_pct: float,
    corr_btc: float,
    active_signals_count: int
) -> float:
    base_risk = portfolio * (risk_pct / 100)

    if corr_btc > 0.85:
        base_risk *= 0.70
    elif corr_btc > 0.70:
        base_risk *= 0.85

    max_total_risk  = portfolio * (MAX_TOTAL_RISK_PCT / 100)
    if active_signals_count > 0:
        per_signal_cap = max_total_risk / (active_signals_count + 1)
        base_risk      = min(base_risk, per_signal_cap)

    position = base_risk / max(sl_distance_pct, 0.01)
    return min(position, portfolio * 0.25)

# ==========================================
# Multi-Timeframe RSI Confluence
# ==========================================
def get_mtf_rsi_alignment(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    result = {
        "aligned_oversold":  False,
        "aligned_overbought": False,
        "rsi_4h":            None,
        "rsi_1d":            None,
        "mtf_label":         "",
        "confluence_score":  0,
    }
    if df_4h is None or df_1d is None:
        return result

    rsi_4h = df_4h["RSI"].iloc[-1] if "RSI" in df_4h.columns else 50.0

    close_1d = df_1d["close"]
    delta    = close_1d.diff()
    gain     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs       = gain / loss.replace(0, np.nan)
    rsi_1d_series = (100 - 100 / (1 + rs))
    rsi_1d   = float(rsi_1d_series.iloc[-1]) if len(rsi_1d_series) > 0 else 50.0

    result["rsi_4h"] = round(rsi_4h, 2)
    result["rsi_1d"] = round(rsi_1d, 2)

    score = 0
    if rsi_4h <= RSI_OVERSOLD:
        score += 2
    elif rsi_4h <= 45:
        score += 1

    if rsi_1d <= RSI_OVERSOLD:
        score += 2
    elif rsi_1d <= 45:
        score += 1

    result["confluence_score"] = score

    if score >= 4:
        result["aligned_oversold"] = True
        result["mtf_label"] = (
            f"💎 MTF RSI Oversold พร้อมกัน! "
            f"(4H:{rsi_4h:.1f} / 1D:{rsi_1d:.1f})"
        )
    elif score >= 2:
        result["mtf_label"] = (
            f"🟡 MTF RSI อ่อนแรงปานกลาง "
            f"(4H:{rsi_4h:.1f} / 1D:{rsi_1d:.1f})"
        )
    elif rsi_4h >= RSI_OVERBOUGHT and rsi_1d >= RSI_OVERBOUGHT:
        result["aligned_overbought"] = True
        result["mtf_label"] = (
            f"🔴 MTF RSI Overbought พร้อมกัน! "
            f"(4H:{rsi_4h:.1f} / 1D:{rsi_1d:.1f})"
        )

    return result

# ==========================================
# Signal Scoring System
# ==========================================
def calculate_signal_score(
    rsi:             float,
    bounce_info:     dict,
    candle_info:     dict,
    vol_confirmed:   bool,
    in_fibo_zone:    bool,
    in_ob_zone:      bool,
    in_fvg_zone:     bool,
    weekly_ctx:      dict,
    mtf_info:        dict,
    adx:             float,
    is_divergence:   bool,
    trend_info:      dict,
) -> tuple[int, str]:
    score   = 0
    factors = []

    # RSI Zone (max 25)
    if rsi <= RSI_OVERSOLD:
        score += 25; factors.append(f"RSI Oversold({rsi:.1f})")
    elif rsi <= 40:
        score += 15; factors.append(f"RSI ต่ำ({rsi:.1f})")
    elif rsi <= 50:
        score += 8

    # Bounce Quality (max 20)
    bq_map = {"strong": 20, "moderate": 12, "weak": 5, "none": 0}
    bq     = bq_map.get(bounce_info.get("quality", "none"), 0)
    score += bq
    if bq >= 12:
        factors.append("RSI Bounce")

    # Candle Confirmation (max 15)
    candle_pts = min(candle_info.get("reversal_strength", 0) * 7, 15)
    score += candle_pts
    if candle_pts > 0:
        factors.append("Candle Confirm")

    # Zone Confluence (max 20)
    if in_fibo_zone:
        score += 8;  factors.append("Fibo Zone")
    if in_ob_zone:
        score += 7;  factors.append("OB Zone")
    if in_fvg_zone:
        score += 5;  factors.append("FVG Zone")

    # Bullish Divergence (max 10)
    if is_divergence:
        score += 10; factors.append("Bullish Div")

    # Volume (max 10)
    if vol_confirmed:
        score += 10; factors.append("Volume OK")

    # MTF RSI Alignment (max 10)
    mtf_pts = min(mtf_info.get("confluence_score", 0) * 2, 10)
    score  += mtf_pts
    if mtf_info.get("aligned_oversold"):
        factors.append("MTF Aligned")

    # Weekly context bonus (max 5)
    if weekly_ctx.get("weekly_bullish_div"):
        score += 5;  factors.append("1W Div")
    elif weekly_ctx.get("rsi_weekly") and weekly_ctx["rsi_weekly"] <= 35:
        score += 3

    # ADX adjustment (±5)
    if adx > 25:
        score += 5
    elif adx < 15:
        score -= 5

    # Trend continuity bonus
    ts = trend_info.get("trend_strength", "sideways")
    if ts == "strong_up":
        score += 5;  factors.append("Strong Uptrend")
    elif ts == "moderate_up":
        score += 2

    score = max(0, min(100, score))

    if score >= 70:
        grade = "🔥 A+ (แนะนำเข้า)"
    elif score >= 55:
        grade = "✅ A  (พิจารณาได้)"
    elif score >= MINIMUM_SIGNAL_SCORE:
        grade = "🟡 B  (รอยืนยันเพิ่ม)"
    else:
        grade = "⬜ C  (ยังไม่แนะนำ)"

    label = f"{grade} | Score: {score}/100 | [{', '.join(factors)}]"
    return score, label

# ==========================================
# Weekly / Monthly / Cycle / Death Cross
# ==========================================
def analyze_weekly_context(df_1d: pd.DataFrame) -> dict:
    result = {
        "rsi_weekly":          None,
        "weekly_bullish_div":  False,
        "weekly_status_label": "↔️ ไม่พบข้อมูลระบุระดับสัปดาห์ชัดเจน",
        "fibo_618":    None, "fibo_786":  None,
        "fibo_886":    None, "liquidity_pool": None, "psycho_support": None,
    }
    if df_1d is None or len(df_1d) < RSI_PERIOD + LOOKBACK_BARS + 5:
        return result
    try:
        df_w = df_1d.resample("W").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(df_w) < RSI_PERIOD + LOOKBACK_BARS + 5:
            return result

        fibo_window = df_w.iloc[-104:]
        w_max  = fibo_window["high"].max()
        w_min  = fibo_window["low"].min()
        w_diff = w_max - w_min
        result.update({
            "fibo_618":      w_max - (0.618 * w_diff),
            "fibo_786":      w_max - (0.786 * w_diff),
            "fibo_886":      w_max - (0.886 * w_diff),
            "liquidity_pool": w_min,
        })

        curr_price = df_w["close"].iloc[-1]
        if curr_price > 0:
            magnitude = 10 ** math.floor(math.log10(curr_price))
            step      = magnitude if curr_price >= magnitude * 2 else magnitude / 2
            result["psycho_support"] = math.floor(curr_price / step) * step

        delta    = df_w["close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df_w["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

        curr_rsi_w  = round(df_w["RSI"].iloc[-1], 2)
        result["rsi_weekly"] = curr_rsi_w

        prev_window = df_w.iloc[-(LOOKBACK_BARS + 1): -(LOOKBACK_SKIP_BARS)]
        if len(prev_window) > 0:
            min_idx       = prev_window["low"].argmin()
            prev_low_p    = prev_window["low"].iloc[min_idx]
            prev_low_rsi  = prev_window["RSI"].iloc[min_idx]
            curr_low      = df_w["low"].iloc[-1]
            curr_rsi_now  = df_w["RSI"].iloc[-1]
            if (prev_low_rsi <= RSI_BULL_DIV_MAX and
                    curr_low < prev_low_p and curr_rsi_now > prev_low_rsi):
                result["weekly_bullish_div"] = True

        if result["weekly_bullish_div"]:
            result["weekly_status_label"] = (
                f"👑 <b>เกิด Weekly Bullish Divergence!</b> (RSI: {curr_rsi_w})"
            )
        elif curr_rsi_w <= RSI_OVERSOLD:
            result["weekly_status_label"] = (
                f"🔥 <b>ภาพใหญ่ Oversold รุนแรง ({curr_rsi_w})</b>"
            )
        elif result["fibo_786"] and curr_price <= result["fibo_786"]:
            result["weekly_status_label"] = (
                "⚠️ ราคาลงทดสอบโซน Deep Discount (ใต้ Fibo 78.6%)"
            )
        elif curr_rsi_w >= RSI_OVERBOUGHT:
            result["weekly_status_label"] = (
                f"⚠️ ภาพใหญ่ Overbought ({curr_rsi_w}) ระวังปรับฐาน"
            )
        else:
            result["weekly_status_label"] = (
                f"↔️ ภาพใหญ่ทรงตัวปกติ (RSI 1W: {curr_rsi_w})"
            )
    except Exception as e:
        logger.error(f"analyze_weekly_context error: {e}")
    return result


def analyze_monthly_targets(df_1d: pd.DataFrame) -> dict:
    result = {
        "m_resistance_target": None, "m_support_target": None,
        "monthly_summary_label": "⏳ ไม่สามารถคำนวณเป้าหมายระดับเดือนได้",
        "monthly_trend": "sideways",
    }
    if df_1d is None or len(df_1d) < 30:
        return result
    try:
        df_m = df_1d.resample("ME").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(df_m) < 5:
            return result
        last_month    = df_m.iloc[-2]
        current_month = df_m.iloc[-1]
        m_high, m_low, m_close = last_month["high"], last_month["low"], last_month["close"]
        pivot = (m_high + m_low + m_close) / 3
        r1    = (2 * pivot) - m_low
        s1    = (2 * pivot) - m_high
        m_ema12 = (df_m["close"].ewm(span=12, adjust=False).mean().iloc[-1]
                   if len(df_m) >= 12 else pivot)
        curr_price = current_month["close"]
        if curr_price >= m_ema12:
            result.update({
                "m_resistance_target":  max(r1, m_high),
                "m_support_target":     pivot,
                "monthly_summary_label": "🔮 <b>ภาพ 1M (Bullish):</b> เป้าหมายราคาทดสอบกรอบบน",
                "monthly_trend":        "bullish",
            })
        else:
            result.update({
                "m_resistance_target":  pivot,
                "m_support_target":     min(s1, m_low),
                "monthly_summary_label": "🔮 <b>ภาพ 1M (Bearish):</b> แนวโน้มไหลลงหาแนวรับกรอบล่าง",
                "monthly_trend":        "bearish",
            })
    except Exception:
        pass
    return result


def analyze_cycle_targets(df_1d: pd.DataFrame) -> dict:
    result = {
        "cycle_target_zone": None, "cycle_confluence_factors": [],
        "cycle_summary_label": "⏳ ไม่สามารถวิเคราะห์เป้าหมายไซเคิลได้",
    }
    if df_1d is None or len(df_1d) < 150:
        return result
    try:
        df_w = df_1d.resample("W").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(df_w) < 104:
            return result
        curr_price = df_w["close"].iloc[-1]
        df_w["EMA_50"]  = df_w["close"].ewm(span=50,  adjust=False).mean()
        df_w["EMA_200"] = df_w["close"].ewm(span=200, adjust=False).mean()
        w_ema50  = df_w["EMA_50"].iloc[-1]
        w_ema200 = df_w["EMA_200"].iloc[-1]

        macro_window = df_w.iloc[-156:] if len(df_w) >= 156 else df_w
        macro_high, macro_low = macro_window["high"].max(), macro_window["low"].min()
        fib_1618 = macro_low  + ((macro_high - macro_low) * 1.618)
        fib_0618 = macro_low  + ((macro_high - macro_low) * 0.618)

        potential = []
        if macro_high > curr_price * 1.1:
            potential.append(("Historical Resistance (Previous Top)", macro_high))
        if fib_0618 > curr_price * 1.1:
            potential.append(("Macro Fibonacci 61.8% (Golden Pocket)", fib_0618))
        if fib_1618 > curr_price * 1.1:
            potential.append(("Fibonacci Extension 161.8%", fib_1618))
        if w_ema50 > curr_price * 1.05:
            potential.append(("Weekly EMA 50 (Dynamic Resistance)", w_ema50))
        if w_ema200 > curr_price * 1.05:
            potential.append(("Weekly EMA 200 (Macro Trendline)", w_ema200))

        potential.sort(key=lambda x: x[1])
        confluence_zones = []
        for i in range(len(potential)):
            cluster = [potential[i]]
            for j in range(i + 1, len(potential)):
                if (potential[j][1] - cluster[0][1]) / cluster[0][1] <= 0.10:
                    cluster.append(potential[j])
            if len(cluster) >= 2:
                avg_price = sum(x[1] for x in cluster) / len(cluster)
                if not any(abs(avg_price - cz["price"]) / cz["price"] < 0.1
                           for cz in confluence_zones):
                    confluence_zones.append({
                        "price": avg_price,
                        "min_zone": cluster[0][1],
                        "max_zone": cluster[-1][1],
                        "factors": [x[0] for x in cluster],
                    })

        if confluence_zones:
            pt = confluence_zones[0]
            result.update({
                "cycle_target_zone": f"${format_price(pt['min_zone'])} - ${format_price(pt['max_zone'])}",
                "cycle_confluence_factors": pt["factors"],
                "cycle_summary_label": (
                    f"🎯 <b>เป้าหมายรอบไซเคิล (Confluence):</b> "
                    f"<code>{format_price(pt['min_zone'])} - {format_price(pt['max_zone'])}</code>\n"
                    f"   <i>(จุดบรรจบ: {' และ '.join(pt['factors'])})</i>"
                ),
            })
        elif potential:
            nt = potential[0]
            result.update({
                "cycle_target_zone": f"${format_price(nt[1])}",
                "cycle_summary_label": (
                    f"🎯 <b>เป้าหมายรอบไซเคิล:</b> <code>${format_price(nt[1])}</code>\n"
                    f"   <i>(อ้างอิง: {nt[0]})</i>"
                ),
            })
    except Exception:
        pass
    return result


def check_death_cross_1d(df_1d: pd.DataFrame) -> dict:
    result = {
        "has_death_cross":  False,
        "death_cross_label": "⚪ 1D Death Cross: ปลอดภัย",
    }
    if df_1d is None or len(df_1d) < 200:
        return result
    try:
        df       = df_1d.copy()
        df["close"] = pd.to_numeric(df["close"])
        df["EMA_50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
        df_last = df.iloc[-30:]
        for i in range(1, len(df_last)):
            p50, p200 = df_last["EMA_50"].iloc[i-1], df_last["EMA_200"].iloc[i-1]
            c50, c200 = df_last["EMA_50"].iloc[i],   df_last["EMA_200"].iloc[i]
            if p50 >= p200 and c50 < c200:
                result["has_death_cross"]  = True
                result["death_cross_label"] = (
                    "☠️ <b>Warning! เกิด Death Cross (1D) ในช่วง 1 เดือนที่ผ่านมา</b>"
                )
                break
    except Exception as e:
        logger.warning(f"Death Cross check error: {e}")
    return result

# ==========================================
# On-Chain (ปรับปรุง: แนบ api_key ใน params ตรงๆ)
# ==========================================
def analyze_onchain_momentum(coin: str) -> dict:
    result = {
        "onchain_warning":      False,
        "onchain_label":        "🔹 On-Chain (1M): ปกติ",
        "large_tx_change_pct":  0.0,
    }
    url    = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
    params = {
        "fsym": coin, 
        "limit": 30,
        "api_key": CRYPTOCOMPARE_API_KEY  # <--- แก้ไขจุดที่ 3
    }
    try:
        resp = api_session.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("Response") == "Success" and isinstance(
                data.get("Data", {}).get("Data"), list):
            df_oc = pd.DataFrame(data["Data"]["Data"])
            if len(df_oc) >= 15:
                tx_col = next(
                    (c for c in
                     ["large_transaction_count", "transaction_count",
                      "zero_balance_addresses_all_time"]
                     if c in df_oc.columns), None)
                if tx_col:
                    recent_avg = df_oc[tx_col].iloc[-5:].mean()
                    prior_avg  = df_oc[tx_col].iloc[-30:-5].mean()
                    if prior_avg > 0:
                        chg = ((recent_avg - prior_avg) / prior_avg) * 100
                        result["large_tx_change_pct"] = round(chg, 2)
                        if chg > 40.0:
                            result["onchain_warning"] = True
                            result["onchain_label"] = (
                                f"🚨 <b>On-Chain Warning:</b> ธุรกรรมวาฬพุ่ง {chg:.1f}%"
                            )
                        else:
                            result["onchain_label"] = (
                                f"🔹 On-Chain (1M): ปกติ ({chg:+.1f}%)"
                            )
    except Exception as e:
        logger.warning(f"On-Chain {coin}: {e}")
    return result

# ==========================================
# Futures Context (Bulk Funding Rate)
# ==========================================
def analyze_futures_context(coin: str, bulk_funding_rate: float | None) -> dict:
    result = {
        "futures_warning": False,
        "futures_label":   "📊 Futures (1M): ปกติ / ไม่มีข้อมูล",
    }
    symbol = f"{coin}USDT"
    try:
        oi_url    = "https://fapi.binance.com/futures/data/openInterestHist"
        oi_params = {"symbol": symbol, "period": "1d", "limit": 30}
        oi_resp   = api_session.get(oi_url, params=oi_params, timeout=10)
        if oi_resp.status_code == 200 and bulk_funding_rate is not None:
            oi_data = oi_resp.json()
            if len(oi_data) >= 2:
                first_oi = float(oi_data[0]["sumOpenInterestValue"])
                last_oi  = float(oi_data[-1]["sumOpenInterestValue"])
                oi_chg   = ((last_oi - first_oi) / first_oi) * 100 if first_oi > 0 else 0
                fr_pct   = bulk_funding_rate * 100
                if oi_chg > 25.0 and fr_pct > 0.015:
                    result.update({
                        "futures_warning": True,
                        "futures_label": (
                            f"🚨 <b>Futures Warning: ระวัง Long Squeeze!</b> "
                            f"OI +{oi_chg:.1f}%, FR {fr_pct:.4f}%"
                        ),
                    })
                elif oi_chg > 25.0 and fr_pct < -0.015:
                    result.update({
                        "futures_warning": True,
                        "futures_label": (
                            f"🚨 <b>Futures Warning: ระวัง Short Squeeze!</b> "
                            f"OI +{oi_chg:.1f}%, FR {fr_pct:.4f}%"
                        ),
                    })
                else:
                    result["futures_label"] = (
                        f"📊 Futures (1M): ปกติ (OI {oi_chg:+.1f}%, FR {fr_pct:.4f}%)"
                    )
    except Exception:
        pass
    return result

# ==========================================
# Technical Analysis Helpers
# ==========================================
def calculate_correlation(df: pd.DataFrame, btc_df: pd.DataFrame) -> float:
    if btc_df is None or df is None:
        return 0.0
    try:
        aligned = df["close"].align(btc_df["close"], join="inner")
        if len(aligned[0]) < 2:
            return 0.0
        return round(float(aligned[0].corr(aligned[1])), 2)
    except Exception:
        return 0.0


def estimate_price_for_target_rsi(
    df: pd.DataFrame, target_rsi: float = 70.0, period: int = RSI_PERIOD
) -> float | None:
    if len(df) < period + 1:
        return None
    close    = df["close"]
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean().iloc[-1]
    curr_price = close.iloc[-1]
    if target_rsi >= 100:
        return curr_price * 1.5
    if target_rsi <= 0:
        return curr_price * 0.5
    target_rs = target_rsi / (100.0 - target_rsi)
    if target_rsi > 50:
        next_avg_loss   = (avg_loss * (period - 1)) / period
        req_avg_gain    = target_rs * next_avg_loss
        required_gain   = max(0, (req_avg_gain * period) - (avg_gain * (period - 1)))
        return curr_price + required_gain
    else:
        next_avg_gain   = (avg_gain * (period - 1)) / period
        req_avg_loss    = next_avg_gain / target_rs
        required_loss   = max(0, (req_avg_loss * period) - (avg_loss * (period - 1)))
        return curr_price - required_loss


def find_fair_value_gaps(df: pd.DataFrame) -> dict:
    fvg = {"has_fvg_support": False, "fvg_top": None, "fvg_bottom": None}
    if len(df) < 4:
        return fvg
    for i in range(len(df) - 1, 2, -1):
        h_m2 = df["high"].iloc[i - 2]
        l_cur = df["low"].iloc[i]
        c_m1  = df["close"].iloc[i - 1]
        o_m1  = df["open"].iloc[i - 1]
        if l_cur > h_m2 and c_m1 > o_m1:
            gap_pct = ((l_cur - h_m2) / h_m2) * 100
            if gap_pct >= FVG_THRESHOLD_PCT:
                if df["close"].iloc[-1] > h_m2:
                    fvg.update({"has_fvg_support": True, "fvg_top": l_cur, "fvg_bottom": h_m2})
                    break
    return fvg


def analyze_trend_continuity(df: pd.DataFrame) -> dict:
    result = {
        "ema50_slope_pct": 0.0, "ema200_slope_pct": 0.0,
        "ema50_trending_up": False, "ema200_trending_up": False,
        "consecutive_up": 0, "consecutive_down": 0,
        "trend_strength": "sideways", "trend_label": "↔️ ไม่ชัดเจน",
    }
    n = TREND_SLOPE_BARS
    if len(df) < n + 2:
        return result

    ema50_now,  ema50_prev  = df["EMA_50"].iloc[-1],  df["EMA_50"].iloc[-(n+1)]
    ema200_now, ema200_prev = df["EMA_200"].iloc[-1], df["EMA_200"].iloc[-(n+1)]
    slope50  = ((ema50_now  - ema50_prev)  / ema50_prev)  * 100 if ema50_prev  != 0 else 0
    slope200 = ((ema200_now - ema200_prev) / ema200_prev) * 100 if ema200_prev != 0 else 0

    result.update({
        "ema50_slope_pct":  round(slope50, 4),
        "ema200_slope_pct": round(slope200, 4),
        "ema50_trending_up":  slope50  > 0,
        "ema200_trending_up": slope200 > 0,
    })

    closes = df["close"].iloc[-20:]
    diffs  = closes.diff().iloc[1:]
    up_streak, dn_streak = 0, 0
    for val in reversed(diffs.values):
        if val > 0:
            if dn_streak == 0: up_streak += 1
            else: break
        elif val < 0:
            if up_streak == 0: dn_streak += 1
            else: break
        else:
            break

    result.update({"consecutive_up": up_streak, "consecutive_down": dn_streak})
    both_up   = slope50 > 0  and slope200 > 0
    both_down = slope50 <= 0 and slope200 <= 0

    if both_up and up_streak >= TREND_MIN_CONSECUTIVE:
        strength, label = "strong_up",   f"🚀 ขาขึ้นแข็งแกร่ง ({up_streak} แท่ง)"
    elif slope50 > 0 and up_streak >= 1:
        strength, label = "moderate_up", f"📈 ขาขึ้นปานกลาง ({up_streak} แท่ง)"
    elif both_down and dn_streak >= TREND_MIN_CONSECUTIVE:
        strength, label = "strong_down", f"🔻 ขาลงแข็งแกร่ง ({dn_streak} แท่ง)"
    elif slope50 <= 0 and dn_streak >= 1:
        strength, label = "moderate_down", f"📉 ขาลงปานกลาง ({dn_streak} แท่ง)"
    else:
        strength, label = "sideways", "↔️ Sideways / แนวโน้มไม่ชัด"

    result.update({"trend_strength": strength, "trend_label": label})
    return result


def analyze_rsi_bounce(df: pd.DataFrame) -> dict:
    window = LOOKBACK_BARS
    result = {
        "touched_oversold": False, "rsi_low": None, "rsi_rise": 0.0,
        "consecutive_rise": 0, "below_midline": False,
        "quality": "none", "quality_label": "⬜ ไม่มีสัญญาณดีดกลับ", "entry_timing": "",
    }
    if len(df) < window + RSI_BOUNCE_CONFIRM_BARS + 2:
        return result

    rsi_series = df["RSI"].iloc[-(window + 1):-1]
    rsi_curr   = df["RSI"].iloc[-1]
    rsi_min    = rsi_series.min()
    touched    = rsi_min <= RSI_OVERSOLD
    result.update({"touched_oversold": touched, "rsi_low": round(rsi_min, 2)})
    if not touched:
        return result

    rsi_rise = rsi_curr - rsi_min
    result["rsi_rise"] = round(rsi_rise, 2)
    recent_rsi = df["RSI"].iloc[-(RSI_BOUNCE_CONFIRM_BARS + 3):]
    consec     = sum(1 for v in reversed(recent_rsi.diff().iloc[1:].values) if v > 0)
    result.update({"consecutive_rise": consec, "below_midline": rsi_curr < 50})

    has_recovered = (df["RSI"].iloc[-RSI_RECOVERY_LOOKBACK:] >= RSI_RECOVERY_THRESHOLD).any()
    score = sum([
        rsi_rise >= RSI_BOUNCE_MIN_RISE,
        consec   >= RSI_BOUNCE_CONFIRM_BARS,
        rsi_curr < 50 or has_recovered,
    ])

    if score == 3:
        q, lbl, tim = "strong",   f"✅ ดีดกลับแข็งแกร่ง ({rsi_min:.1f} → +{rsi_rise:.1f})", "⭐ จังหวะเข้าซื้อดีที่สุด"
    elif score == 2:
        q, lbl, tim = "moderate", f"🟡 ดีดกลับปานกลาง ({rsi_min:.1f} → +{rsi_rise:.1f})", "⚡ พิจารณาเข้าซื้อได้"
    elif score == 1:
        q, lbl, tim = "weak",     f"🟠 ดีดกลับอ่อน (+{rsi_rise:.1f} จุด)", "⚠️ ยังไม่แนะนำ"
    else:
        q, lbl, tim = "none",     f"⬜ ยังไม่ดีดกลับ ({rsi_min:.1f})", "🚫 รอ RSI ดีดกลับก่อน"

    result.update({"quality": q, "quality_label": lbl, "entry_timing": tim})
    return result


def find_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> dict:
    ob = {"bullish_ob_price": None, "bearish_ob_price": None,
          "has_bullish_ob": False, "has_bearish_ob": False}
    if len(df) < lookback + 5:
        return ob
    body_sizes = (df["close"] - df["open"]).abs()
    avg_body   = body_sizes.rolling(20).mean().iloc[-1]
    curr_close = df["close"].iloc[-1]
    curr_open  = df["open"].iloc[-1]
    curr_body  = abs(curr_close - curr_open)
    past_df    = df.iloc[-(lookback + 1): -(LOOKBACK_SKIP_BARS)]
    recent_high, recent_low = past_df["high"].max(), past_df["low"].min()

    if curr_close > recent_high and curr_body > avg_body * OB_IMBALANCE_RATIO:
        for i in range(2, min(15, len(df))):
            idx    = -i
            p_open = df["open"].iloc[idx]
            p_close= df["close"].iloc[idx]
            p_low  = df["low"].iloc[idx]
            if p_close < p_open and not (df["low"].iloc[idx+1:] < p_low).any():
                ob.update({"has_bullish_ob": True, "bullish_ob_price": p_low})
                break
    elif curr_close < recent_low and curr_body > avg_body * OB_IMBALANCE_RATIO:
        for i in range(2, min(15, len(df))):
            idx    = -i
            p_open = df["open"].iloc[idx]
            p_close= df["close"].iloc[idx]
            p_high = df["high"].iloc[idx]
            if p_close > p_open and not (df["high"].iloc[idx+1:] > p_high).any():
                ob.update({"has_bearish_ob": True, "bearish_ob_price": p_high})
                break
    return ob


def check_bullish_divergence(df: pd.DataFrame) -> bool:
    if len(df) < LOOKBACK_BARS + 2:
        return False
    prev = df.iloc[-(LOOKBACK_BARS + 1): -(LOOKBACK_SKIP_BARS)]
    if len(prev) == 0:
        return False
    min_idx       = prev["low"].argmin()
    prev_low_p    = prev["low"].iloc[min_idx]
    prev_low_rsi  = prev["RSI"].iloc[min_idx]
    if prev_low_rsi > RSI_BULL_DIV_MAX:
        return False
    return df["low"].iloc[-1] < prev_low_p and df["RSI"].iloc[-1] > prev_low_rsi


def is_volume_confirmed(row: pd.Series) -> bool:
    if pd.isna(row.get("VOL_MA20")) or row["VOL_MA20"] == 0:
        return False
    return row["volumeto"] > row["VOL_MA20"]


def format_price(price: float) -> str:
    if price is None:
        return "N/A"
    if price < 0.0001:
        return f"{price:.8f}"
    elif price < 0.001:
        return f"{price:.6f}"
    elif price < 1:
        return f"{price:.4f}"
    else:
        return f"{price:.2f}"

# ==========================================
# Market Scanner (Optimized Pipeline)
# ==========================================
def scan_market():
    buy_signals, sell_signals         = [], []
    bullish_coins, bearish_coins      = 0, 0
    total_valid_coins                 = 0
    coin_trends_summary               = []
    active_signal_count               = 0

    logger.info("Phase 1: Bulk fetch 4H data...")
    all_4h = bulk_fetch_4h(COINS)

    logger.info("Phase 2: Bulk fetch 1D data...")
    all_1d = bulk_fetch_1d(COINS)

    logger.info("Phase 3: วิเคราะห์ Market Regime (BTC)...")
    btc_df       = all_4h.get("BTC")
    market_regime = "Unknown ⚪"
    if btc_df is not None:
        btc_df     = calculate_indicators(btc_df)
        btc_price  = btc_df["close"].iloc[-1]
        btc_ema200 = btc_df["EMA_200"].iloc[-1]
        market_regime = (
            "Bull Market 🟢 (BTC ยืนเหนือ 4H EMA200)"
            if btc_price > btc_ema200
            else "Bear Market 🔴 (BTC หลุด 4H EMA200)"
        )

    logger.info("Phase 4: Bulk Funding Rate (Binance)...")
    funding_map = {}
    try:
        pi_resp = api_session.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
        if pi_resp.status_code == 200:
            for item in pi_resp.json():
                if "symbol" in item and "lastFundingRate" in item:
                    funding_map[item["symbol"]] = float(item["lastFundingRate"])
    except Exception as e:
        logger.warning(f"Bulk Funding Rate error: {e}")

    logger.info("Phase 5: Scan all coins...")
    for coin in COINS:
        df        = all_4h.get(coin)
        df_daily  = all_1d.get(coin)

        if df is None or len(df) < EMA_LONG + 10:
            continue

        weekly_ctx     = analyze_weekly_context(df_daily) if df_daily is not None else _empty_weekly()
        monthly_ctx    = analyze_monthly_targets(df_daily) if df_daily is not None else _empty_monthly()
        cycle_ctx      = analyze_cycle_targets(df_daily)   if df_daily is not None else _empty_cycle()
        death_cross_ctx= check_death_cross_1d(df_daily)    if df_daily is not None else _empty_dc()

        df  = calculate_indicators(df)
        row = df.iloc[-1]

        current_price = row["close"]
        rsi           = row["RSI"]
        ema_50        = row["EMA_50"]
        ema_200       = row["EMA_200"]
        atr           = row["ATR"]
        adx           = row["ADX"]
        vol_confirmed = is_volume_confirmed(row)
        corr_btc      = calculate_correlation(df, btc_df) if coin != "BTC" else 1.0
        squeeze_warning = adx < 20

        atr_pct           = (atr / current_price) * 100 if current_price > 0 else 2.0
        tier              = COIN_TIER.get(coin, "mid")
        dynamic_multiplier = get_dynamic_atr_multiplier(tier, adx, atr_pct)
        trailing_stop_val = current_price - (atr * dynamic_multiplier)

        total_valid_coins += 1
        is_divergence     = check_bullish_divergence(df)
        rsi_rounded       = round(rsi, 2)

        trend_info  = analyze_trend_continuity(df)
        bounce_info = analyze_rsi_bounce(df)
        ob_info     = find_order_blocks(df)
        fvg_info    = find_fair_value_gaps(df)
        candle_info = confirm_reversal_candle(df)
        mtf_info    = get_mtf_rsi_alignment(df, df_daily)

        dynamic_tp_ob = estimate_price_for_target_rsi(df, target_rsi=70.0)

        fibo_4h_max  = df["high"].iloc[-60:].max()
        fibo_4h_min  = df["low"].iloc[-60:].min()
        fibo_4h_618  = fibo_4h_max - (0.618 * (fibo_4h_max - fibo_4h_min))

        tp1_pct = TP_TIERS[tier]["tp1"]
        tp2_pct = TP_TIERS[tier]["tp2"]
        sl_buf  = TP_TIERS[tier]["sl_buffer"]
        vol_tag = " 🔊" if vol_confirmed else ""

        in_fibo_zone    = weekly_ctx["fibo_618"] is not None and current_price <= weekly_ctx["fibo_618"] * 1.02
        in_4h_fibo_zone = current_price <= fibo_4h_618 * 1.01
        in_ob_zone      = ob_info["has_bullish_ob"] and current_price <= ob_info["bullish_ob_price"] * 1.03
        in_fvg_zone     = (fvg_info["has_fvg_support"] and
                           current_price <= fvg_info["fvg_top"] and
                           current_price >= fvg_info["fvg_bottom"] * 0.99)
        in_deep_support = False
        if weekly_ctx.get("fibo_786") is not None:
            if (current_price <= weekly_ctx["fibo_786"] * 1.02 or
                    (weekly_ctx.get("liquidity_pool") and
                     current_price <= weekly_ctx["liquidity_pool"] * 1.05) or
                    (weekly_ctx.get("psycho_support") and
                     current_price <= weekly_ctx["psycho_support"] * 1.02)):
                in_deep_support = True

        signal_type = ""

        if current_price > ema_200:
            coin_trend = "🟢 ขาขึ้น (Above EMA 200)"
            bullish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🟢 ขาขึ้น (RSI 4H: {rsi_rounded}) | {trend_info['trend_label']}"
            )
            if in_fibo_zone or in_4h_fibo_zone or in_ob_zone or in_fvg_zone:
                if current_price > (ema_50 * 0.98) and (rsi <= RSI_OVERSOLD or rsi <= RSI_PULLBACK_THRESHOLD):
                    if (bounce_info["quality"] in ["strong", "moderate"] and
                            candle_info["reversal_strength"] >= 1):
                        signal_type = f"Institution Dip & Rebound 📉{vol_tag}"
                    elif rsi <= RSI_OVERSOLD:
                        signal_type = f"Golden Fib / OB Zone Oversold 📉{vol_tag}"
                if is_divergence and not signal_type:
                    signal_type = f"Confluence Bullish Divergence 📈{vol_tag}"
                if ob_info["has_bullish_ob"] and not signal_type:
                    signal_type = f"Smart Money OB Reversal 🚀{vol_tag}"
        else:
            coin_trend = "🔴 ขาลง (Below EMA 200)"
            bearish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🔴 ขาลง (RSI 4H: {rsi_rounded}) | {trend_info['trend_label']}"
            )
            if in_deep_support and is_divergence:
                signal_type = f"🚨 DEEP REVERSAL + Bullish Div 🐳{vol_tag}"
            elif in_deep_support and bounce_info["quality"] == "strong":
                signal_type = f"🛡️ Deep Support Strong Bounce 📉{vol_tag}"
            elif in_fibo_zone or in_ob_zone:
                if rsi <= RSI_OVERSOLD:
                    signal_type = f"Deep Retracement Buy (เสี่ยงสูง) 📉{vol_tag}"
                elif is_divergence:
                    signal_type = f"Macro Support Divergence 📈{vol_tag}"

        if signal_type:
            sig_score, score_label = calculate_signal_score(
                rsi=rsi, bounce_info=bounce_info, candle_info=candle_info,
                vol_confirmed=vol_confirmed, in_fibo_zone=(in_fibo_zone or in_4h_fibo_zone),
                in_ob_zone=in_ob_zone, in_fvg_zone=in_fvg_zone,
                weekly_ctx=weekly_ctx, mtf_info=mtf_info, adx=adx,
                is_divergence=is_divergence, trend_info=trend_info,
            )
            if sig_score < MINIMUM_SIGNAL_SCORE:
                signal_type = ""  # กรองออก

        if signal_type:
            if coin in ONCHAIN_SUPPORTED_COINS:
                onchain_ctx = analyze_onchain_momentum(coin)
            else:
                onchain_ctx = {
                    "onchain_warning": False,
                    "onchain_label": "🔹 On-Chain: ไม่มีข้อมูลสำหรับเหรียญนี้",
                    "large_tx_change_pct": 0.0,
                }

            futures_ctx = analyze_futures_context(coin, funding_map.get(f"{coin}USDT"))

            if onchain_ctx.get("onchain_warning"):
                signal_type = f"⚠️ On-Chain เสี่ยงสูง + {signal_type}"
            if weekly_ctx.get("weekly_bullish_div"):
                signal_type = f"⭐ {signal_type} + [1W Bullish Div]"
            elif weekly_ctx.get("rsi_weekly") and weekly_ctx["rsi_weekly"] <= 35:
                signal_type = f"💎 {signal_type} + [1W Oversold Zone]"
            elif monthly_ctx.get("monthly_trend") == "bullish" and in_fvg_zone:
                signal_type = f"🔥 {signal_type} + [1M Trend + FVG]"

            entry_min    = format_price(current_price * 0.98)
            entry_max    = format_price(current_price * 1.01)
            target_tp1   = current_price * (1 + tp1_pct)
            target_tp2   = current_price * (1 + tp2_pct)

            sl_reference = ema_200
            if in_deep_support and weekly_ctx.get("fibo_886"):
                sl_reference = weekly_ctx["fibo_886"]
            elif ob_info["has_bullish_ob"]:
                sl_reference = ob_info["bullish_ob_price"]
            elif fvg_info["has_fvg_support"]:
                sl_reference = fvg_info["fvg_bottom"]

            sl_val = (sl_reference * (1 - sl_buf)
                      if current_price > sl_reference
                      else current_price * (1 - sl_buf))

            sl_distance_pct = max((current_price - sl_val) / current_price, 0.01)

            active_signal_count += 1
            pos_size = get_correlation_adjusted_position(
                portfolio=PORTFOLIO_USDT, risk_pct=RISK_PER_TRADE_PCT,
                sl_distance_pct=sl_distance_pct, corr_btc=corr_btc,
                active_signals_count=active_signal_count - 1,
            )

            buy_signals.append({
                "coin": coin, "trend": coin_trend,
                "price": format_price(current_price), "rsi": rsi_rounded,
                "type": signal_type,
                "ema_50": format_price(ema_50), "ema_200": format_price(ema_200),
                "entry":  f"${entry_min} - ${entry_max}",
                "tp1":    f"${format_price(target_tp1)} (+{tp1_pct*100:.0f}%)",
                "tp2":    f"${format_price(target_tp2)} (+{tp2_pct*100:.0f}%)",
                "dynamic_tp":    f"${format_price(dynamic_tp_ob)}",
                "trailing_stop": f"${format_price(max(trailing_stop_val, sl_val))}",
                "sl":            f"${format_price(sl_val)}",
                "vol_confirmed": vol_confirmed,
                "trend_info":    trend_info,
                "bounce_info":   bounce_info,
                "ob_info":       ob_info,
                "fvg_info":      fvg_info,
                "candle_info":   candle_info,
                "mtf_info":      mtf_info,
                "sig_score":     sig_score,
                "score_label":   score_label,
                "weekly_ctx":    weekly_ctx,
                "monthly_ctx":   monthly_ctx,
                "cycle_ctx":     cycle_ctx,
                "onchain_ctx":   onchain_ctx,
                "death_cross_ctx": death_cross_ctx,
                "futures_ctx":   futures_ctx,
                "corr_btc":      corr_btc,
                "squeeze_warning": squeeze_warning,
                "adx":           round(adx, 2),
                "atr_pct":       round(atr_pct, 2),
                "dyn_multiplier": dynamic_multiplier,
                "pos_size":      f"${pos_size:.2f}",
                "sl_risk_pct":   f"{sl_distance_pct*100:.1f}%",
            })

        if rsi >= RSI_OVERBOUGHT or ob_info["has_bearish_ob"]:
            tp_min  = format_price(current_price * 1.00)
            tp_max  = format_price(current_price * (1 + tp1_pct * 0.4))
            exit_val = ema_50 if current_price > ema_50 else current_price * (1 - sl_buf)
            sell_signals.append({
                "coin": coin, "trend": coin_trend,
                "price": format_price(current_price), "rsi": rsi_rounded,
                "tp_zone":     f"${tp_min} - ${tp_max}",
                "exit":        f"${format_price(exit_val)}",
                "vol_confirmed": vol_confirmed,
                "trend_info":  trend_info,
                "ob_info":     ob_info,
                "weekly_ctx":  weekly_ctx,
                "cycle_ctx":   cycle_ctx,
            })

    if total_valid_coins > 0:
        bullish_ratio = (bullish_coins / total_valid_coins) * 100
        summary_msg   = (
            f"📊 <b>[Market Summary v2]</b>\n"
            f"ดัชนีหลัก (BTC): <b>{market_regime}</b>\n"
            f"ทุนระบบ: {PORTFOLIO_USDT} USDT | Risk {RISK_PER_TRADE_PCT}%/ไม้ | Max {MAX_TOTAL_RISK_PCT}%/รอบ\n"
            f"📈 ขาขึ้น: {bullish_coins} | 📉 ขาลง: {bearish_coins} เหรียญ\n"
            f"🔎 Signal Threshold: {MINIMUM_SIGNAL_SCORE}/100\n"
        )
        if bullish_ratio >= 65:
            summary_msg += "🔥 ภาพรวม: <b>🟢 Strong Bullish</b>\n<i>กลยุทธ์: ดักย่อที่ Confluence Zone</i>"
        elif bullish_ratio >= 40:
            summary_msg += "🔥 ภาพรวม: <b>🟡 Sideways</b>\n<i>กลยุทธ์: รอราคาลงแนวรับ Fibonacci 61.8%</i>"
        else:
            summary_msg += "🔥 ภาพรวม: <b>🔴 Bearish</b>\n<i>กลยุทธ์: ระวังสูง เฉพาะโซนรับ 1W เท่านั้น</i>"
        summary_msg += "\n\n📋 <b>สรุปแนวโน้มรายเหรียญ:</b>\n" + "\n".join(coin_trends_summary)
    else:
        summary_msg = "⚠️ ไม่สามารถดึงข้อมูลเหรียญได้"

    return buy_signals, sell_signals, summary_msg

# ==========================================
# Fallback helpers
# ==========================================
def _empty_weekly(): return {
    "rsi_weekly": None, "weekly_bullish_div": False,
    "weekly_status_label": "↔️ ไม่พบข้อมูลสัปดาห์",
    "fibo_618": None, "fibo_786": None, "fibo_886": None,
    "liquidity_pool": None, "psycho_support": None,
}
def _empty_monthly(): return {
    "m_resistance_target": None, "m_support_target": None,
    "monthly_summary_label": "⏳ ไม่มีข้อมูลเดือน", "monthly_trend": "sideways",
}
def _empty_cycle(): return {
    "cycle_target_zone": None, "cycle_confluence_factors": [],
    "cycle_summary_label": "⏳ ไม่มีข้อมูลไซเคิล",
}
def _empty_dc(): return {
    "has_death_cross": False, "death_cross_label": "⚪ 1D Death Cross: ปลอดภัย",
}

# ==========================================
# Message Builder
# ==========================================
def build_messages(buy_list: list, sell_list: list, market_summary: str) -> list:
    blocks = [market_summary]

    if buy_list:
        header = "🎯 <b>[Crypto Screener 4H v2 – สัญญาณช้อนซื้อ]</b>"
        current = header

        for opt in buy_list:
            vol_note   = ("\n🔊 Volume: <b>ยืนยัน (สูงกว่า MA20)</b>"
                          if opt["vol_confirmed"] else "\n🔇 Volume: ไม่ยืนยัน")
            corr_alert = (
                f"\n⚠️ <b>Correlation BTC:</b> {opt['corr_btc']} (สูงมาก – กระจายความเสี่ยงด้วย)"
                if opt.get("corr_btc", 0) > 0.85 and opt["coin"] != "BTC" else ""
            )
            time_alert = (
                f"\n⏳ <b>ADX ต่ำ ({opt.get('adx',0)}):</b> ตลาด Sideways – อาจต้องถือรอนาน"
                if opt.get("squeeze_warning") else ""
            )
            ti   = opt["trend_info"]
            bi   = opt["bounce_info"]
            ob   = opt["ob_info"]
            fvg  = opt["fvg_info"]
            ci   = opt["candle_info"]
            mtfi = opt["mtf_info"]
            w_ctx= opt.get("weekly_ctx",  {})
            m_ctx= opt.get("monthly_ctx", {})
            c_ctx= opt.get("cycle_ctx",   {})
            oc   = opt.get("onchain_ctx", {})
            dc   = opt.get("death_cross_ctx", {})
            ft   = opt.get("futures_ctx", {})

            confluence = "\n🛡️ <b>แนวรับสถาบัน:</b>"
            if w_ctx.get("fibo_618"):
                confluence += (
                    f"\n   🔹 Fibo 1W (61.8%): <code>${format_price(w_ctx['fibo_618'])}</code>"
                    f"\n   🔸 Fibo 1W (78.6%): <code>${format_price(w_ctx['fibo_786'])}</code>"
                    f"\n   🔻 Deep Support:"
                    f"\n      - Fibo 88.6%: <code>${format_price(w_ctx.get('fibo_886'))}</code>"
                    f"\n      - Liquidity Pool: <code>${format_price(w_ctx.get('liquidity_pool'))}</code>"
                    f"\n      - Psycho Support: <code>${format_price(w_ctx.get('psycho_support'))}</code>"
                )
            if fvg.get("has_fvg_support"):
                confluence += (
                    f"\n   ⚡ FVG (4H): <code>${format_price(fvg['fvg_bottom'])} - "
                    f"${format_price(fvg['fvg_top'])}</code>"
                )
            if ob.get("has_bullish_ob"):
                confluence += f"\n   🐳 OB Support: <code>${format_price(ob['bullish_ob_price'])}</code>"

            candle_block = f"\n{ci['reversal_label']}" if ci.get("reversal_strength", 0) > 0 else ""
            mtf_block    = f"\n📡 <b>MTF RSI:</b> {mtfi['mtf_label']}" if mtfi.get("mtf_label") else ""
            score_block  = f"\n🏆 <b>Signal Score:</b> {opt['score_label']}"
            trend_block  = f"\n📐 <b>แนวโน้ม (4H):</b> {ti['trend_label']}"
            bounce_block = (
                f"\n🔄 <b>RSI Bounce:</b> {bi['quality_label']}"
                + (f"\n   {bi['entry_timing']}" if bi["entry_timing"] else "")
            )
            atr_block    = (
                f"\n📏 <b>ATR SL Multiplier:</b> {opt['dyn_multiplier']}x "
                f"(ATR {opt['atr_pct']}% ของราคา)"
            )
            weekly_block = (
                f"\n🗓️ <b>ภาพสัปดาห์ (1W):</b> {w_ctx['weekly_status_label']}"
                if w_ctx and w_ctx.get("rsi_weekly") else ""
            )
            onchain_block = (
                f"\n📊 <b>On-Chain:</b>\n   {oc.get('onchain_label','')}"
                if oc else ""
            )
            dc_block = (
                f"\n   {dc.get('death_cross_label','')}"
                if dc and dc.get("has_death_cross") else ""
            )
            ft_block = (
                f"\n   {ft.get('futures_label','')}" if ft else ""
            )
            monthly_block = (
                f"\n🔮 <b>กรอบ 1M:</b>"
                f"\n   🔼 เป้าขึ้น: <code>${format_price(m_ctx['m_resistance_target'])}</code>"
                f"\n   🔽 แนวรับ: <code>${format_price(m_ctx['m_support_target'])}</code>"
                if m_ctx and m_ctx.get("m_resistance_target") else ""
            )
            cycle_block = (
                f"\n{c_ctx['cycle_summary_label']}"
                if c_ctx and c_ctx.get("cycle_target_zone") else ""
            )

            coin_msg = (
                f"\n\n🪙 <b>{opt['coin']}</b>"
                f"\n📊 เทรนด์: {opt['trend']}"
                f"\n🚨 รูปแบบ: <b>{opt['type']}</b>"
                f"\n💵 ราคา: ${opt['price']}"
                f"\n📉 RSI (4H): {opt['rsi']}"
                f"{vol_note}{corr_alert}{time_alert}"
                f"{score_block}"
                f"{mtf_block}"
                f"{candle_block}"
                f"{confluence}"
                f"{trend_block}{bounce_block}"
                f"{atr_block}"
                f"{weekly_block}{onchain_block}{dc_block}{ft_block}"
                f"{monthly_block}{cycle_block}"
                f"\n\n🛡️ <b>[Risk Management]</b>"
                f"\n💼 Position Size: <code>{opt['pos_size']}</code>"
                f"\n📉 ระยะ SL: {opt['sl_risk_pct']}"
                f"\n❌ Hard SL: <code>{opt['sl']}</code>"
                f"\n\n🚀 <b>[Take Profit]</b>"
                f"\nTP1 (Fix): <code>{opt['tp1']}</code>"
                f"\nTP2 (Fix): <code>{opt['tp2']}</code>"
                f"\n🔥 Dynamic TP (RSI=70): <code>{opt['dynamic_tp']}</code>"
                f"\n🔗 Trailing Stop: <code>{opt['trailing_stop']}</code>"
            )

            if len(current) + len(coin_msg) > 3500:
                blocks.append(current)
                current = header + coin_msg
            else:
                current += coin_msg
        blocks.append(current)

    if sell_list:
        header  = "⚠️ <b>[Crypto Screener 4H v2 – โซนทำกำไร / แนวต้าน]</b>"
        current = header
        for opt in sell_list:
            vol_note = (
                "\n🔊 Volume: <b>ยืนยันแรงซื้อ (ระวังพักตัว)</b>"
                if opt["vol_confirmed"] else "\n🔇 Volume: ปกติ"
            )
            ti   = opt["trend_info"]
            ob   = opt["ob_info"]
            w_ctx= opt.get("weekly_ctx", {})
            c_ctx= opt.get("cycle_ctx",  {})

            coin_msg = (
                f"\n\n🪙 <b>{opt['coin']}</b>"
                f"\n💵 ราคา: ${opt['price']}"
                f"\n📈 RSI (4H): {opt['rsi']}"
                f"{vol_note}"
                f"\n📐 แนวโน้ม: {ti['trend_label']}"
                + (f"\n🚨 Bearish OB: ${format_price(ob['bearish_ob_price'])}"
                   if ob.get("has_bearish_ob") else "")
                + (f"\n🗓️ ภาพ 1W: {w_ctx['weekly_status_label']}"
                   if w_ctx and w_ctx.get("rsi_weekly") else "")
                + (f"\n{c_ctx['cycle_summary_label']}"
                   if c_ctx and c_ctx.get("cycle_target_zone") else "")
                + f"\n🔴 ช่วงราคาขาย: <code>{opt['tp_zone']}</code>"
                + f"\n❌ Safety Exit: <code>{opt['exit']}</code>"
            )

            if len(current) + len(coin_msg) > 3500:
                blocks.append(current)
                current = header + coin_msg
            else:
                current += coin_msg
        blocks.append(current)

    if not buy_list and not sell_list:
        blocks.append(
            "\n=========================\n"
            "😴 <i>ตลาดนิ่ง: ไม่มีเหรียญผ่าน Signal Score Threshold ในรอบนี้</i>"
        )

    return blocks

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Crypto Screener v2 – Starting (Optimized + Parameter Auth)")
    logger.info("=" * 60)

    buy_list, sell_list, market_summary = scan_market()
    logger.info(
        f"สแกนเสร็จ → Buy: {len(buy_list)} | Sell: {len(sell_list)}"
    )

    final_messages = build_messages(buy_list, sell_list, market_summary)
    send_telegram_messages(final_messages)

    logger.info("ส่ง Telegram สำเร็จ!")
