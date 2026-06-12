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
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = str(os.getenv("CRYPTOCOMPARE_API_KEY") or "").strip()

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

# [B] Binance Endpoints
BINANCE_ENDPOINTS = [
    "https://api.binance.us",           
    "https://data-api.binance.vision",  
    "https://api.binance.com",          
    "https://api3.binance.com",         
    "https://api1.binance.com",
    "https://api2.binance.com",
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

# รายชื่อเหรียญที่รองรับการดึง On-chain (Blockchain Histo) จาก CryptoCompare
ONCHAIN_SUPPORTED_COINS = {"BTC", "ETH", "ADA", "DOGE", "LTC", "BCH", "LINK"}

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
    "User-Agent": "CryptoScreenerBot/3.9 (Multi-Source + OnChain)",
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
_cache_4h:       dict  = {}
_cache_1d:       dict  = {}
_cache_onchain:  dict  = {}
_cache_ts_4h:    float = 0.0
_cache_ts_1d:    float = 0.0
_cache_ts_onchain: float = 0.0

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
# Data Fetching
# ==========================================
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
# Bulk Cache Fetchers
# ==========================================
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

def calculate_signal_score(rsi, bounce_info, candle_info, vol_confirmed, in_fibo_zone, in_ob_zone, in_fvg_zone, weekly_ctx, mtf_info, adx, is_divergence, trend_info, onchain_info) -> tuple[int, str]:
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
    
    score = max(0, min(100, score))
    grade = "🔥 A+" if score >= 70 else "✅ A" if score >= 55 else "🟡 B" if score >= MINIMUM_SIGNAL_SCORE else "⬜ C"
    return score, f"{grade} | Score: {score}/100"

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

# ==========================================
# Market Scanner
# ==========================================
def scan_market():
    buy_signals, sell_signals, coin_trends_summary = [], [], []
    bullish_coins, bearish_coins, active_signal_count = 0, 0, 0

    logger.info("Phase 1: Bulk fetch 4H (A+B+C)...")
    all_4h = bulk_fetch_4h(COINS)
    logger.info("Phase 2: Bulk fetch 1D (A+B+C)...")
    all_1d = bulk_fetch_1d(COINS)
    logger.info("Phase 3: Bulk fetch On-chain (1M)...")
    all_onchain = bulk_fetch_onchain(COINS)

    btc_df = all_4h.get("BTC")
    market_regime = "Unknown ⚪"
    if btc_df is not None:
        btc_df = calculate_indicators(btc_df)
        market_regime = "Bull Market 🟢" if btc_df["close"].iloc[-1] > btc_df["EMA_200"].iloc[-1] else "Bear Market 🔴"

    for coin in COINS:
        df, df_daily = all_4h.get(coin), all_1d.get(coin)
        onchain_info = all_onchain.get(coin, {"has_data": False, "onchain_label": "⚪ N/A"})
        
        if df is None or len(df) < EMA_LONG + 10: continue

        weekly_ctx = analyze_weekly_context(df_daily)
        df = calculate_indicators(df)
        row = df.iloc[-1]

        price, rsi, ema50, ema200 = row["close"], row["RSI"], row["EMA_50"], row["EMA_200"]
        atr, adx, vol_confirmed = row["ATR"], row["ADX"], is_volume_confirmed(row)
        
        tier = COIN_TIER.get(coin, "mid")
        dyn_mult = get_dynamic_atr_multiplier(tier, adx, (atr / price) * 100)
        
        trend_info, bounce_info = analyze_trend_continuity(df), analyze_rsi_bounce(df)
        ob_info, fvg_info = find_order_blocks(df), find_fair_value_gaps(df)
        candle_info, mtf_info = confirm_reversal_candle(df), get_mtf_rsi_alignment(df, df_daily)
        is_div = check_bullish_divergence(df)

        in_fibo = weekly_ctx["fibo_618"] is not None and price <= weekly_ctx["fibo_618"] * 1.02
        in_ob = ob_info["has_bullish_ob"] and price <= ob_info["bullish_ob_price"] * 1.03
        in_fvg = fvg_info["has_fvg_support"] and fvg_info["fvg_bottom"] * 0.99 <= price <= fvg_info["fvg_top"]
        
        signal_type = ""
        if price > ema200:
            bullish_coins += 1
            coin_trends_summary.append(f"• {coin}: 🟢 ขาขึ้น | {trend_info['trend_label']}")
            if (in_fibo or in_ob or in_fvg) and price > (ema50 * 0.98) and rsi <= 55:
                if bounce_info["quality"] in ["strong", "moderate"]: signal_type = "Institution Dip & Rebound 📉"
            if is_div and not signal_type: signal_type = "Confluence Bullish Divergence 📈"
        else:
            bearish_coins += 1
            coin_trends_summary.append(f"• {coin}: 🔴 ขาลง | {trend_info['trend_label']}")
            if weekly_ctx.get("fibo_786") and price <= weekly_ctx["fibo_786"] * 1.02:
                if is_div: signal_type = "🚨 DEEP REVERSAL + Bullish Div 🐳"
                elif bounce_info["quality"] == "strong": signal_type = "🛡️ Deep Support Strong Bounce 📉"

        if signal_type:
            sig_score, score_label = calculate_signal_score(
                rsi, bounce_info, candle_info, vol_confirmed, 
                in_fibo, in_ob, in_fvg, weekly_ctx, mtf_info, adx, is_div, trend_info, onchain_info
            )
            
            if sig_score >= MINIMUM_SIGNAL_SCORE:
                tp1_pct, tp2_pct, sl_buf = TP_TIERS[tier]["tp1"], TP_TIERS[tier]["tp2"], TP_TIERS[tier]["sl_buffer"]
                sl_ref = ob_info["bullish_ob_price"] if ob_info["has_bullish_ob"] else fvg_info["fvg_bottom"] if fvg_info["has_fvg_support"] else ema200
                sl_val = (sl_ref * (1 - sl_buf) if price > sl_ref else price * (1 - sl_buf))
                sl_dist = max((price - sl_val) / price, 0.01)
                
                corr_btc = 0.5
                if btc_df is not None and not df.empty:
                    corr_val = df["close"].corr(btc_df["close"])
                    if not pd.isna(corr_val):
                        corr_btc = corr_val
                
                pos_size = get_correlation_adjusted_position(PORTFOLIO_USDT, RISK_PER_TRADE_PCT, sl_dist, corr_btc, active_signal_count)
                active_signal_count += 1

                buy_signals.append({
                    "coin": coin, "price": format_price(price), "rsi": round(rsi, 2),
                    "type": signal_type, "score_label": score_label,
                    "tp1": f"${format_price(price * (1 + tp1_pct))}", 
                    "tp2": f"${format_price(price * (1 + tp2_pct))}",
                    "dynamic_tp": f"${format_price(estimate_price_for_target_rsi(df) or price * 1.1)}",
                    "sl": f"${format_price(sl_val)}", "pos_size": f"${pos_size:.2f}",
                    "sl_risk_pct": f"{sl_dist*100:.1f}%", "vol_confirmed": vol_confirmed,
                    "mtf_label": mtf_info.get("mtf_label", ""), "reversal_label": candle_info["reversal_label"],
                    "trend_label": trend_info["trend_label"], "bounce_label": bounce_info["quality_label"],
                    "onchain_label": onchain_info["onchain_label"]
                })

    summary_msg = f"📊 <b>[Market Summary – v3.9 On-Chain]</b>\nดัชนีหลัก (BTC): <b>{market_regime}</b>\n📈 ขาขึ้น: {bullish_coins} | 📉 ขาลง: {bearish_coins}\n\n" + "\n".join(coin_trends_summary)
    return buy_signals, sell_signals, summary_msg

def build_messages(buy_list, sell_list, market_summary) -> list:
    blocks = [market_summary]
    if buy_list:
        current = "🎯 <b>[Crypto Screener 4H – สัญญาณซื้อ]</b>"
        for opt in buy_list:
            vol = "🔊 Volume ยืนยัน" if opt["vol_confirmed"] else "🔇 Volume ต่ำ"
            msg = (
                f"\n\n🪙 <b>{opt['coin']}</b> | ราคา: ${opt['price']}\n"
                f"🚨 สัญญาณ: <b>{opt['type']}</b> | RSI: {opt['rsi']}\n"
                f"{vol}\n"
                f"🏆 {opt['score_label']}\n"
                f"📡 MTF: {opt['mtf_label']}\n"
                f"{opt['reversal_label']}\n"
                f"📐 แนวโน้ม: {opt['trend_label']}\n"
                f"🔄 RSI Bounce: {opt['bounce_label']}\n"
                f"🔗 On-Chain (1M): {opt['onchain_label']}\n\n"
                f"💼 Position: <code>{opt['pos_size']}</code> | SL ระยะ: {opt['sl_risk_pct']}\n"
                f"❌ Hard SL: <code>{opt['sl']}</code>\n"
                f"🚀 TP1: <code>{opt['tp1']}</code> | TP2: <code>{opt['tp2']}</code>\n"
                f"🔥 Dynamic TP: <code>{opt['dynamic_tp']}</code>"
            )
            if len(current) + len(msg) > 3500: blocks.append(current); current = "🎯 <b>[ต่อ]</b>" + msg
            else: current += msg
        blocks.append(current)
    if not buy_list: blocks.append("😴 <i>ตลาดนิ่ง: ไม่มีสัญญาณในรอบนี้</i>")
    return blocks

if __name__ == "__main__":
    try:
        ip = api_session.get("https://ifconfig.me", timeout=5).text.strip()
        logger.info(f"🌐 Network IP: {ip} (Proxy Active: {bool(PROXIES)})")
    except Exception: pass
    
    buy_list, sell_list, market_summary = scan_market()
    send_telegram_messages(build_messages(buy_list, sell_list, market_summary))
    logger.info("ระบบทำงานสมบูรณ์!")
