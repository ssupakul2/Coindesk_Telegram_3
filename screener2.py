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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

PORTFOLIO_USDT = 1500.0   
RISK_PER_TRADE_PCT = 2.0  

COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "FLOKI", "SHIB", "EIGEN", "OP", "DOGE", "NEAR",
    "TRX", "AVAX", "SUI"
]

# ==========================================
# Constants & Hyperparameters
# ==========================================
API_RATE_LIMIT_DELAY = 0.35
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2.0
HISTOHOUR_LIMIT = 2000

# --- Indicators ---
RSI_PERIOD = 14
EMA_SHORT = 50
EMA_LONG = 200
RSI_OVERSOLD = 32
RSI_OVERBOUGHT = 70
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5 

# --- RSI Recovery & Pullback Configuration ---
RSI_RECOVERY_THRESHOLD = 45
RSI_PULLBACK_THRESHOLD = 55
RSI_RECOVERY_LOOKBACK = 5

# --- Divergence Configuration ---
RSI_BULL_DIV_MAX = 45
RSI_BEAR_DIV_MIN = 55
LOOKBACK_BARS = 15
LOOKBACK_SKIP_BARS = 3

# --- Trend Continuity Configuration ---
TREND_SLOPE_BARS = 5          
TREND_MIN_CONSECUTIVE = 3     

# --- RSI Bounce Configuration ---
RSI_BOUNCE_CONFIRM_BARS = 2   
RSI_BOUNCE_MIN_RISE = 3.0     

# --- Order Block (SMC) & FVG Configuration ---
OB_LOOKBACK = 20              
OB_IMBALANCE_RATIO = 1.5      
FVG_THRESHOLD_PCT = 0.2       

# --- Take Profit Tiers ---
TP_TIERS = {
    "major":  {"tp1": 0.10, "tp2": 0.15, "sl_buffer": 0.025},
    "mid":    {"tp1": 0.15, "tp2": 0.20, "sl_buffer": 0.050},
    "small":  {"tp1": 0.20, "tp2": 0.35, "sl_buffer": 0.080},
}

COIN_TIER = {
    "BTC": "major", "ETH": "major",
    "BNB": "mid",   "SOL": "mid",   "XRP": "mid",
    "ADA": "mid",   "NEAR": "mid",  "OP": "mid",
    "TRX": "mid",   "AVAX": "mid",
    "FLOKI": "small","SHIB": "small","EIGEN": "small","DOGE": "small",
    "SUI": "small"
}

# ==========================================
# Optimization 1: Global API Session Setup & Anti-Bot Headers
# ==========================================
api_session = requests.Session()
api_session.headers.update({
    "User-Agent": "CryptoScreenerBot/1.0 (Mozilla/5.0; Trading API System)",
    "Accept": "application/json",
    "Authorization": f"Apikey {str(CRYPTOCOMPARE_API_KEY or '').strip()}"
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
# Telegram Integration
# ==========================================
def send_telegram_messages(chunks: list) -> None:
    token = str(TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = str(TELEGRAM_CHAT_ID or "").strip()

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID ไม่ได้ตั้งค่าใน Environment Variables")
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
# Data Fetching & Core Technical Analysis
# ==========================================
def get_historical_data(coin: str) -> pd.DataFrame | None:
    """ดึงข้อมูล 4H จาก CryptoCompare"""
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": HISTOHOUR_LIMIT,
    }

    try:
        resp = api_session.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            df_4h = df.resample("4h").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volumeto": "sum",
                }
            ).dropna()
            return df_4h
        else:
            logger.warning(f"{coin} 4H: API ตอบกลับผิดปกติ – {data.get('Message')}")
    except Exception as e:
        logger.warning(f"{coin} 4H: {e}")

    return None

def get_histoday_data(coin: str) -> pd.DataFrame | None:
    """ดึงข้อมูล 1D แกนกลางเพียงครั้งเดียวเพื่อนำไปแจกจ่ายให้ฟังก์ชันย่อย"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": 2000,
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
            logger.warning(f"{coin} 1D: API ตอบกลับผิดปกติ – {data.get('Message')}")
    except Exception as e:
        logger.warning(f"{coin} 1D: {e}")
    return None

# ==========================================
# Optimization 2: Refactored Sub-functions to accept 1D DataFrame
# ==========================================
def analyze_weekly_context(df_1d: pd.DataFrame) -> dict:
    result = {
        "rsi_weekly": None,
        "weekly_bullish_div": False,
        "weekly_status_label": "↔️ ไม่พบข้อมูลระบุระดับสัปดาห์ชัดเจน",
        "fibo_618": None,
        "fibo_786": None,
        "fibo_886": None,
        "liquidity_pool": None,
        "psycho_support": None
    }

    if df_1d is None or len(df_1d) < RSI_PERIOD + LOOKBACK_BARS + 5:
        return result

    try:
        df_w = df_1d.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

        if len(df_w) < RSI_PERIOD + LOOKBACK_BARS + 5:
            return result

        # --- Macro Fibonacci & Liquidity Pools ---
        fibo_window = df_w.iloc[-104:]
        w_max = fibo_window["high"].max()
        w_min = fibo_window["low"].min() 
        w_diff = w_max - w_min
        
        result["fibo_618"] = w_max - (0.618 * w_diff)
        result["fibo_786"] = w_max - (0.786 * w_diff)
        result["fibo_886"] = w_max - (0.886 * w_diff)
        result["liquidity_pool"] = w_min

        # --- Psychological Support ---
        curr_price = df_w["close"].iloc[-1]
        if curr_price > 0:
            magnitude = 10 ** math.floor(math.log10(curr_price))
            step = magnitude if curr_price >= magnitude * 2 else magnitude / 2
            result["psycho_support"] = math.floor(curr_price / step) * step

        # --- RSI & Divergence Calculation ---
        close = df_w["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_w["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

        curr_rsi_w = round(df_w["RSI"].iloc[-1], 2)
        result["rsi_weekly"] = curr_rsi_w

        prev_window = df_w.iloc[-(LOOKBACK_BARS + 1) : -(LOOKBACK_SKIP_BARS)]
        if len(prev_window) > 0:
            min_low_idx = prev_window["low"].argmin()
            prev_low_price = prev_window["low"].iloc[min_low_idx]
            prev_low_rsi   = prev_window["RSI"].iloc[min_low_idx]
            curr_price_w = df_w["low"].iloc[-1]
            curr_rsi_w_now = df_w["RSI"].iloc[-1]

            if (prev_low_rsi <= RSI_BULL_DIV_MAX) and (curr_price_w < prev_low_price) and (curr_rsi_w_now > prev_low_rsi):
                result["weekly_bullish_div"] = True

        if result["weekly_bullish_div"]:
            result["weekly_status_label"] = f"👑 <b>เกิด Weekly Bullish Divergence ในระดับภาพใหญ่!</b> (RSI: {curr_rsi_w})"
        elif curr_rsi_w <= RSI_OVERSOLD:
            result["weekly_status_label"] = f"🔥 <b>ภาพใหญ่เข้าเขต Oversold รุนแรง ({curr_rsi_w})</b>"
        elif curr_price <= result["fibo_786"]:
            result["weekly_status_label"] = f"⚠️ ราคาลงมาทดสอบโซน Deep Discount (ใต้ Fibo 78.6%)"
        elif curr_rsi_w >= RSI_OVERBOUGHT:
            result["weekly_status_label"] = f"⚠️ ภาพใหญ่เกิด Overbought ({curr_rsi_w}) ระวังความเสี่ยงการปรับฐาน"
        else:
            result["weekly_status_label"] = f"↔️ ภาพใหญ่ทรงตัวปกติ (RSI 1W: {curr_rsi_w})"

    except Exception as e:
        logger.error(f"Error calculating Weekly Context from 1D Data: {e}")
        
    return result

def analyze_monthly_targets(df_1d: pd.DataFrame) -> dict:
    result = {
        "m_resistance_target": None, "m_support_target": None,
        "monthly_summary_label": "⏳ ไม่สามารถคำนวณเป้าหมายระดับเดือนได้", "monthly_trend": "sideways"
    }

    if df_1d is None or len(df_1d) < 30: 
        return result

    try:
        df_m = df_1d.resample("ME").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

        if len(df_m) < 5: return result

        last_month = df_m.iloc[-2]
        current_month = df_m.iloc[-1]
        m_high, m_low, m_close = last_month["high"], last_month["low"], last_month["close"]
        
        pivot = (m_high + m_low + m_close) / 3
        r1 = (2 * pivot) - m_low
        s1 = (2 * pivot) - m_high
        
        if len(df_m) >= 12:
            df_m["EMA_12"] = df_m["close"].ewm(span=12, adjust=False).mean()
            m_ema12 = df_m["EMA_12"].iloc[-1]
        else:
            m_ema12 = pivot

        curr_price = current_month["close"]
        
        if curr_price >= m_ema12:
            target_up, target_down, trend_status = max(r1, m_high), pivot, "bullish"
            status_text = "🔮 <b>ภาพ 1M (Bullish):</b> ทิศทางหลักเป็นขาขึ้น มีเป้าหมายราคาวิ่งทดสอบกรอบบน"
        else:
            target_up, target_down, trend_status = pivot, min(s1, m_low), "bearish"
            status_text = "🔮 <b>ภาพ 1M (Bearish):</b> ทิศทางหลักเป็นขาลง/พักฐาน มีแนวโน้มไหลลงหาแนวรับกรอบล่าง"

        result.update({"m_resistance_target": target_up, "m_support_target": target_down, "monthly_summary_label": status_text, "monthly_trend": trend_status})
    except Exception:
        pass
    return result

def analyze_cycle_targets(df_1d: pd.DataFrame) -> dict:
    result = {"cycle_target_zone": None, "cycle_confluence_factors": [], "cycle_summary_label": "⏳ ไม่สามารถวิเคราะห์เป้าหมายไซเคิลได้"}

    if df_1d is None or len(df_1d) < 150: 
        return result

    try:
        df_w = df_1d.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

        if len(df_w) < 104: return result

        curr_price = df_w["close"].iloc[-1]
        df_w["EMA_50"] = df_w["close"].ewm(span=50, adjust=False).mean()
        df_w["EMA_200"] = df_w["close"].ewm(span=200, adjust=False).mean()
        w_ema50, w_ema200 = df_w["EMA_50"].iloc[-1], df_w["EMA_200"].iloc[-1]

        macro_window = df_w.iloc[-156:] if len(df_w.iloc[-156:]) > 0 else df_w
        macro_high, macro_low = macro_window["high"].max(), macro_window["low"].min()
        
        fib_1618_ext = macro_low + ((macro_high - macro_low) * 1.618)
        fib_0618_ret = macro_low + ((macro_high - macro_low) * 0.618)

        potential_targets = []
        if macro_high > curr_price * 1.1: potential_targets.append(("Historical Resistance (Previous Top)", macro_high))
        if fib_0618_ret > curr_price * 1.1: potential_targets.append(("Macro Fibonacci 61.8% (Golden Pocket)", fib_0618_ret))
        if fib_1618_ext > curr_price * 1.1: potential_targets.append(("Fibonacci Extension 161.8%", fib_1618_ext))
        if w_ema50 > curr_price * 1.05: potential_targets.append(("Weekly EMA 50 (Dynamic Resistance)", w_ema50))
        if w_ema200 > curr_price * 1.05: potential_targets.append(("Weekly EMA 200 (Macro Trendline)", w_ema200))

        potential_targets.sort(key=lambda x: x[1])
        confluence_zones = []
        
        for i in range(len(potential_targets)):
            cluster = [potential_targets[i]]
            for j in range(i + 1, len(potential_targets)):
                if (potential_targets[j][1] - cluster[0][1]) / cluster[0][1] <= 0.10:
                    cluster.append(potential_targets[j])
            
            if len(cluster) >= 2:
                avg_price = sum([item[1] for item in cluster]) / len(cluster)
                factors = [item[0] for item in cluster]
                if not any(abs(avg_price - cz['price']) / cz['price'] < 0.1 for cz in confluence_zones):
                    confluence_zones.append({"price": avg_price, "min_zone": cluster[0][1], "max_zone": cluster[-1][1], "factors": factors})

        if confluence_zones:
            primary_target = confluence_zones[0]
            result["cycle_target_zone"] = f"${format_price(primary_target['min_zone'])} - ${format_price(primary_target['max_zone'])}"
            result["cycle_confluence_factors"] = primary_target["factors"]
            factors_str = " และ ".join(primary_target["factors"])
            result["cycle_summary_label"] = f"🎯 <b>เป้าหมายรอบไซเคิล (Confluence):</b> <code>{result['cycle_target_zone']}</code>\n   <i>(จุดบรรจบของ: {factors_str})</i>"
        elif potential_targets:
            next_target = potential_targets[0]
            result["cycle_target_zone"] = f"${format_price(next_target[1])}"
            result["cycle_summary_label"] = f"🎯 <b>เป้าหมายรอบไซเคิล:</b> <code>${format_price(next_target[1])}</code>\n   <i>(อ้างอิงจาก: {next_target[0]})</i>"

    except Exception:
        pass
    return result

def check_death_cross_1d(df_1d: pd.DataFrame) -> dict:
    result = {
        "has_death_cross": False,
        "death_cross_label": "⚪ 1D Death Cross: ปลอดภัย (ยังไม่เกิดการตัดลง)"
    }

    if df_1d is None or len(df_1d) < 200:
        return result

    try:
        df = df_1d.copy()
        df["close"] = pd.to_numeric(df["close"])
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()

        df_last_30 = df.iloc[-30:]
        
        for i in range(1, len(df_last_30)):
            prev_50 = df_last_30["EMA_50"].iloc[i-1]
            prev_200 = df_last_30["EMA_200"].iloc[i-1]
            curr_50 = df_last_30["EMA_50"].iloc[i]
            curr_200 = df_last_30["EMA_200"].iloc[i]

            if prev_50 >= prev_200 and curr_50 < curr_200:
                result["has_death_cross"] = True
                result["death_cross_label"] = "☠️ <b>Warning! เกิด Death Cross (1D) ในช่วง 1 เดือนที่ผ่านมา</b> (เทรนด์หลักอาจเปลี่ยนเป็นขาลงยาว)"
                break
    except Exception as e:
        logger.warning(f"Error checking Death Cross: {e}")
        
    return result

def analyze_onchain_momentum(coin: str) -> dict:
    url = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
    params = {
        "fsym": coin,
        "limit": 30,
    }
    
    result = {
        "onchain_warning": False,
        "onchain_label": "🔹 On-Chain (1M): ปกติ (วาฬยังไม่มีพฤติกรรมผิดปกติ)",
        "large_tx_change_pct": 0.0
    }
    
    try:
        resp = api_session.get(url, params=params, timeout=15)
        data = resp.json()
        
        if data.get("Response") == "Success" and data.get("Data") and isinstance(data["Data"].get("Data"), list):
            df_onchain = pd.DataFrame(data["Data"]["Data"])
            if len(df_onchain) >= 15:
                tx_col = None
                for col in ['large_transaction_count', 'transaction_count', 'zero_balance_addresses_all_time']:
                    if col in df_onchain.columns:
                        tx_col = col
                        break
                
                if tx_col:
                    recent_avg = df_onchain[tx_col].iloc[-5:].mean()
                    prior_avg = df_onchain[tx_col].iloc[-30:-5].mean()
                    
                    if prior_avg > 0:
                        change_pct = ((recent_avg - prior_avg) / prior_avg) * 100
                        result["large_tx_change_pct"] = round(change_pct, 2)
                        
                        if change_pct > 40.0:
                            result["onchain_warning"] = True
                            result["onchain_label"] = f"🚨 <b>On-Chain Warning (1M): วาฬเคลื่อนไหวผิดปกติ!</b> มีปริมาณธุรกรรมยักษ์เพิ่มขึ้น {change_pct:.1f}% ในช่วงนี้"
                        else:
                            result["onchain_label"] = f"🔹 On-Chain (1M): ปกติ (ปริมาณธุรกรรมวาฬเปลี่ยนแปลง {change_pct:+.1f}%)"
    except Exception as e:
        logger.warning(f"ไม่สามารถตรวจสอบข้อมูล On-Chain ของ {coin} ได้: {e}")
        
    return result

# ==========================================
# Optimization 3: Refactored Futures context with Bulk Funding Rate
# ==========================================
def analyze_futures_context(coin: str, bulk_funding_rate: float | None) -> dict:
    """วิเคราะห์ข้อมูลฟิวเจอร์ส โดยลดการเรียก API Funding Rate รายตัวสลับไปใช้ข้อมูลชุด Bulk ล่าสุดแทน"""
    result = {
        "futures_warning": False,
        "futures_label": "📊 Futures (1M): ปกติ / ไม่มีข้อมูล"
    }
    symbol = f"{coin}USDT"
    
    try:
        # Open Interest ยังคงต้องยิงแยกเนื่องจากไม่มีจุดดึงภาพรวมย้อนหลังแบบ Bulk ใน API พื้นฐาน
        oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
        oi_params = {"symbol": symbol, "period": "1d", "limit": 30}
        oi_resp = api_session.get(oi_url, params=oi_params, timeout=10)

        if oi_resp.status_code == 200 and bulk_funding_rate is not None:
            oi_data = oi_resp.json()

            if len(oi_data) >= 2:
                first_oi = float(oi_data[0]['sumOpenInterestValue'])
                last_oi = float(oi_data[-1]['sumOpenInterestValue'])
                oi_change_pct = ((last_oi - first_oi) / first_oi) * 100 if first_oi > 0 else 0

                # แปลงค่า Funding Rate เป็นเปอร์เซ็นต์
                funding_pct = bulk_funding_rate * 100

                if oi_change_pct > 25.0 and funding_pct > 0.015:
                    result["futures_warning"] = True
                    result["futures_label"] = f"🚨 <b>Futures Warning: ระวัง Long Squeeze!</b> OI พุ่ง {oi_change_pct:.1f}% พร้อม Funding Rate ล่าสุดบวกเดือด ({funding_pct:.4f}%) ตลาดเสี่ยงดัมพ์กวาด Long"
                elif oi_change_pct > 25.0 and funding_pct < -0.015:
                    result["futures_warning"] = True
                    result["futures_label"] = f"🚨 <b>Futures Warning: ระวัง Short Squeeze!</b> OI พุ่ง {oi_change_pct:.1f}% แต่ Funding Rate ล่าสุดติดลบหนัก ({funding_pct:.4f}%) ตลาดเสี่ยงปั๊มสวน"
                else:
                    result["futures_label"] = f"📊 Futures (1M): ปกติ (OI เปลี่ยน {oi_change_pct:+.1f}%, FR ล่าสุด {funding_pct:.4f}%)"
                    
    except Exception:
        pass

    return result

# ==========================================
# Indicators & Technical Logic Calculations
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    df["EMA_50"] = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LONG, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

    # ATR (Average True Range)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    
    # ADX
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_safe = df["ATR"].replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)

    df["VOL_MA20"] = df["volumeto"].rolling(20).mean()
    return df

def calculate_correlation(df: pd.DataFrame, btc_df: pd.DataFrame) -> float:
    if btc_df is None or df is None: return 0.0
    try:
        aligned = df["close"].align(btc_df["close"], join="inner")
        if len(aligned[0]) < 2: return 0.0
        return round(aligned[0].corr(aligned[1]), 2)
    except:
        return 0.0

def estimate_price_for_target_rsi(df: pd.DataFrame, target_rsi: float = 70.0, period: int = RSI_PERIOD) -> float:
    if len(df) < period + 1: return None
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    curr_price = close.iloc[-1]
    
    if target_rsi >= 100: return curr_price * 1.5
    if target_rsi <= 0: return curr_price * 0.5
    
    target_rs = target_rsi / (100.0 - target_rsi)
    
    if target_rsi > 50:
        next_avg_loss = (avg_loss * (period - 1)) / period
        required_avg_gain = target_rs * next_avg_loss
        required_gain = max(0, (required_avg_gain * period) - (avg_gain * (period - 1)))
        return curr_price + required_gain
    else:
        next_avg_gain = (avg_gain * (period - 1)) / period
        required_avg_loss = next_avg_gain / target_rs
        required_loss = max(0, (required_avg_loss * period) - (avg_loss * (period - 1)))
        return curr_price - required_loss

def find_fair_value_gaps(df: pd.DataFrame) -> dict:
    fvg_result = {"has_fvg_support": False, "fvg_top": None, "fvg_bottom": None}
    if len(df) < 4: return fvg_result
    for i in range(len(df) - 1, 2, -1):
        high_minus2, low_current = df["high"].iloc[i - 2], df["low"].iloc[i]
        close_minus1, open_minus1 = df["close"].iloc[i - 1], df["open"].iloc[i - 1]
        
        if low_current > high_minus2 and close_minus1 > open_minus1:
            gap_pct = ((low_current - high_minus2) / high_minus2) * 100
            if gap_pct >= FVG_THRESHOLD_PCT:
                curr_price = df["close"].iloc[-1]
                if curr_price > high_minus2:
                    fvg_result.update({"has_fvg_support": True, "fvg_top": low_current, "fvg_bottom": high_minus2})
                    break
    return fvg_result

def analyze_trend_continuity(df: pd.DataFrame) -> dict:
    result = {
        "ema50_slope_pct": 0.0, "ema200_slope_pct": 0.0,
        "ema50_trending_up": False, "ema200_trending_up": False,
        "consecutive_up": 0, "consecutive_down": 0,
        "trend_strength": "sideways", "trend_label": "↔️ ไม่ชัดเจน",
    }
    n = TREND_SLOPE_BARS
    if len(df) < n + 2: return result

    ema50_now, ema50_prev = df["EMA_50"].iloc[-1], df["EMA_50"].iloc[-(n + 1)]
    ema200_now, ema200_prev = df["EMA_200"].iloc[-1], df["EMA_200"].iloc[-(n + 1)]

    slope50  = ((ema50_now  - ema50_prev)  / ema50_prev)  * 100 if ema50_prev  != 0 else 0
    slope200 = ((ema200_now - ema200_prev) / ema200_prev) * 100 if ema200_prev != 0 else 0

    result.update({
        "ema50_slope_pct": round(slope50, 4), "ema200_slope_pct": round(slope200, 4),
        "ema50_trending_up": slope50 > 0, "ema200_trending_up": slope200 > 0
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
        else: break

    result.update({"consecutive_up": up_streak, "consecutive_down": dn_streak})
    both_up, both_down = (slope50 > 0 and slope200 > 0), (slope50 <= 0 and slope200 <= 0)

    if both_up and up_streak >= TREND_MIN_CONSECUTIVE: strength, label = "strong_up", f"🚀 ขาขึ้นต่อเนื่องแข็งแกร่ง ({up_streak} แท่ง, EMA ชันขึ้นทั้งคู่)"
    elif slope50 > 0 and up_streak >= 1: strength, label = "moderate_up", f"📈 ขาขึ้นปานกลาง ({up_streak} แท่ง, EMA50 ชันขึ้น)"
    elif both_down and dn_streak >= TREND_MIN_CONSECUTIVE: strength, label = "strong_down", f"🔻 ขาลงต่อเนื่องแข็งแกร่ง ({dn_streak} แท่ง, EMA ชันลงทั้งคู่)"
    elif slope50 <= 0 and dn_streak >= 1: strength, label = "moderate_down", f"📉 ขาลงปานกลาง ({dn_streak} แท่ง, EMA50 ชันลง)"
    else: strength, label = "sideways", "↔️ Sideways / แนวโน้มไม่ชัด"

    result.update({"trend_strength": strength, "trend_label": label})
    return result

def analyze_rsi_bounce(df: pd.DataFrame) -> dict:
    window = LOOKBACK_BARS
    result = {"touched_oversold": False, "rsi_low": None, "rsi_rise": 0.0, "consecutive_rise": 0, "below_midline": False, "quality": "none", "quality_label": "⬜ ไม่มีสัญญาณดีดกลับ", "entry_timing": ""}
    if len(df) < window + RSI_BOUNCE_CONFIRM_BARS + 2: return result

    rsi_series, rsi_curr = df["RSI"].iloc[-(window + 1):-1], df["RSI"].iloc[-1]
    rsi_min = rsi_series.min()
    touched_oversold = rsi_min <= RSI_OVERSOLD
    result.update({"touched_oversold": touched_oversold, "rsi_low": round(rsi_min, 2)})

    if not touched_oversold: return result

    rsi_rise = rsi_curr - rsi_min
    result["rsi_rise"] = round(rsi_rise, 2)

    recent_rsi = df["RSI"].iloc[-(RSI_BOUNCE_CONFIRM_BARS + 3):]
    rsi_diffs  = recent_rsi.diff().iloc[1:]
    consec = sum(1 for val in reversed(rsi_diffs.values) if val > 0)
    
    result.update({"consecutive_rise": consec, "below_midline": rsi_curr < 50})
    has_recovered = (df["RSI"].iloc[-RSI_RECOVERY_LOOKBACK:] >= RSI_RECOVERY_THRESHOLD).any()

    score = sum([rsi_rise >= RSI_BOUNCE_MIN_RISE, consec >= RSI_BOUNCE_CONFIRM_BARS, rsi_curr < 50 or has_recovered])

    if score == 3: quality, label, timing = "strong", f"✅ ดีดกลับแข็งแกร่ง (ต่ำสุด {rsi_min:.1f} → ขึ้น {rsi_rise:.1f} จุด)", "⭐ จังหวะเข้าซื้อดีที่สุด"
    elif score == 2: quality, label, timing = "moderate", f"🟡 ดีดกลับปานกลาง (ต่ำสุด {rsi_min:.1f} → ขึ้น {rsi_rise:.1f} จุด)", "⚡ พิจารณาเข้าซื้อได้"
    elif score == 1: quality, label, timing = "weak", f"🟠 ดีดกลับอ่อน (ขึ้น {rsi_rise:.1f} จุด)", "⚠️ ยังไม่แนะนำ: สัญญาณดีดกลับยังไม่ชัด"
    else: quality, label, timing = "none", f"⬜ ยังไม่ดีดกลับ (ต่ำสุด {rsi_min:.1f})", "🚫 รอให้ RSI ดีดกลับก่อน"

    result.update({"quality": quality, "quality_label": label, "entry_timing": timing})
    return result

def find_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> dict:
    ob_result = {"bullish_ob_price": None, "bearish_ob_price": None, "has_bullish_ob": False, "has_bearish_ob": False}
    if len(df) < lookback + 5: return ob_result

    body_sizes = (df["close"] - df["open"]).abs()
    avg_body = body_sizes.rolling(20).mean().iloc[-1]
    curr_close, curr_open = df["close"].iloc[-1], df["open"].iloc[-1]
    curr_body = abs(curr_close - curr_open)

    past_df = df.iloc[-(lookback + 1):-(LOOKBACK_SKIP_BARS)]
    recent_high, recent_low = past_df["high"].max(), past_df["low"].min()

    if curr_close > recent_high and curr_body > (avg_body * OB_IMBALANCE_RATIO):
        for i in range(2, min(15, len(df))):
            idx = -i
            p_open, p_close, p_low = df["open"].iloc[idx], df["close"].iloc[idx], df["low"].iloc[idx]
            if p_close < p_open and not (df["low"].iloc[idx+1:] < p_low).any():
                ob_result.update({"has_bullish_ob": True, "bullish_ob_price": p_low})
                break
    elif curr_close < recent_low and curr_body > (avg_body * OB_IMBALANCE_RATIO):
        for i in range(2, min(15, len(df))):
            idx = -i
            p_open, p_close, p_high = df["open"].iloc[idx], df["close"].iloc[idx], df["high"].iloc[idx]
            if p_close > p_open and not (df["high"].iloc[idx+1:] > p_high).any():
                ob_result.update({"has_bearish_ob": True, "bearish_ob_price": p_high})
                break
    return ob_result

def check_bullish_divergence(df: pd.DataFrame) -> bool:
    if len(df) < LOOKBACK_BARS + 2: return False
    prev_window = df.iloc[-(LOOKBACK_BARS + 1) : -(LOOKBACK_SKIP_BARS)]
    if len(prev_window) == 0: return False
        
    min_low_idx = prev_window["low"].argmin()
    prev_low_price, prev_low_rsi = prev_window["low"].iloc[min_low_idx], prev_window["RSI"].iloc[min_low_idx]

    if prev_low_rsi > RSI_BULL_DIV_MAX: return False
    return (df["low"].iloc[-1] < prev_low_price) and (df["RSI"].iloc[-1] > prev_low_rsi)

def is_volume_confirmed(row: pd.Series) -> bool:
    if pd.isna(row.get("VOL_MA20")) or row["VOL_MA20"] == 0: return False
    return row["volumeto"] > row["VOL_MA20"]

def format_price(price: float) -> str:
    if price is None: return "N/A"
    if price < 0.0001: return f"{price:.8f}"
    elif price < 0.001: return f"{price:.6f}"
    elif price < 1: return f"{price:.4f}"
    else: return f"{price:.2f}"

# ==========================================
# Market Scanner with Multi-Timeframe Filter & Bulk optimization
# ==========================================
def scan_market():
    buy_signals, sell_signals = [], []
    bullish_coins, bearish_coins, total_valid_coins = 0, 0, 0
    coin_trends_summary = []

    # --- 1. Fetch Baseline Market Data (BTC) ---
    logger.info("ดึงข้อมูลดัชนีตลาด (BTC) เพื่อวิเคราะห์ Market Regime & Correlation...")
    btc_df = get_historical_data("BTC")
    market_regime = "Unknown ⚪"
    if btc_df is not None:
        btc_df = calculate_indicators(btc_df)
        btc_price = btc_df["close"].iloc[-1]
        btc_ema200 = btc_df["EMA_200"].iloc[-1]
        # แก้ไขบรรทัดนี้: เปลี่ยนเครื่องหมายเปรียบเทียบในข้อความเป็นคำอธิบายภาษาไทย ป้องกัน HTML Parsing Error ใน Telegram
        market_regime = "Bull Market 🟢 (BTC ยืนเหนือ 4H EMA200)" if btc_price > btc_ema200 else "Bear Market 🔴 (BTC หลุด 4H EMA200)"

    # --- 2. Optimization 3: Bulk Fetch Binance Funding Rate ---
    logger.info("ดึงข้อมูล Bulk Funding Rate จาก Binance (ยิงครั้งเดียวประหยัดพลังงาน)...")
    funding_map = {}
    try:
        pi_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        pi_resp = api_session.get(pi_url, timeout=10)
        if pi_resp.status_code == 200:
            pi_data = pi_resp.json()
            if isinstance(pi_data, list):
                for item in pi_data:
                    if "symbol" in item and "lastFundingRate" in item:
                        funding_map[item["symbol"]] = float(item["lastFundingRate"])
    except Exception as e:
        logger.warning(f"ไม่สามารถดึงข้อมูล Premium Index ภาพรวมได้: {e}")

    # --- 3. Scan All Coins ---
    for coin in COINS:
        logger.info(f"กำลังประมวลผลเหรียญ: {coin}...")
        
        # --- Optimization 2: ดึง 1D ครั้งเดียว จ่ายแจกให้ฟังก์ชันลูก ---
        df_daily = get_histoday_data(coin)
        time.sleep(API_RATE_LIMIT_DELAY)
        
        if df_daily is not None:
            weekly_ctx = analyze_weekly_context(df_daily)
            monthly_ctx = analyze_monthly_targets(df_daily)
            cycle_ctx = analyze_cycle_targets(df_daily)
            death_cross_ctx = check_death_cross_1d(df_daily)
        else:
            # Fallbacks ถ้าข้อมูลขาดหาย
            weekly_ctx = {"rsi_weekly": None, "weekly_bullish_div": False, "weekly_status_label": "↔️ ไม่พบข้อมูลระบุระดับสัปดาห์ชัดเจน", "fibo_618": None, "fibo_786": None, "fibo_886": None, "liquidity_pool": None, "psycho_support": None}
            monthly_ctx = {"m_resistance_target": None, "m_support_target": None, "monthly_summary_label": "⏳ ไม่สามารถคำนวณเป้าหมายระดับเดือนได้", "monthly_trend": "sideways"}
            cycle_ctx = {"cycle_target_zone": None, "cycle_confluence_factors": [], "cycle_summary_label": "⏳ ไม่สามารถวิเคราะห์เป้าหมายไซเคิลได้"}
            death_cross_ctx = {"has_death_cross": False, "death_cross_label": "⚪ 1D Death Cross: ปลอดภัย (ยังไม่เกิดการตัดลง)"}

        # ดึง 4H มาคำนวณ Indicator ประจำแท่งหลัก
        df = get_historical_data(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        if df is None or len(df) < EMA_LONG + 10:
            logger.warning(f"{coin}: ข้อมูลไม่พอสำหรับ Indicators 4H – ข้ามเหรียญนี้")
            continue

        onchain_ctx = analyze_onchain_momentum(coin) 
        time.sleep(API_RATE_LIMIT_DELAY) 
        
        # ค้นหาค่าและประกอบข้อมูลฟิวเจอร์สจากชุดข้อมูล Bulk
        symbol_futures = f"{coin}USDT"
        current_fr = funding_map.get(symbol_futures, None)
        futures_ctx = analyze_futures_context(coin, current_fr)
        time.sleep(API_RATE_LIMIT_DELAY) 
        
        df = calculate_indicators(df)
        row = df.iloc[-1]

        current_price = row["close"]
        rsi = row["RSI"]
        ema_50 = row["EMA_50"]
        ema_200 = row["EMA_200"]
        atr = row["ATR"]
        adx = row["ADX"]
        vol_confirmed = is_volume_confirmed(row)
        
        corr_btc = calculate_correlation(df, btc_df) if coin != "BTC" else 1.0
        squeeze_warning = adx < 20
        trailing_stop_val = current_price - (atr * ATR_MULTIPLIER)

        total_valid_coins += 1
        is_divergence = check_bullish_divergence(df)
        rsi_rounded = round(rsi, 2)

        trend_info = analyze_trend_continuity(df)
        bounce_info = analyze_rsi_bounce(df)
        ob_info = find_order_blocks(df)
        fvg_info = find_fair_value_gaps(df)

        dynamic_tp_ob = estimate_price_for_target_rsi(df, target_rsi=70.0)

        fibo_4h_max = df["high"].iloc[-60:].max()
        fibo_4h_min = df["low"].iloc[-60:].min()
        fibo_4h_618 = fibo_4h_max - (0.618 * (fibo_4h_max - fibo_4h_min))

        tier = COIN_TIER.get(coin, "mid")
        tp1_pct = TP_TIERS[tier]["tp1"]
        tp2_pct = TP_TIERS[tier]["tp2"]
        sl_buf = TP_TIERS[tier]["sl_buffer"]
        vol_tag = " 🔊" if vol_confirmed else ""

        in_fibo_zone = (weekly_ctx["fibo_618"] is not None) and (current_price <= weekly_ctx["fibo_618"] * 1.02)
        in_4h_fibo_zone = current_price <= (fibo_4h_618 * 1.01)
        in_ob_zone = ob_info["has_bullish_ob"] and (current_price <= ob_info["bullish_ob_price"] * 1.03)
        in_fvg_zone = fvg_info["has_fvg_support"] and (current_price <= fvg_info["fvg_top"]) and (current_price >= fvg_info["fvg_bottom"] * 0.99)

        in_deep_support = False
        if weekly_ctx.get("fibo_786") is not None:
            if (current_price <= weekly_ctx["fibo_786"] * 1.02) or \
               (weekly_ctx.get("liquidity_pool") and current_price <= weekly_ctx["liquidity_pool"] * 1.05) or \
               (weekly_ctx.get("psycho_support") and current_price <= weekly_ctx["psycho_support"] * 1.02):
                in_deep_support = True

        signal_type = ""

        if current_price > ema_200:
            coin_trend = "🟢 ขาขึ้น (Above EMA 200)"
            bullish_coins += 1
            coin_trends_summary.append(f"• {coin}: 🟢 ขาขึ้น (RSI 4H: {rsi_rounded}) | {trend_info['trend_label']}")

            if in_fibo_zone or in_4h_fibo_zone or in_ob_zone or in_fvg_zone:
                if current_price > (ema_50 * 0.98) and (rsi <= RSI_OVERSOLD or rsi <= RSI_PULLBACK_THRESHOLD):
                    if bounce_info["quality"] in ["strong", "moderate"]: signal_type = f"Institution Dip & Rebound 📉{vol_tag}"
                    elif rsi <= RSI_OVERSOLD: signal_type = f"Golden Fib / OB Zone Oversold 📉{vol_tag}"
                        
                if is_divergence and not signal_type: signal_type = f"Confluence Bullish Divergence 📈{vol_tag}"
                if ob_info["has_bullish_ob"] and not signal_type: signal_type = f"Smart Money OB Reversal 🚀{vol_tag}"

        else:
            coin_trend = "🔴 ขาลง (Below EMA 200)"
            bearish_coins += 1
            coin_trends_summary.append(f"• {coin}: 🔴 ขาลง (RSI 4H: {rsi_rounded}) | {trend_info['trend_label']}")

            if in_deep_support and is_divergence: signal_type = f"🚨 DEEP REVERSAL (Liquidity/Fibo 78.6-88.6) + Bullish Div 🐳{vol_tag}"
            elif in_deep_support and bounce_info["quality"] == "strong": signal_type = f"🛡️ Deep Support Strong Bounce (รอคอนเฟิร์มเทรนด์) 📉{vol_tag}"
            elif in_fibo_zone or in_ob_zone:
                if rsi <= RSI_OVERSOLD: signal_type = f"Deep Retracement Buy (เสี่ยงสูง) 📉{vol_tag}"
                elif is_divergence: signal_type = f"Macro Support Divergence (สวนเทรนด์) 📈{vol_tag}"

        if signal_type:
            if onchain_ctx.get("onchain_warning"):
                signal_type = f"⚠️ ปัจจัย On-Chain เสี่ยงสูง + {signal_type}"

            if weekly_ctx.get("weekly_bullish_div"): signal_type = f"⭐ {signal_type} + [1W Bullish Divergence แม่นยำสูง]"
            elif weekly_ctx.get("rsi_weekly") and weekly_ctx["rsi_weekly"] <= 35: signal_type = f"💎 {signal_type} + [1W คอนเฟิร์มโซนก้นหลุมสัปดาห์]"
            elif monthly_ctx.get("monthly_trend") == "bullish" and in_fvg_zone: signal_type = f"🔥 {signal_type} + [1M มหาเทรนด์หนุน + FVG เติมเต็ม]"

            entry_min = format_price(current_price * 0.98)
            entry_max = format_price(current_price * 1.01)
            target_tp1 = current_price * (1 + tp1_pct)
            target_tp2 = current_price * (1 + tp2_pct)
            
            sl_reference = ema_200
            if in_deep_support and weekly_ctx.get("fibo_886"): sl_reference = weekly_ctx["fibo_886"]
            elif ob_info["has_bullish_ob"]: sl_reference = ob_info["bullish_ob_price"]
            elif fvg_info["has_fvg_support"]: sl_reference = fvg_info["fvg_bottom"]
            
            sl_val = sl_reference * (1 - sl_buf) if current_price > sl_reference else current_price * (1 - sl_buf)
            
            # --- Position Sizing Calculation ---
            sl_distance_pct = (current_price - sl_val) / current_price
            if sl_distance_pct <= 0: sl_distance_pct = 0.01 
            risk_amount_usdt = PORTFOLIO_USDT * (RISK_PER_TRADE_PCT / 100)
            position_size_usdt = risk_amount_usdt / sl_distance_pct
            position_size_usdt = min(position_size_usdt, PORTFOLIO_USDT) 

            buy_signals.append({
                "coin": coin, "trend": coin_trend, "price": format_price(current_price), "rsi": rsi_rounded,
                "type": signal_type, "ema_50": format_price(ema_50), "ema_200": format_price(ema_200),
                "entry": f"${entry_min} - ${entry_max}",
                "tp1": f"${format_price(target_tp1)} (+{tp1_pct*100:.0f}%)",
                "tp2": f"${format_price(target_tp2)} (+{tp2_pct*100:.0f}%)",
                "dynamic_tp": f"${format_price(dynamic_tp_ob)}",
                "trailing_stop": f"${format_price(max(trailing_stop_val, sl_val))}",
                "sl": f"${format_price(sl_val)}",
                "vol_confirmed": vol_confirmed,
                "trend_info": trend_info, "bounce_info": bounce_info, "ob_info": ob_info, "fvg_info": fvg_info,
                "weekly_ctx": weekly_ctx, "monthly_ctx": monthly_ctx, "cycle_ctx": cycle_ctx, "onchain_ctx": onchain_ctx,
                "death_cross_ctx": death_cross_ctx, 
                "futures_ctx": futures_ctx,         
                "corr_btc": corr_btc, "squeeze_warning": squeeze_warning, "adx": round(adx, 2),
                "pos_size": f"${position_size_usdt:.2f}", "sl_risk_pct": f"{sl_distance_pct*100:.1f}%"
            })

        if rsi >= RSI_OVERBOUGHT or ob_info["has_bearish_ob"]:
            tp_min = format_price(current_price * 1.00)
            tp_max = format_price(current_price * (1 + tp1_pct * 0.4))
            exit_val = ema_50 if current_price > ema_50 else current_price * (1 - sl_buf)

            sell_signals.append({
                "coin": coin, "trend": coin_trend, "price": format_price(current_price), "rsi": rsi_rounded,
                "tp_zone": f"${tp_min} - ${tp_max}", "exit": f"${format_price(exit_val)}",
                "vol_confirmed": vol_confirmed, "trend_info": trend_info, "ob_info": ob_info,
                "weekly_ctx": weekly_ctx, "cycle_ctx": cycle_ctx
            })

    if total_valid_coins > 0:
        bullish_ratio = (bullish_coins / total_valid_coins) * 100
        summary_msg = f"📊 <b>[Market & Portfolio Summary]</b>\n"
        summary_msg += f"ทิศทางดัชนีหลัก (BTC): <b>{market_regime}</b>\n"
        summary_msg += f"ทุนระบบตั้งต้น: {PORTFOLIO_USDT} USDT (Risk {RISK_PER_TRADE_PCT}%/ไม้)\n"
        summary_msg += f"📈 ขาขึ้น: {bullish_coins} เหรียญ | 📉 ขาลง: {bearish_coins} เหรียญ\n"

        if bullish_ratio >= 65: summary_msg += "🔥 ภาพรวม: <b>🟢 ขาขึ้นชัดเจน (Strong Bullish)</b>\n<i>กลยุทธ์: ดักย่อซื้อเฉพาะจุดร่วมแนวรับระดับสถาบัน (Confluence Zone)</i>"
        elif bullish_ratio >= 40: summary_msg += "🔥 ภาพรวม: <b>🟡 ไซด์เวย์ / เลือกทาง (Sideways)</b>\n<i>กลยุทธ์: ตลาดก้ำกึ่ง ควรรอราคาลงสู้กรอบล่าง Fibonacci 61.8%</i>"
        else: summary_msg += "🔥 ภาพรวม: <b>🔴 ขาลง / พักฐานแรง (Bearish)</b>\n<i>กลยุทธ์: ตลาดเสี่ยงสูงมาก หลีกเลี่ยงสัญญาณทั่วไป ยกเว้นแนวรับทองคำ 1W</i>"

        summary_msg += "\n\n📋 <b>สรุปแนวโน้มรายเหรียญ:</b>\n" + "\n".join(coin_trends_summary)
    else:
        summary_msg = "⚠️ ไม่สามารถดึงข้อมูลเหรียญเพื่อวิเคราะห์ภาพรวมได้"

    return buy_signals, sell_signals, summary_msg

# ==========================================
# Message Builder with Confluence Alerts
# ==========================================
def build_messages(buy_list: list, sell_list: list, market_summary: str) -> list:
    message_blocks = [market_summary]

    if buy_list:
        buy_header = "🎯 <b>[Crypto Screener 4H - สัญญาณช้อนซื้อจุดแนวรับสำคัญ]</b>"
        current_block = buy_header

        for opt in buy_list:
            vol_note = "\n🔊 Volume: <b>ยืนยันสัญญาณ (สูงกว่า MA20)</b>" if opt["vol_confirmed"] else "\n🔇 Volume: ไม่ยืนยัน"
            
            corr_alert = f"\n⚠️ <b>Correlation:</b> {opt['corr_btc']} (วิ่งตาม BTC สูงมาก ควรระวังการกระจายความเสี่ยง)" if opt.get('corr_btc', 0) > 0.85 and opt['coin'] != 'BTC' else ""
            time_alert = f"\n⏳ <b>Time Decay Alert:</b> ADX ต่ำ ({opt.get('adx', 0)}) ตลาดไซด์เวย์ อาจต้องถือรอนาน" if opt.get('squeeze_warning') else ""

            ti, bi, ob, fvg = opt["trend_info"], opt["bounce_info"], opt["ob_info"], opt["fvg_info"]
            w_ctx, m_ctx, c_ctx = opt.get("weekly_ctx", {}), opt.get("monthly_ctx", {}), opt.get("cycle_ctx", {})
            onchain_ctx = opt.get("onchain_ctx", {})
            dc_ctx = opt.get("death_cross_ctx", {}) 
            ft_ctx = opt.get("futures_ctx", {})     

            confluence_report = "\n🛡️ <b>การทดสอบแนวรับสถาบัน:</b>"
            if w_ctx and w_ctx.get("fibo_618"):
                confluence_report += f"\n   🔹 Fibo 1W (61.8%): <code>${format_price(w_ctx['fibo_618'])}</code>"
                confluence_report += f"\n   🔸 Fibo 1W (78.6%): <code>${format_price(w_ctx['fibo_786'])}</code>"
                confluence_report += f"\n   🔻 <b>Deep Support (กรณีหลุด):</b>"
                confluence_report += f"\n      - Fibo 88.6%: <code>${format_price(w_ctx.get('fibo_886'))}</code>"
                confluence_report += f"\n      - Liquidity Pool: <code>${format_price(w_ctx.get('liquidity_pool'))}</code>"
                confluence_report += f"\n      - Psycho Support: <code>${format_price(w_ctx.get('psycho_support'))}</code>"

            if fvg.get("has_fvg_support"): confluence_report += f"\n   ⚡พบช่องว่าง FVG ยักษ์ (4H): <code>${format_price(fvg['fvg_bottom'])} - ${format_price(fvg['fvg_top'])}</code>"
            if ob.get("has_bullish_ob"): confluence_report += f"\n   🐳 Smart Money OB Support: <code>${format_price(ob['bullish_ob_price'])}</code>"

            trend_block = f"\n📐 <b>แนวโน้ม (4H):</b> {ti['trend_label']}"
            bounce_block = f"\n🔄 <b>RSI Bounce Check:</b> {bi['quality_label']}" + (f"\n   {bi['entry_timing']}" if bi["entry_timing"] else "")
            weekly_block = f"\n🗓️ <b>ภาพรวมระดับสัปดาห์ (1W):</b> {w_ctx['weekly_status_label']}" if w_ctx and w_ctx.get("rsi_weekly") else ""
            onchain_block = f"\n📊 <b>ข้อมูล On-Chain เชิงลึก:</b>\n   {onchain_ctx.get('onchain_label', '')}" if onchain_ctx else ""
            death_cross_block = f"\n   {dc_ctx.get('death_cross_label', '')}" if dc_ctx and dc_ctx.get("has_death_cross") else ""
            futures_block = f"\n   {ft_ctx.get('futures_label', '')}" if ft_ctx else ""
            monthly_block = f"\n🔮 <b>กรอบเป้าหมาย (1M):</b>\n   🔼 โซนเป้าหมายขึ้น: <code>${format_price(m_ctx['m_resistance_target'])}</code>\n   🔽 แนวรับถัดไป: <code>${format_price(m_ctx['m_support_target'])}</code>" if m_ctx and m_ctx.get("m_resistance_target") else ""
            cycle_block = f"\n{c_ctx['cycle_summary_label']}" if c_ctx and c_ctx.get("cycle_target_zone") else ""

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n📊 เทรนด์หลัก: {opt['trend']}"
                f"\n🚨 รูปแบบ: <b>{opt['type']}</b>"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']}"
                f"\n📉 RSI (4H): {opt['rsi']}"
                f"{vol_note}"
                f"{corr_alert}"
                f"{time_alert}"
                f"{confluence_report}"
                f"{trend_block}"
                f"{bounce_block}"
                f"{weekly_block}"
                f"{onchain_block}"
                f"{death_cross_block}" 
                f"{futures_block}"     
                f"{monthly_block}"
                f"{cycle_block}"
                f"\n\n🛡️ <b>[Risk Management]</b>"
                f"\n💼 <b>แนะนำขนาดไม้ซื้อ (Position Size): <code>{opt['pos_size']}</code></b>"
                f"\n📉 ระยะความเสี่ยง: {opt['sl_risk_pct']}"
                f"\n❌ ตัดขาดทุนป้องกันภัย (Hard SL): <code>{opt['sl']}</code>"
                f"\n\n🚀 <b>[Take Profit Strategy]</b>"
                f"\nเป้าหมาย Fix (TP1): <code>{opt['tp1']}</code>"
                f"\nเป้าหมาย Fix (TP2): <code>{opt['tp2']}</code>"
                f"\n🔥 <b>เป้าหมาย Dynamic (RSI=70): <code>{opt['dynamic_tp']}</code></b>"
                f"\n🔗 <b>จุดรันเทรนด์ (Trailing Stop): <code>{opt['trailing_stop']}</code></b>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = buy_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    if sell_list:
        sell_header = "⚠️ <b>[Crypto Screener 4H - เตือนโซนทำกำไร / แนวต้านยักษ์]</b>"
        current_block = sell_header

        for opt in sell_list:
            vol_note = "\n🔊 Volume: <b>ยืนยันแรงซื้อ (ระวังเกิดการพักตัว)</b>" if opt["vol_confirmed"] else "\n🔇 Volume: ปกติ"
            ti, ob = opt["trend_info"], opt["ob_info"]
            w_ctx, c_ctx = opt.get("weekly_ctx", {}), opt.get("cycle_ctx", {})
            
            trend_block = f"\n📐 <b>แนวโน้ม (4H):</b> {ti['trend_label']}"
            ob_block = f"\n🚨 <b>Smart Money Bearish OB:</b> กำแพงขายสถาบันที่ ${format_price(ob['bearish_ob_price'])}" if ob.get("has_bearish_ob") else ""
            weekly_block = f"\n🗓️ <b>ภาพรวมระดับสัปดาห์ (1W):</b> {w_ctx['weekly_status_label']}" if w_ctx and w_ctx.get("rsi_weekly") else ""
            cycle_block = f"\n{c_ctx['cycle_summary_label']}" if c_ctx and c_ctx.get("cycle_target_zone") else ""

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']}"
                f"\n📈 RSI (4H): {opt['rsi']}"
                f"{vol_note}"
                f"{trend_block}"
                f"{ob_block}"
                f"{weekly_block}"
                f"{cycle_block}"
                f"\n🔴 ช่วงราคาทยอยขายทำกำไร: <code>{opt['tp_zone']}</code>"
                f"\n❌ จุดล็อกกำไร (Safety Exit): <code>{opt['exit']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = sell_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    if not buy_list and not sell_list:
        message_blocks.append("\n=========================\n😴 <i>ตลาดนิ่ง: ไม่มีเหรียญย่อเข้าโซนแนวรับระดับสถาบันในรอบนี้</i>")

    return message_blocks

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    logger.info("เริ่มต้นใช้งาน Crypto Screener (SMC + Bulk Request & Session Optimization Ready)...")

    buy_list, sell_list, market_summary = scan_market()
    logger.info(f"สแกนระบบเสร็จสมบูรณ์ → พบสัญญาณซื้อคุณภาพ: {len(buy_list)} ตัว | พบสัญญาณขาย: {len(sell_list)} ตัว")

    final_messages = build_messages(buy_list, sell_list, market_summary)
    send_telegram_messages(final_messages)

    logger.info("บอททำงานและแจ้งเตือนผ่าน Telegram สำเร็จ!")
