"""
backtest_v1.py — Vectorized backtest framework (เบื้องต้น, ไม่ walk-forward)

วัตถุประสงค์:
    ทดสอบ entry/exit logic ของ screener_v4.py บนข้อมูล 4H ย้อนหลัง เพื่อประเมิน
    win rate, average R-multiple, max drawdown ก่อนนำ parameter ไปใช้จริง

ขอบเขต (ตามที่ตกลง):
    - Vectorized เบื้องต้น: ใช้ pandas เพื่อจำลอง entry/exit ตาม rule ที่กำหนด
    - ไม่ใช่ walk-forward / ไม่ re-optimize parameter ระหว่างทาง
    - ใช้ข้อมูล 4H ย้อนหลังจาก Binance (หรือแหล่งสำรองเดียวกับ screener_v4)
    - Logic การเข้า/ออก เป็น "simplified" version ของ screener_v4 — ไม่ใช่การ
      replay ของทุกฟังก์ชันแบบ exact (on-chain data, weekly fibo ฯลฯ ถูกตัดออก
      เพื่อความเร็วและความเรียบง่ายของการ backtest เบื้องต้น)

วิธีใช้:
    python3 backtest_v1.py --coin BTC --days 365
    python3 backtest_v1.py --coin ETH --days 180 --min-score 50

ผลลัพธ์:
    พิมพ์สรุป: จำนวน trades, win rate, avg R-multiple, total R, max drawdown (R)
    และ list ของแต่ละ trade (entry/exit time, price, P&L in R-multiples)
"""

import argparse
import logging
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# ==========================================
# Re-used constants from screener_v4 (kept in sync manually)
# ==========================================
RSI_PERIOD     = 14
EMA_SHORT      = 50
EMA_LONG       = 200
RSI_OVERSOLD   = 32
RSI_OVERBOUGHT = 70
ATR_PERIOD     = 14

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

BINANCE_ENDPOINTS = [
    "https://api.binance.us",
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api3.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]

session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


# ==========================================
# Data fetching (Binance only, simplified — for backtest convenience)
# ==========================================
def fetch_klines(symbol: str, interval: str = "4h", days: int = 365) -> pd.DataFrame:
    """ดึงข้อมูล klines ย้อนหลังจาก Binance โดยวน loop ตาม limit ของ API (1000 แท่ง/ครั้ง)"""
    bars_per_day = 6 if interval == "4h" else 1
    total_bars_needed = days * bars_per_day
    all_rows = []
    end_time = None

    while len(all_rows) < total_bars_needed:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time:
            params["endTime"] = end_time

        data = None
        for base_url in BINANCE_ENDPOINTS:
            try:
                resp = session.get(f"{base_url}/api/v3/klines", params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    break
            except Exception:
                continue

        if not data:
            logger.error(f"ไม่สามารถดึงข้อมูล {symbol} {interval} ได้")
            break

        all_rows = data + all_rows
        if len(data) < 1000:
            break
        end_time = data[0][0] - 1  # go further back

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "volumeto", "trades", "taker_base", "taker_quote", "ignore",
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index("time", inplace=True)
    for col in ["open", "high", "low", "close", "volumeto"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["open", "high", "low", "close", "volumeto"]].dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.tail(total_bars_needed)


# ==========================================
# Indicators (subset needed for backtest)
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

    df["VOL_MA20"] = df["volumeto"].rolling(20).mean()
    return df


def analyze_rsi_bounce_quality(df: pd.DataFrame, i: int) -> str:
    """Simplified RSI bounce quality at row index i (mirrors screener_v4 logic)."""
    if i < 20:
        return "none"
    rsi_series = df["RSI"].iloc[i-15:i]
    rsi_curr = df["RSI"].iloc[i]
    rsi_min = rsi_series.min()
    if rsi_min > RSI_OVERSOLD:
        return "none"
    rsi_rise = rsi_curr - rsi_min
    diffs = df["RSI"].iloc[i-4:i+1].diff().iloc[1:].values[::-1]
    consec = sum(1 for v in diffs if v > 0)
    score = (1 if rsi_rise >= 3.0 else 0) + (1 if consec >= 2 else 0) + (1 if rsi_curr < 50 or (df["RSI"].iloc[i-4:i+1] >= 45).any() else 0)
    if score == 3: return "strong"
    if score == 2: return "moderate"
    return "none"


def check_bullish_divergence(df: pd.DataFrame, i: int) -> bool:
    if i < 17:
        return False
    prev = df.iloc[i-16:i-3]
    if len(prev) == 0:
        return False
    min_idx_pos = prev["low"].values.argmin()
    return (
        prev["RSI"].iloc[min_idx_pos] <= 45
        and df["low"].iloc[i] < prev["low"].iloc[min_idx_pos]
        and df["RSI"].iloc[i] > prev["RSI"].iloc[min_idx_pos]
    )


# ==========================================
# Simplified entry rule (mirrors screener_v4 "Confluence Bullish Divergence"
# and basic dip-buy conditions; on-chain / weekly fibo / OB / FVG omitted
# for backtest simplicity per agreed scope)
# ==========================================
def entry_signal(df: pd.DataFrame, i: int, min_score: int = 50) -> bool:
    if i < EMA_LONG + 10:
        return False
    row = df.iloc[i]
    price, rsi, ema50, ema200 = row["close"], row["RSI"], row["EMA_50"], row["EMA_200"]
    if price <= ema200:
        return False

    bounce = analyze_rsi_bounce_quality(df, i)
    is_div = check_bullish_divergence(df, i)

    signal = False
    if price > (ema50 * 0.98) and rsi <= 55 and bounce in ("strong", "moderate"):
        signal = True
    if is_div:
        signal = True

    if not signal:
        return False

    # Simplified scoring (subset of calculate_signal_score)
    score = (25 if rsi <= RSI_OVERSOLD else 15 if rsi <= 40 else 8 if rsi <= 50 else 0)
    score += {"strong": 20, "moderate": 12, "weak": 5, "none": 0}.get(bounce, 0)
    if is_div:
        score += 10
    vol = row["volumeto"]
    vol_ma = row["VOL_MA20"]
    if not pd.isna(vol_ma) and vol_ma > 0 and vol > vol_ma:
        score += 10

    return score >= min_score


# ==========================================
# Backtest engine (vectorized iteration over bars; one position at a time)
# ==========================================
def run_backtest(df: pd.DataFrame, tier: str, min_score: int = 50,
                 breakeven_trigger_pct: float = 0.5, trail_atr_mult: float = 1.5,
                 time_stop_bars: int = 42) -> pd.DataFrame:
    tp1_pct, tp2_pct, sl_buf = TP_TIERS[tier]["tp1"], TP_TIERS[tier]["tp2"], TP_TIERS[tier]["sl_buffer"]
    trades = []
    in_position = False
    entry_price = sl = tp1 = tp2 = 0.0
    entry_idx = 0
    tp1_hit = False

    for i in range(len(df)):
        row = df.iloc[i]
        price, atr = row["close"], row["ATR"]

        if not in_position:
            if entry_signal(df, i, min_score=min_score):
                entry_price = price
                sl = entry_price * (1 - sl_buf)
                tp1 = entry_price * (1 + tp1_pct)
                tp2 = entry_price * (1 + tp2_pct)
                entry_idx = i
                tp1_hit = False
                in_position = True
            continue

        # In position: check exit conditions using this bar's high/low
        bar_high, bar_low = row["high"], row["low"]
        exit_price, exit_reason = None, None

        # SL hit (check low first — conservative assumption)
        if bar_low <= sl:
            exit_price, exit_reason = sl, "SL"
        elif bar_high >= tp2:
            exit_price, exit_reason = tp2, "TP2"
        else:
            # Trailing stop logic
            halfway = entry_price + (tp1 - entry_price) * breakeven_trigger_pct
            if not tp1_hit and bar_high >= tp1:
                tp1_hit = True
            if tp1_hit and not pd.isna(atr) and atr > 0:
                trail_sl = price - (atr * trail_atr_mult)
                if trail_sl > sl:
                    sl = trail_sl
            elif not tp1_hit and price >= halfway and sl < entry_price:
                sl = entry_price

            # Time stop
            if not tp1_hit and (i - entry_idx) >= time_stop_bars:
                exit_price, exit_reason = price, "TIME_STOP"

        if exit_price is not None:
            r_multiple = (exit_price - entry_price) / (entry_price - (entry_price * (1 - sl_buf)))
            trades.append({
                "entry_time": df.index[entry_idx], "exit_time": df.index[i],
                "entry_price": entry_price, "exit_price": exit_price,
                "reason": exit_reason, "r_multiple": r_multiple,
                "bars_held": i - entry_idx,
            })
            in_position = False

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, coin: str) -> None:
    if trades.empty:
        print(f"\n=== {coin}: ไม่มี trades ในช่วงที่ทดสอบ ===")
        return

    wins = trades[trades["r_multiple"] > 0]
    losses = trades[trades["r_multiple"] <= 0]
    total_r = trades["r_multiple"].sum()
    win_rate = len(wins) / len(trades) * 100

    # Max drawdown in R, based on cumulative R curve
    cum_r = trades["r_multiple"].cumsum()
    running_max = cum_r.cummax()
    drawdown = cum_r - running_max
    max_dd_r = drawdown.min()

    print(f"\n=== ผลทดสอบ: {coin} ===")
    print(f"จำนวน Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}% ({len(wins)} win / {len(losses)} loss)")
    print(f"Total R: {total_r:.2f}")
    print(f"Avg R per Trade: {trades['r_multiple'].mean():.2f}")
    print(f"Max Drawdown: {max_dd_r:.2f} R")
    print(f"Avg Bars Held: {trades['bars_held'].mean():.1f} (4H bars)")
    print("\nExit Reason Breakdown:")
    print(trades["reason"].value_counts().to_string())
    print("\nรายการ Trades (5 รายการล่าสุด):")
    print(trades.tail(5)[["entry_time", "exit_time", "entry_price", "exit_price", "reason", "r_multiple"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Vectorized backtest สำหรับ screener_v4 entry/exit logic (เบื้องต้น, ไม่ walk-forward)")
    parser.add_argument("--coin", type=str, default="BTC", help="ชื่อเหรียญ (เช่น BTC, ETH, SOL)")
    parser.add_argument("--days", type=int, default=365, help="จำนวนวันย้อนหลัง (4H bars)")
    parser.add_argument("--min-score", type=int, default=50, help="Minimum signal score สำหรับเข้า position")
    args = parser.parse_args()

    coin = args.coin.upper()
    tier = COIN_TIER.get(coin, "mid")
    symbol = f"{coin}USDT"

    logger.info(f"ดึงข้อมูล {symbol} 4H ย้อนหลัง {args.days} วัน...")
    df = fetch_klines(symbol, "4h", args.days)
    if df.empty or len(df) < EMA_LONG + 20:
        logger.error("ข้อมูลไม่พอสำหรับ backtest (ต้องการอย่างน้อย EMA_LONG + 20 แท่ง)")
        return

    df = calculate_indicators(df)
    trades = run_backtest(df, tier, min_score=args.min_score)
    summarize(trades, coin)

    print("\n⚠️  หมายเหตุ: ผลทดสอบนี้เป็น 'simplified vectorized backtest' — ไม่รวม")
    print("   on-chain data, weekly fibo zones, order blocks, FVG, MTF confluence,")
    print("   และไม่ใช่ walk-forward (parameter ไม่ถูก re-optimize ระหว่างทาง)")
    print("   ใช้เป็นแนวทางเบื้องต้นเท่านั้น ไม่ใช่การันตีผลจริง")


if __name__ == "__main__":
    main()
