# debug_fetch.py — วางไว้ใน repo เดียวกัน รันแยก
import os, requests, pandas as pd

CRYPTOCOMPARE_API_KEY = str(os.getenv("CRYPTOCOMPARE_API_KEY") or "").strip()
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://api3.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com"
]

# ทดสอบ Binance
print("=== TEST BINANCE ===")
for base in BINANCE_ENDPOINTS:
    try:
        r = requests.get(f"{base}/api/v3/klines",
                         params={"symbol":"BTCUSDT","interval":"4h","limit":10},
                         timeout=10)
        print(f"{base} → HTTP {r.status_code} | rows={len(r.json()) if r.status_code==200 else 'N/A'}")
    except Exception as e:
        print(f"{base} → ERROR: {e}")

# ทดสอบ CryptoCompare
print("\n=== TEST CRYPTOCOMPARE ===")
params = {"fsym":"BTC","tsym":"USD","limit":10}
if CRYPTOCOMPARE_API_KEY:
    params["api_key"] = CRYPTOCOMPARE_API_KEY
r = requests.get("https://min-api.cryptocompare.com/data/v2/histohour",
                 params=params, timeout=10)
j = r.json()
print(f"HTTP {r.status_code} | Response={j.get('Response')} | Message={j.get('Message','')}")
if j.get("Response") == "Success":
    print(f"rows={len(j['Data']['Data'])}")
    print(f"columns={list(pd.DataFrame(j['Data']['Data']).columns)}")

print("\n=== API KEY LOADED ===")
print(f"CC Key: {'YES (' + CRYPTOCOMPARE_API_KEY[:6] + '...)' if CRYPTOCOMPARE_API_KEY else 'NO KEY'}")
