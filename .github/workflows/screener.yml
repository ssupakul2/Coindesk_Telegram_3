name: Crypto Screener (4H)

on:
  repository_dispatch:
    types: [trigger-screener]
#  workflow_dispatch:

# on:
  #schedule:
   # - cron: "5 */4 * * *"   # ทุก 4 ชั่วโมง (ตาม timeframe ของสัญญาณ)
  workflow_dispatch: {}

# จำเป็นสำหรับให้ workflow commit ไฟล์ positions.json / btc_dominance_history.json /
# trade_log.jsonl กลับเข้า repo ได้
permissions:
  contents: write

concurrency:
  group: crypto-screener
  cancel-in-progress: false   # ห้ามให้สอง run ทับไฟล์ state พร้อมกัน

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install requests pandas numpy

      - name: Run screener
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          CRYPTOCOMPARE_API_KEY: ${{ secrets.CRYPTOCOMPARE_API_KEY }}
          # [v6 #5] Macro News Filter — optional. ถ้าไม่ตั้งค่า secret นี้
          # MACRO_FILTER_ENABLED จะเป็น False อัตโนมัติ ไม่ error
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
          POSITIONS_FILE: positions.json
          # [v6 #4] BTC Dominance history (default ถ้าไม่ตั้งค่าก็ได้)
          BTC_DOMINANCE_HISTORY_FILE: btc_dominance_history.json
          # [v6] Realized PnL ledger (default ถ้าไม่ตั้งค่าก็ได้)
          TRADE_LOG_FILE: trade_log.jsonl
        run: python screener.py

      # commit ไฟล์ state ทั้งหมดกลับเข้า repo เพื่อให้ run ครั้งถัดไปอ่านต่อได้:
      #   - positions.json              (open positions, partial TP, runner state)
      #   - btc_dominance_history.json  (BTC.D snapshots สำหรับ #4 trend filter)
      #   - trade_log.jsonl             (append-only realized PnL ledger)
      - name: Commit updated state files
        run: |
          git config user.name "crypto-screener-bot"
          git config user.email "actions@users.noreply.github.com"

          git add -A positions.json btc_dominance_history.json trade_log.jsonl

          # ถ้าไม่มีอะไรเปลี่ยน ให้ skip commit แทน error
          if git diff --cached --quiet; then
            echo "No state changes to commit."
            exit 0
          fi

          git commit -m "chore: update screener state [skip ci]"
          git push
