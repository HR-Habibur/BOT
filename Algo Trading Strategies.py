import ccxt
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from statistics import stdev

# =============================================================================
# 1. Strategy Parameters
# =============================================================================
symbol = 'BTC/USDT'           # Example symbol; change as needed
RSI_Period = 6                # Period for raw RSI
SF = 5                        # Smoothing period for RSI Ma (EMA)
QQE = 3                       # QQE multiplier
Threshold = 3                 # (Unused in this snippet, but often part of QQE)
Wilders_Period = RSI_Period * 2 - 1  # Typical Wilder’s smoothing for ATR of RSI
length = 50                   # Length for final SMA + StdDev
mult = 0.35                   # Multiplier for the standard deviation

# =============================================================================
# 2. Data Retrieval
# =============================================================================
exchange = ccxt.phemex()  # Example; configure API keys if needed
timeframe = '15m'
limit = 500

# Fetch OHLCV bars
# (You can replace this with dfn.df_sma or any existing helper you have)
bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Keep only the timestamp and close for our strategy
src = df[['timestamp', 'close']].copy()

# =============================================================================
# 3. Compute RSI and Smoothed RSI (RSIndex)
# =============================================================================
# 3.1 Raw RSI on close
rsi_indicator = RSIIndicator(close=src['close'], window=RSI_Period)
src['Rsi'] = rsi_indicator.rsi()

# 3.2 EMA‐smoothed RSI (RsiMa)
ema_rsi = EMAIndicator(close=src['Rsi'], window=SF)
src['RsiMa'] = ema_rsi.ema_indicator()

# Shifted version for comparing one bar back
src['RsiMa_1back'] = src['RsiMa'].shift(1)

# For ease, rename RsiMa → RSIndex (i.e., we treat the smoothed RSI as our index)
src['RSIndex'] = src['RsiMa']

# =============================================================================
# 4. Compute ATR‐of‐RSI (Dar) and Delta
# =============================================================================
# 4.1 Absolute change of RsiMa from one bar back
src['AtrRsi'] = (src['RsiMa'].shift(1) - src['RsiMa']).abs()

# 4.2 Wilder’s EMA on AtrRsi
ema_atr_rsi = EMAIndicator(close=src['AtrRsi'], window=Wilders_Period)
src['MaAtrRsi'] = ema_atr_rsi.ema_indicator()

# 4.3 DeltaFastAtrRsi = MaAtrRsi * QQE
src['DeltaFastAtrRsi'] = src['MaAtrRsi'] * QQE

# =============================================================================
# 5. Define New Long/Short Bands and Their 2‐Bar Shifts
# =============================================================================
# 5.1 Current new long/short band
src['newlongband'] = src['RSIndex'] - src['DeltaFastAtrRsi']
src['newshortband'] = src['RSIndex'] + src['DeltaFastAtrRsi']

# 5.2 Shift both by 2 bars to reference “2 bars back”
src['newlongband_2back'] = src['newlongband'].shift(2)
src['newshortband_2back'] = src['newshortband'].shift(2)

# =============================================================================
# 6. Initialize Rolling Band Columns
# =============================================================================
src['longband'] = 0.0       # will hold our final long band level
src['shortband'] = 0.0      # will hold our final short band level
# Temporary columns for “computed max/min” at each step
src['longband_temp'] = np.nan  # ADDED/NOTIFIED: placeholder before finalizing each bar
src['shortband_temp'] = np.nan  # ADDED/NOTIFIED

# =============================================================================
# 7. Rolling Band Logic (Iterate Over Rows)
# =============================================================================
# We need to loop row by row because each band depends on its previous value.
# If you want pure vectorization, you’d need a more complex rolling apply. For clarity, I’m using a python loop.

# First, ensure no NaNs remain in key series (we’ll drop initial NaNs at the end)
src[['RsiMa_1back', 'newlongband_2back', 'newshortband_2back']] = src[['RsiMa_1back', 'newlongband_2back', 'newshortband_2back']].fillna(method='ffill')

for idx in range(1, len(src)):
    # Grab “yesterday’s” bands
    prev_long = src.at[idx - 1, 'longband']
    prev_short = src.at[idx - 1, 'shortband']
    newlb_2 = src.at[idx, 'newlongband_2back']
    newsb_2 = src.at[idx, 'newshortband_2back']
    rsi1 = src.at[idx - 1, 'RsiMa_1back']  # actually RsiMa two bars back if 1back was shifted
    rsi_curr = src.at[idx, 'RSIndex']

    # ----- Long Band Logic -----
    # If the RSI from one bar ago AND the current RSIndex are above the previous longband,
    # take max of previous longband vs. newlongband_2back
    if (src.at[idx - 1, 'RsiMa_1back'] > prev_long) and (rsi_curr > prev_long):
        src.at[idx, 'longband'] = max(prev_long, newlb_2)
    else:
        src.at[idx, 'longband'] = src.at[idx, 'newlongband']  # current new long band

    # ----- Short Band Logic -----
    # If the RSI from one bar ago AND the current RSIndex are below the previous shortband,
    # take min of previous shortband vs. newshortband_2back
    if (src.at[idx - 1, 'RsiMa_1back'] < prev_short) and (rsi_curr < prev_short):
        src.at[idx, 'shortband'] = min(prev_short, newsb_2)
    else:
        src.at[idx, 'shortband'] = src.at[idx, 'newshortband']  # current new short band

# =============================================================================
# 8. Prepare Lagged Band Columns for Crossover Detection
# =============================================================================
src['longband_1back'] = src['longband'].shift(1)
src['longband_2back'] = src['longband'].shift(2)
src['shortband_1back'] = src['shortband'].shift(1)
src['shortband_2back'] = src['shortband'].shift(2)

# =============================================================================
# 9. Crossover Utility Function
# =============================================================================
def find_cross(cur_val, prev_val, threshold):
    """
    Returns True if there is a crossover (or crossunder) of 'threshold' between prev_val→cur_val.
    I.e., prev_val < threshold < cur_val  OR  prev_val > threshold > cur_val.
    """
    if (cur_val > threshold and prev_val < threshold) or (cur_val < threshold and prev_val > threshold):
        return True
    return False

# =============================================================================
# 10. Detect Crossovers on Each Bar
# =============================================================================
src['prev_RSIndex'] = src['RSIndex'].shift(1)

# Cross of longband(1back) vs. RSIndex(current)
src['cross_1'] = src.apply(
    lambda row: find_cross(row['longband_1back'], row['longband_2back'], row['RSIndex']),
    axis=1
)

# Cross of RSIndex(current) vs. shortband(1back)
src['cross_s'] = src.apply(
    lambda row: find_cross(row['RSIndex'], row['prev_RSIndex'], row['shortband_1back']),
    axis=1
)

# =============================================================================
# 11. Trend Determination
# =============================================================================
# Initialize trend as 1 for the very first row (arbitrary default)
src['trend'] = 1

for idx in range(1, len(src)):
    if src.at[idx, 'cross_s']:
        src.at[idx, 'trend'] = 1
    elif src.at[idx, 'cross_1']:
        src.at[idx, 'trend'] = -1
    else:
        # Carry forward previous trend
        src.at[idx, 'trend'] = src.at[idx - 1, 'trend']

# =============================================================================
# 12. FastAtrRsiTL Line (Either Long or Short Band Based on Trend)
# =============================================================================
src['FastAtrRsiTL'] = np.where(src['trend'] == 1, src['longband'], src['shortband'])

# =============================================================================
# 13. Bollinger‐Style SMA + StdDev on FastAtrRsiTL
# =============================================================================
sma_fastrsi = SMAIndicator(close=src['FastAtrRsiTL'], window=length)
src['basis'] = sma_fastrsi.sma_indicator()

# Because pandas’ rolling std uses the sample std by default (ddof=1), this matches a typical Bollinger‐style.
src['dev'] = src['FastAtrRsiTL'].rolling(window=length).std(ddof=1) * mult

src['upper'] = src['basis'] + src['dev']
src['lower'] = src['basis'] - src['dev']

# =============================================================================
# 14. Color‐Coding the (RsiMa − 50) Histogram
# =============================================================================
# If (RsiMa − 50) > upper → Blue; if (RsiMa − 50) < lower → Red; else Gray.
src['hist_val'] = src['RsiMa'] - 50  # centered RSI
src['color'] = 'Gray'  # default
src.loc[src['hist_val'] > src['upper'], 'color'] = 'Blue'
src.loc[src['hist_val'] < src['lower'], 'color'] = 'Red'

# =============================================================================
# 15. QQE Long/Short Counters
# =============================================================================
# Initialize counters in case of NaNs
src['QQEzlong'] = 0
src['QQEzshort'] = 0

for idx in range(len(src)):
    if src.at[idx, 'RSIndex'] >= 50:
        # If RSIndex ≥ 50, increment QQEzlong; reset QQEzshort
        if idx == 0:
            src.at[idx, 'QQEzlong'] = 1
        else:
            src.at[idx, 'QQEzlong'] = src.at[idx - 1, 'QQEzlong'] + 1
        src.at[idx, 'QQEzshort'] = 0
    else:
        # If RSIndex < 50, increment QQEzshort; reset QQEzlong
        if idx == 0:
            src.at[idx, 'QQEzshort'] = 1
        else:
            src.at[idx, 'QQEzshort'] = src.at[idx - 1, 'QQEzshort'] + 1
        src.at[idx, 'QQEzlong'] = 0

# =============================================================================
# 16. Final Cleanup
# =============================================================================
# Drop any initial rows that contain NaNs due to shifts
src.dropna(inplace=True)

# Reset index if desired
src.reset_index(drop=True, inplace=True)

# =============================================================================
# 17. Inspect the Tail of DataFrame
# =============================================================================
print(src.tail(10))
