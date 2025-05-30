import time
import ccxt
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from Phemex_trading_config import API_KEY, API_SECRET, SYMBOL

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Phemex with CCXT
exchange = ccxt.phemex({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

# Function to fetch market data
def fetch_market_data(symbol, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=limit)  # Fetch OHLCV data (1 minute resolution)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return None

# Function to calculate moving averages
def moving_average(df, window=14):
    return df['close'].rolling(window=window).mean()

# Function to calculate volatility
def calculate_volatility(df, window=14):
    return df['close'].pct_change().rolling(window=window).std()

# Function to calculate buy-sell difference
def buy_sell_diff(df):
    return df['high'] - df['low']

# Function to calculate total buy and sell amount
def total_buy_sell(df):
    return (df['close'] * df['volume']).sum()

# Function to calculate total volume
def total_volume(df):
    return df['volume'].sum()

# Function to calculate standard deviation
def calculate_std(df, window=14):
    return df['close'].rolling(window=window).std()

# Function to place an order
def place_order(side, quantity, price=None):
    try:
        if price:
            order = exchange.create_limit_order(SYMBOL, side, quantity, price)
        else:
            order = exchange.create_market_order(SYMBOL, side, quantity)
        logging.info(f"Order placed: {side} {quantity} at {price if price else 'market'}")
        return order
    except Exception as e:
        logging.error(f"Error placing order: {str(e)}")
        return None

# Trading strategy
def trading_strategy():
    logging.info("Starting trading bot...")
    while True:
        data = fetch_market_data(SYMBOL)
        if data is None or data.empty:
            time.sleep(60)
            continue

        # Calculate indicators
        data['MA'] = moving_average(data)
        data['Volatility'] = calculate_volatility(data)
        data['BuySellDiff'] = buy_sell_diff(data)
        data['TotalBuySell'] = total_buy_sell(data)
        data['TotalVolume'] = total_volume(data)
        data['StdDev'] = calculate_std(data)

        # Print the DataFrame for debugging
        print(data)

        latest_price = data['close'].iloc[-1]
        ma_value = data['MA'].iloc[-1]
        vol_value = data['Volatility'].iloc[-1]
        bs_diff = data['BuySellDiff'].iloc[-1]
        total_bs = data['TotalBuySell']
        total_vol = data['TotalVolume']
        std_dev = data['StdDev'].iloc[-1]

        logging.info(
            f"Latest Price: {latest_price}, MA: {ma_value}, Volatility: {vol_value}, Buy-Sell Diff: {bs_diff}, "
            f"Total Buy-Sell: {total_bs}, Total Volume: {total_vol}, Std Dev: {std_dev}")

        # Trading decision
        if latest_price > ma_value and vol_value < 0.02:
            logging.info("Price above MA, low volatility: Buying...")
            place_order("buy", 1)
        elif latest_price < ma_value and vol_value > 0.03:
            logging.info("Price below MA, high volatility: Selling...")
            place_order("sell", 1)

        # Wait for 60 seconds before next check
        time.sleep(60)

# Start the trading strategy if the script is executed directly
if __name__ == "__main__":
    trading_strategy()
