import time
import ccxt
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import schedule
from Phemex_trading_config import API_KEY, API_SECRET, SYMBOL

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Phemex with CCXT
exchange = ccxt.phemex({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})


# Function to fetch recent trades and save as JSON
def fetch_and_process_trades():
    try:
        # Fetch recent trades and save to JSON
        trades = exchange.fetch_trades(SYMBOL)
        with open('recenttrades.json', 'w') as out:
            json.dump(trades, out, indent=6)

        # Read JSON to DataFrame
        df = pd.read_json('recenttrades.json')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.strftime('%m/%d/%Y %H:%M:%S')

        # Clean and filter
        df = df.drop(['cost', 'info', 'timestamp', 'id', 'fee', 'order', 'takeOrMaker', 'fees', 'type'], axis=1,
                     errors='ignore')
        df['amount'] = df['amount'] * 100000000
        df = df[df.amount > 10000]

        print("\nFiltered Recent Trades:\n", df)
        return df
    except Exception as e:
        logging.error(f"Error processing trades: {e}")
        return pd.DataFrame()


# Fetch OHLCV Market Data
def fetch_market_data(symbol, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"Error fetching market data: {str(e)}")
        return None


# Indicators
def moving_average(df, window=14):
    return df['close'].rolling(window=window).mean()


def calculate_volatility(df, window=14):
    return df['close'].pct_change().rolling(window=window).std()


def buy_sell_diff(df):
    return df['high'] - df['low']


def total_buy_sell(df):
    return (df['close'] * df['volume']).sum()


def total_volume(df):
    return df['volume'].sum()


def calculate_std(df, window=14):
    return df['close'].rolling(window=window).std()


# Place order
def place_order(side, quantity, price=None):
    try:
        if price:
            order = exchange.create_limit_order(SYMBOL, side, quantity, price)
        else:
            order = exchange.create_market_order(SYMBOL, side, quantity)
        logging.info(f"Order placed: {side} {quantity} at {price if price else 'market'}")
        return order
    except Exception as e:
        logging.error(f"Order placement error: {e}")
        return None


# Trading strategy
def trading_strategy():
    logging.info("Running trading strategy...")

    # Fetch and filter recent trades
    recent_trades_df = fetch_and_process_trades()

    # Fetch market data
    data = fetch_market_data(SYMBOL)
    if data is None or data.empty:
        return

    # Calculate indicators
    data['MA'] = moving_average(data)
    data['Volatility'] = calculate_volatility(data)
    data['BuySellDiff'] = buy_sell_diff(data)
    data['TotalBuySell'] = total_buy_sell(data)
    data['TotalVolume'] = total_volume(data)
    data['StdDev'] = calculate_std(data)

    print("\nLatest Market Data:\n", data.tail(3))

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

    # Trading decision logic
    if latest_price > ma_value and vol_value < 0.02:
        logging.info("Buy condition met.")
        place_order("buy", 1)
    elif latest_price < ma_value and vol_value > 0.03:
        logging.info("Sell condition met.")
        place_order("sell", 1)


# Scheduler to run every 1 minute
schedule.every(1).minutes.do(trading_strategy)

if __name__ == "__main__":
    logging.info("Starting trading bot every 15 seconds...")
    while True:
        try:
            trading_strategy()
            time.sleep(15)  # Run every 15 seconds
        except Exception as e:
            logging.error(f"Unexpected error in loop: {e}")
            time.sleep(15)

