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
exchange.load_trading_limits()
exchange.load_markets()
# Function to fetch recent trades and save as JSON
trade_tracking_df = pd.DataFrame(columns=[
    "timestamp", "latest_price", "moving_average", "volatility",
    "buy_sell_diff", "total_buy_sell", "total_volume", "std_dev",
    "buy_volume", "sell_volume", "buy_sell_ratio", "dominant_side"
])

def fetch_and_process_trades():
    params = {'type': 'swap', 'code': 'USD'}
    balances = exchange.fetch_balance(params=params)
    open_position = balances['info']['data']['positions']

    try:
        # Fetch recent trades and save to JSON
        trades = exchange.fetch_trades(SYMBOL)
        with open('recenttrades.json', 'w') as out:
            json.dump(trades, out, indent=6)

        # Read JSON to DataFrame
        df = pd.read_json('recenttrades.json')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime_str'] = df['datetime'].dt.strftime('%m/%d/%Y %H:%M:%S')

        # Clean and filter data
        df = df.drop(['cost', 'info', 'timestamp', 'id', 'fee', 'order', 'takeOrMaker', 'fees', 'type'], axis=1, errors='ignore')
        df['amount'] = df['amount'] * 100000000  # Scale up for visibility
        df = df[df['amount'] > 10000]  # Filter out small trades

        # Convert datetime to epoch time (in seconds)
        df['epoch'] = df['datetime'].apply(lambda x: x.timestamp())

        # Calculate total buy amount
        df.loc[df['side'] == 'buy', 'buy_amount'] = df['amount']
        df['buy_amount'] = df['buy_amount'].fillna(0)
        total_buy_amount = df['buy_amount'].sum()

        # Calculate total sell amount
        df.loc[df['side'] == 'sell', 'sell_amount'] = df['amount']
        df['sell_amount'] = df['sell_amount'].fillna(0)
        total_sell_amount = df['sell_amount'].sum()

        # Compute absolute difference
        diff = abs(total_sell_amount - total_buy_amount)

        # Determine which side dominates
        if total_buy_amount > total_sell_amount:
            print('There are more BUYS than SELLS')
            perc = round((diff / total_buy_amount) * 100, 2)
            moreof = 'BUYS'
            lessof = 'SELLS'
        else:
            print('There are more SELLS than BUYS')
            perc = round((diff / total_sell_amount) * 100, 2)
            moreof = 'SELLS'
            lessof = 'BUYS'

        # Format numbers for readability
        total_buy_amount_fmt = '{:,}'.format(total_buy_amount)
        total_sell_amount_fmt = '{:,}'.format(total_sell_amount)
        diff_fmt = '{:,}'.format(diff)

        # Print summary
        print(f'Total Buy Amount: {total_buy_amount_fmt}')
        print(f'Total Sell Amount: {total_sell_amount_fmt}')
        print(f'Difference: {diff_fmt}')
        print(f'{perc}% more {moreof} than {lessof}')

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


# Trading bot
def bot():
    logging.info("Running trading bot...")

    # Fetch and filter recent trades
    recent_trades_df = fetch_and_process_trades()

    # Fetch market data
    data = fetch_market_data(SYMBOL)
    if data is None or data.empty:
        logging.warning("No market data available.")
        return

    # Calculate indicators
    data['MA'] = moving_average(data)
    data['Volatility'] = calculate_volatility(data)
    data['BuySellDiff'] = buy_sell_diff(data)
    data['TotalBuySell'] = total_buy_sell(data)
    data['TotalVolume'] = total_volume(data)
    data['StdDev'] = calculate_std(data)

    # Show last few rows of processed market data
    print("\nLatest Market Data:\n", data.tail(3))

    # Extract latest values for logging or display
    latest_price = data['close'].iloc[-1]
    ma_value = data['MA'].iloc[-1]
    vol_value = data['Volatility'].iloc[-1]
    bs_diff = data['BuySellDiff'].iloc[-1]
    total_bs = data['TotalBuySell'].iloc[-1]
    total_vol = data['TotalVolume'].iloc[-1]
    std_dev = data['StdDev'].iloc[-1]

    # Extract buy/sell volumes from recent trades
    buy_volume = recent_trades_df[recent_trades_df['side'] == 'buy']['amount'].sum()
    sell_volume = recent_trades_df[recent_trades_df['side'] == 'sell']['amount'].sum()
    buy_sell_ratio = round(buy_volume / sell_volume, 2) if sell_volume > 0 else float('inf')
    dominant_side = 'BUY' if buy_volume > sell_volume else 'SELL'

    # Create a row of metrics
    row = {
        "timestamp": datetime.utcnow(),
        "latest_price": latest_price,
        "moving_average": ma_value,
        "volatility": vol_value,
        "buy_sell_diff": bs_diff,
        "total_buy_sell": total_bs,
        "total_volume": total_vol,
        "std_dev": std_dev,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_sell_ratio": buy_sell_ratio,
        "dominant_side": dominant_side
    }

    # Append to tracking DataFrame
    global trade_tracking_df
    trade_tracking_df = pd.concat([trade_tracking_df, pd.DataFrame([row])], ignore_index=True)

    # Log the extracted values
    logging.info(
        f"Latest Price: {latest_price}, MA: {ma_value}, Volatility: {vol_value}, Buy-Sell Diff: {bs_diff}, "
        f"Total Buy-Sell: {total_bs}, Total Volume: {total_vol}, Std Dev: {std_dev}")


if __name__ == "__main__":
    logging.info("Starting trading bot every 15 seconds...")
    while True:
        try:
            bot()
            time.sleep(15)  # Run every 15 seconds
        except Exception as e:
            logging.error(f"Unexpected error in loop: {e}")
            time.sleep(15)

