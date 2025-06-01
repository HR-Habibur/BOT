from Alpaca_trading_config import API_SECRET,API_KEY,BASE_URL
import time
import json
import pandas as pd
from datetime import datetime, timedelta, timezone  # Added timezone and timedelta
import schedule
import logging
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# --- Configuration Loading ---
try:
    from Alpaca_trading_config import API_KEY, API_SECRET, SYMBOL, BASE_URL

    logging.info("Successfully loaded configuration from Alpaca_trading_config.py")
except ImportError:
    logging.error("Alpaca_trading_config.py not found or missing variables. Please create it.")
    # Provide default dummy values for the script to be syntactically complete,
    # but it won't run without a real config.
    API_KEY = API_KEY # Placeholder
    API_SECRET = API_SECRET  # Placeholder
    SYMBOL = "BTC/USD"  # Placeholder
    BASE_URL = "https://paper-api.alpaca.markets"  # Placeholder
    if API_KEY == API_KEY:
        logging.warning("Using default/placeholder API credentials. The bot will likely fail to connect to Alpaca.")
        # Consider exiting if config is essential: exit(1)

# Logging configuration
# Moved after config loading in case config itself has logging settings in a more complex setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Alpaca Client Initialization and Symbol Validation ---
alpaca = None
try:
    alpaca = REST(API_KEY, API_SECRET, base_url=BASE_URL)
    logging.info(f"Attempting to validate symbol: {SYMBOL} with Alpaca...")

    # Validate that the symbol is an active crypto asset
    active_crypto_assets = alpaca.list_assets(status='active', asset_class='crypto')
    symbol_is_valid_and_active = any(asset.symbol == SYMBOL for asset in active_crypto_assets)

    if not symbol_is_valid_and_active:
        available_symbols = [asset.symbol for asset in active_crypto_assets[:10]]  # Show a few examples
        logging.error(f"Symbol {SYMBOL} is not listed as an active crypto asset by Alpaca or does not exist.")
        logging.info(f"Some available active crypto symbols: {available_symbols}...")
        exit(1)

    logging.info(f"Successfully initialized Alpaca client. Using active crypto symbol: {SYMBOL}")

except APIError as e:
    logging.error(f"Alpaca API error during initialization or symbol validation: {e}")
    if hasattr(e, '_response') and e._response is not None:
        try:
            logging.error(f"Alpaca API response: {e._response.json()}")
        except ValueError:
            logging.error(f"Alpaca API response (text): {e._response.text}")
    exit(1)
except Exception as e:
    logging.error(f"Failed to initialize Alpaca client or validate symbol: {e}")
    exit(1)


# Function to fetch recent trades and save as JSON
def fetch_and_process_trades():
    """
    Fetches recent market trades for the configured SYMBOL using Alpaca API,
    processes them to calculate buy/sell volumes, and saves raw trades to JSON.
    """
    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=60)

        logging.info(f"Fetching trades for {SYMBOL} from {start_dt.isoformat()} to {end_dt.isoformat()}")
        raw_trades_data = alpaca.get_crypto_trades(SYMBOL, start=start_dt.isoformat(), end=end_dt.isoformat())

        # --- DIAGNOSTIC LOGGING (Uncomment if 'super object' error persists after fixing config) ---
        # logging.info(f"Type of raw_trades_data: {type(raw_trades_data)}")
        # if raw_trades_data is not None:
        #     logging.info(f"Length of raw_trades_data (if applicable): {len(raw_trades_data) if hasattr(raw_trades_data, '__len__') else 'N/A'}")
        # if isinstance(raw_trades_data, list) and len(raw_trades_data) > 0:
        #     logging.info(f"Type of first element in raw_trades_data: {type(raw_trades_data[0])}")
        #     logging.info(f"Content of first element: {raw_trades_data[0]}")
        #     try:
        #        logging.info(f"Vars of first element: {vars(raw_trades_data[0])}")
        #     except TypeError:
        #        logging.info(f"Could not get vars for first element (type: {type(raw_trades_data[0])})")
        # elif raw_trades_data:
        #     logging.info(f"Content of raw_trades_data (if not list or empty): {raw_trades_data}")
        # --- END DIAGNOSTIC LOGGING ---

        if not raw_trades_data:  # This checks if the list is empty or None
            logging.info(f"No trades found for {SYMBOL} in the last 60 minutes.")
            return pd.DataFrame()

        trade_list = []
        for i, t in enumerate(raw_trades_data):
            # --- DIAGNOSTIC LOGGING (Uncomment if 'super object' error persists after fixing config) ---
            # logging.info(f"Processing trade element index {i}, type: {type(t)}, content: {t}")
            # if i < 2: # Log details for the first few elements
            #     try:
            #         logging.info(f"Vars for trade element {i}: {vars(t)}")
            #     except TypeError:
            #         logging.info(f"Could not get vars for trade element {i} (type: {type(t)})")
            # --- END DIAGNOSTIC LOGGING ---
            try:
                trade_list.append({
                    'timestamp': t.timestamp,  # Error occurs here if t is not a valid Trade object
                    'price': t.price,
                    'size': t.size,
                    'side': str(t.side).lower(),
                    'exchange': t.exchange,
                    'id': t.id
                })
            except AttributeError as e_attr:
                logging.error(f"AttributeError on trade element {i} (type: {type(t)}): {e_attr}. Content: {t}")
                continue  # Skip this problematic trade element
            except Exception as e_loop:
                logging.error(
                    f"Unexpected error processing trade element {i} (type: {type(t)}): {e_loop}. Content: {t}")
                continue

        df = pd.DataFrame(trade_list)
        if df.empty and raw_trades_data:  # If trade_list is empty but raw_trades_data was not
            logging.warning(
                f"Trade list is empty after processing {len(raw_trades_data)} raw trade entries. Check for attribute errors above.")
            return pd.DataFrame()
        elif df.empty:
            logging.info(f"No valid trades processed into DataFrame.")
            return pd.DataFrame()

        logging.info(f"Fetched and processed {len(df)} trades for {SYMBOL}.")

        df_for_json = df.copy()
        # Ensure timestamp is JSON serializable if it's not already a string
        if 'timestamp' in df_for_json.columns and not pd.api.types.is_string_dtype(df_for_json['timestamp']):
            df_for_json['timestamp_iso'] = pd.to_datetime(df_for_json['timestamp']).apply(lambda x: x.isoformat())
            trades_to_dump = df_for_json[['timestamp_iso', 'price', 'size', 'side', 'exchange', 'id']].to_dict(
                orient='records')
        else:  # If timestamp is already suitable or 'timestamp_iso' was the original name
            # Adjust columns if 'timestamp' was already iso string or if 'size' was 'amount'
            cols_for_json = ['timestamp', 'price', 'amount' if 'amount' in df_for_json.columns else 'size', 'side',
                             'exchange', 'id']
            # Ensure all selected columns exist
            cols_for_json = [col for col in cols_for_json if col in df_for_json.columns]
            trades_to_dump = df_for_json[cols_for_json].to_dict(orient='records')

        with open('recenttrades.json', 'w') as out:
            json.dump(trades_to_dump, out, indent=4)

        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.rename(columns={'size': 'amount'}, inplace=True, errors='ignore')  # errors='ignore' if 'size' not present

        if 'amount' not in df.columns:
            logging.error(
                "'amount' column not found after processing trades. Aborting further processing in this function.")
            return pd.DataFrame()

        df['epoch'] = df['datetime'].apply(lambda x: x.timestamp())

        df.loc[df['side'] == 'buy', 'buy_amount'] = df['amount']
        df['buy_amount'] = df['buy_amount'].fillna(0)
        total_buy_amount = df['buy_amount'].sum()

        df.loc[df['side'] == 'sell', 'sell_amount'] = df['amount']
        df['sell_amount'] = df['sell_amount'].fillna(0)
        total_sell_amount = df['sell_amount'].sum()

        diff = abs(total_sell_amount - total_buy_amount)
        moreof, lessof, perc = '', '', 0.0

        if total_buy_amount > total_sell_amount:
            logging.info('There are more BUYS than SELLS in recent trades.')
            moreof, lessof = 'BUYS', 'SELLS'
            if total_buy_amount > 0: perc = round((diff / total_buy_amount) * 100, 2)
        elif total_sell_amount > total_buy_amount:
            logging.info('There are more SELLS than BUYS in recent trades.')
            moreof, lessof = 'SELLS', 'BUYS'
            if total_sell_amount > 0: perc = round((diff / total_sell_amount) * 100, 2)
        else:
            logging.info('BUY and SELL amounts are equal in recent trades.')

        total_buy_amount_fmt = '{:,.8f}'.format(total_buy_amount)
        total_sell_amount_fmt = '{:,.8f}'.format(total_sell_amount)
        diff_fmt = '{:,.8f}'.format(diff)

        logging.info(f'Total Buy Amount (Quantity): {total_buy_amount_fmt} {SYMBOL.split("/")[0]}')
        logging.info(f'Total Sell Amount (Quantity): {total_sell_amount_fmt} {SYMBOL.split("/")[0]}')
        logging.info(f'Difference (Quantity): {diff_fmt} {SYMBOL.split("/")[0]}')
        if moreof: logging.info(f'{perc}% more {moreof} than {lessof} (by quantity)')

        return df

    except APIError as e:
        logging.error(f"Alpaca API error processing trades: {e}")
        if hasattr(e, '_response') and e._response is not None:
            try:
                logging.error(f"Alpaca API response on error: {e._response.json()}")
            except ValueError:
                logging.error(f"Alpaca API response on error (text): {e._response.text}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"General error processing trades: {e}", exc_info=True)  # Added exc_info for more details
        return pd.DataFrame()


def process_timeframes(df_trades, last_known_epoch):
    """
    Processes trades for different timeframes (5m, 10m, 15m, 30m)
    based on new trades since last_known_epoch.
    Appends new trades to 'tape_reader_df.csv'.
    """
    now_epoch = datetime.now(timezone.utc).timestamp()

    if df_trades.empty or 'epoch' not in df_trades.columns:
        logging.info("df_trades is empty or missing 'epoch' column for tape reading analysis.")
        return {}, last_known_epoch  # Return original last_known_epoch if no new data to process

    new_trades_df = df_trades[df_trades['epoch'] > last_known_epoch].copy()

    if new_trades_df.empty:
        logging.info("No new trades to process for tape reading analysis (all trades older than last_known_epoch).")
        # Return the latest epoch from df_trades if it's newer than last_known_epoch,
        # otherwise, the original last_known_epoch. This handles cases where df_trades might have recent data
        # but none are *newer* than a very recent last_known_epoch.
        # However, if new_trades_df is empty, it implies no trades are newer.
        # So, we should return the max epoch from the input df_trades if it's more recent than last_known_epoch,
        # or simply the new_trades_df.epoch.max() if it's not empty.
        # The current logic returns now_epoch if new_trades_df is empty, which might be too aggressive.
        # Let's return the latest epoch from the *input* df_trades if it's greater than last_known_epoch,
        # otherwise last_known_epoch.
        # Corrected: if new_trades_df is empty, it means no trades were newer. So last_known_epoch is still valid.
        # The function should return the *new* last known epoch, which would be the max of new_trades_df.
        return {}, last_known_epoch

    try:
        tape_reader_df = pd.read_csv('tape_reader_df.csv')
        if not tape_reader_df.empty and 'epoch' in tape_reader_df.columns:
            tape_reader_df['epoch'] = tape_reader_df['epoch'].astype(float)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        tape_reader_df = pd.DataFrame()

    cols_to_save = ['datetime', 'price', 'amount', 'side', 'epoch', 'buy_amount', 'sell_amount', 'exchange', 'id']
    new_trades_df_to_save = new_trades_df[[col for col in cols_to_save if col in new_trades_df.columns]].copy()

    tape_reader_df = pd.concat([tape_reader_df, new_trades_df_to_save], ignore_index=True)
    if 'id' in tape_reader_df.columns and not tape_reader_df[
        'id'].isnull().all():  # Check if 'id' column has non-null values
        tape_reader_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
    elif 'epoch' in tape_reader_df.columns and 'price' in tape_reader_df.columns and 'amount' in tape_reader_df.columns and 'side' in tape_reader_df.columns:
        tape_reader_df.drop_duplicates(subset=['epoch', 'price', 'amount', 'side'], keep='last', inplace=True)

    tape_reader_df.to_csv('tape_reader_df.csv', index=False)

    current_max_epoch_processed = new_trades_df['epoch'].max()

    timeframes = {
        '5m': now_epoch - 5 * 60, '10m': now_epoch - 10 * 60,
        '15m': now_epoch - 15 * 60, '30m': now_epoch - 30 * 60,
    }
    results = {}
    for name, since_epoch in timeframes.items():
        df_tf = new_trades_df[new_trades_df['epoch'] >= since_epoch]

        if df_tf.empty:
            results[name] = {'Total Buy': 0, 'Total Sell': 0, 'Difference': 0, 'Percent Diff': 0, 'More Of': 'NONE',
                             'Less Of': 'NONE'}
            continue

        total_buy = df_tf['buy_amount'].sum()
        total_sell = df_tf['sell_amount'].sum()
        diff = abs(total_buy - total_sell)
        dominant, less, perc = 'NONE', 'NONE', 0.0
        denominator = 0

        if total_buy > total_sell:
            dominant, less, denominator = 'BUYS', 'SELLS', total_buy
        elif total_sell > total_buy:
            dominant, less, denominator = 'SELLS', 'BUYS', total_sell

        if denominator > 0: perc = round((diff / denominator) * 100, 2)

        results[name] = {
            'Total Buy': total_buy, 'Total Sell': total_sell, 'Difference': diff,
            'Percent Diff': perc, 'More Of': dominant, 'Less Of': less
        }
        logging.info(f"[{name} analysis of new trades] Buy Qty: {total_buy:,.8f}, Sell Qty: {total_sell:,.8f}, "
                     f"Diff: {diff:,.8f}, {perc}% more {dominant} than {less}")

    return results, current_max_epoch_processed


# Fetch OHLCV Market Data
def fetch_market_data(symbol_to_fetch, bar_limit=200):
    """Fetches OHLCV market data for the given symbol using Alpaca API."""
    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=bar_limit * 1.5)  # Fetch slightly more to ensure we get 'limit' bars back
        # Alpaca's limit is on returned bars, not strictly time window size.

        bars_data = alpaca.get_crypto_bars(
            symbol_to_fetch, TimeFrame.Minute,
            start=start_dt.isoformat(), end=end_dt.isoformat(),
            limit=bar_limit
        )

        if not bars_data:  # Check if the list of bars is empty
            logging.warning(f"No OHLCV data returned for {symbol_to_fetch} for the period (bars_data is empty/None).")
            return pd.DataFrame()

        bars_df = bars_data.df  # Convert to DataFrame

        if bars_df.empty:
            logging.warning(f"No OHLCV data returned for {symbol_to_fetch} (DataFrame is empty).")
            return pd.DataFrame()

        bars_df = bars_df.reset_index()
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Ensure all required columns are present
        missing_cols = [col for col in required_cols if col not in bars_df.columns]
        if missing_cols:
            logging.error(
                f"Missing expected columns in OHLCV data: {missing_cols}. Available columns: {bars_df.columns.tolist()}")
            return pd.DataFrame()

        ohlcv_df = bars_df[required_cols].copy()
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'])
        ohlcv_df[required_cols[1:]] = ohlcv_df[required_cols[1:]].astype(float)
        return ohlcv_df

    except APIError as e:
        logging.error(f"Alpaca API error fetching market data for {symbol_to_fetch}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"General error fetching market data for {symbol_to_fetch}: {str(e)}", exc_info=True)
        return pd.DataFrame()


# --- Indicators ---
def moving_average(df, window=14):
    if 'close' not in df.columns or df['close'].isnull().all(): return pd.Series(dtype='float64', index=df.index)
    return df['close'].rolling(window=window, min_periods=1).mean()


def calculate_volatility(df, window=14):
    if 'close' not in df.columns or df['close'].isnull().all(): return pd.Series(dtype='float64', index=df.index)
    return df['close'].pct_change().rolling(window=window, min_periods=1).std() * (252 ** 0.5)  # Annualized


def range_hl(df):
    if not all(col in df.columns for col in ['high', 'low']): return pd.Series(dtype='float64', index=df.index)
    return df['high'] - df['low']


def total_value_traded(df):
    if not all(col in df.columns for col in ['close', 'volume']): return pd.Series(dtype='float64', index=df.index)
    return df['close'] * df['volume']


def calculate_std_dev(df, window=14):
    if 'close' not in df.columns or df['close'].isnull().all(): return pd.Series(dtype='float64', index=df.index)
    return df['close'].rolling(window=window, min_periods=1).std()


# --- Trading bot ---
last_processed_trade_epoch = 0


def bot():
    global last_processed_trade_epoch
    logging.info(f"--- Running trading bot cycle for {SYMBOL} ---")

    recent_trades_df = fetch_and_process_trades()
    if recent_trades_df.empty:
        logging.warning("No recent market trades to analyze in this cycle.")
    else:
        logging.info(f"Successfully fetched {len(recent_trades_df)} recent market trades.")

    if not recent_trades_df.empty and 'epoch' in recent_trades_df.columns:
        timeframe_analysis_results, new_max_epoch = process_timeframes(recent_trades_df, last_processed_trade_epoch)
        if new_max_epoch > last_processed_trade_epoch:
            last_processed_trade_epoch = new_max_epoch
        logging.info(f"Timeframe analysis results: {timeframe_analysis_results}")
        logging.info(f"Updated last_processed_trade_epoch to: {last_processed_trade_epoch}")
    elif recent_trades_df.empty:
        logging.info("Skipping timeframe processing as no recent trades were fetched.")
    else:  # recent_trades_df is not empty but 'epoch' column is missing
        logging.warning("Skipping timeframe processing as 'epoch' column is missing in recent_trades_df.")

    market_data_df = fetch_market_data(SYMBOL, bar_limit=200)
    if market_data_df.empty:
        logging.warning("No market OHLCV data available for indicator calculation.")
        logging.info("--- Trading bot cycle finished (due to no market data) ---")
        return

    market_data_df['MA14'] = moving_average(market_data_df, window=14)
    market_data_df['Volatility14'] = calculate_volatility(market_data_df, window=14)
    market_data_df['RangeHL'] = range_hl(market_data_df)
    market_data_df['ValueTradedBar'] = total_value_traded(market_data_df)
    market_data_df['StdDev14'] = calculate_std_dev(market_data_df, window=14)

    if not market_data_df.empty:
        latest_bar = market_data_df.iloc[-1]
        logging.info("\n--- Latest Market Data & Indicators ---")
        log_output = f"Timestamp: {latest_bar['timestamp']}\n"
        log_output += f"Close Price: {latest_bar['close']:.2f}\n"
        log_output += f"MA14: {latest_bar.get('MA14', float('nan')):.2f} | Volatility14: {latest_bar.get('Volatility14', float('nan')):.6f} | StdDev14: {latest_bar.get('StdDev14', float('nan')):.2f}\n"
        log_output += f"Range (H-L): {latest_bar.get('RangeHL', float('nan')):.2f} | Value Traded (bar): {latest_bar.get('ValueTradedBar', float('nan')):.2f} | Volume (bar): {latest_bar.get('volume', float('nan')):.2f}"
        logging.info(log_output)
    else:
        logging.info("No market data to display indicators for (market_data_df is empty after calculations).")

    logging.info("--- Trading bot cycle finished ---")


# --- Scheduler ---
if __name__ == "__main__":
    if alpaca is None:
        logging.error("Alpaca client not initialized. Exiting.")
        exit(1)

    try:
        # Check if file exists and is not empty
        try:
            if pd.read_csv('tape_reader_df.csv', nrows=1).empty:  # Check if file is empty
                logging.info("tape_reader_df.csv is empty. Starting last_processed_trade_epoch from 0.")
                last_processed_trade_epoch = 0
            else:
                temp_tape_df = pd.read_csv('tape_reader_df.csv')
                if 'epoch' in temp_tape_df.columns and not temp_tape_df['epoch'].isnull().all():
                    last_processed_trade_epoch = temp_tape_df['epoch'].astype(float).max()
                    logging.info(
                        f"Initialized last_processed_trade_epoch from tape_reader_df.csv: {last_processed_trade_epoch}")
                else:
                    logging.info(
                        "tape_reader_df.csv has no 'epoch' column or all epoch values are null. Starting last_processed_trade_epoch from 0.")
                    last_processed_trade_epoch = 0
        except pd.errors.EmptyDataError:  # Handles files with only headers or completely empty
            logging.info("tape_reader_df.csv is empty (EmptyDataError). Starting last_processed_trade_epoch from 0.")
            last_processed_trade_epoch = 0
        except FileNotFoundError:
            logging.info("tape_reader_df.csv not found. Starting last_processed_trade_epoch from 0.")
            last_processed_trade_epoch = 0
    except Exception as e:  # Catch any other unexpected error during init
        logging.error(f"Error reading tape_reader_df.csv for initial epoch: {e}. Starting from 0.", exc_info=True)
        last_processed_trade_epoch = 0

    logging.info(f"Starting trading bot for {SYMBOL}. Scheduled to run every 15 seconds.")

    try:
        bot()
    except Exception as e:
        logging.error(f"Error during initial bot run: {e}", exc_info=True)

    schedule.every(15).seconds.do(bot)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Trading bot stopped by user.")
            break
        except Exception as e:
            logging.error(f"Error in scheduler loop: {e}", exc_info=True)
            time.sleep(5)
