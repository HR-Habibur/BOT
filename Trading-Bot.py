import alpaca_trade_api as tradeapi
import pandas as pd
import ta
from trading_config import ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, SYMBOL
from alpaca_trade_api.rest import TimeFrame


class TradingBot:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')
        self.symbol = SYMBOL

    def fetch_historical_data(self):
        try:
            bars = self.api.get_bars(self.symbol, TimeFrame.Day, limit=100).df
            bars = bars.reset_index()
            bars = bars.rename(columns={'timestamp': 'time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                                        'volume': 'volume'})
            return bars
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, data):
        data['SMA50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['SMA200'] = ta.trend.sma_indicator(data['close'], window=200)
        data['RSI'] = ta.momentum.rsi(data['close'], window=14)
        return data

    def make_decision(self, data):
        if data.empty:
            print("Data is empty, holding position.")
            return 'hold'
        last_row = data.iloc[-1]
        print(f"Latest Indicators - SMA50: {last_row['SMA50']}, SMA200: {last_row['SMA200']}, RSI: {last_row['RSI']}")
        if last_row['SMA50'] > last_row['SMA200'] and last_row['RSI'] < 30:
            return 'buy'
        elif last_row['SMA50'] < last_row['SMA200'] and last_row['RSI'] > 70:
            return 'sell'
        else:
            return 'hold'

    def execute_trade(self, decision):
        try:
            account = self.api.get_account()
            cash = float(account.cash)
            print(f"Account cash: {cash}")
        except Exception as e:
            print(f"Error fetching account details: {e}")
            return

        position = None
        try:
            position = self.api.get_position(self.symbol)
            print(f"Current position: {position.qty} shares")
        except tradeapi.rest.APIError as e:
            if 'position does not exist' in str(e):
                print("No current position in the symbol.")
                position = None
            else:
                raise e

        try:
            current_price = self.api.get_latest_trade(self.symbol).price
            print(f"Current price: {current_price}")
        except Exception as e:
            print(f"Error fetching latest trade: {e}")
            return

        print(f"Decision: {decision}")
        if decision == 'buy' and cash > current_price:
            qty = int(cash // current_price)
            if qty > 0:
                self.api.submit_order(
                    symbol=self.symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f'Bought {qty} shares of {self.symbol}')
            else:
                print("Not enough cash to buy any shares.")
        elif decision == 'sell' and position and int(position.qty) > 0:
            qty = int(position.qty)
            self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f'Sold {qty} shares of {self.symbol}')
        else:
            print('No trade executed')

    def run(self):
        data = self.fetch_historical_data()
        if not data.empty:
            data = self.calculate_indicators(data)
            decision = self.make_decision(data)
            self.execute_trade(decision)
        else:
            print('No data to process')


if __name__ == '__main__':
    bot = TradingBot()
    bot.run()
#######################ggggg
