import pandas as pd
import ccxt
from datetime import datetime, timedelta
from addIndicators import *

def convert_timestamp_to_datetime(timestamp):
    return datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

def mark_labels(df):
    df = df.reset_index() 
    n = len(df)
    for i in range(n):
        signal = df.loc[i, 'my_indicator']
        if signal not in ['BUY', 'SELL']:
            continue
        entry_price = df.loc[i, 'close']
        j = i + 200 #check next 200 candles
        if(j>n): 
            j = n-1
        # while j < n and df.loc[j, 'my_indicator'] == 'HOLD':
        #     j += 1
        if signal == 'BUY':
            stop_loss = entry_price * 0.99
            partial_tp = entry_price * 1.01
            full_tp = entry_price * 1.025
        else:
            stop_loss = entry_price * 1.01
            partial_tp = entry_price * 0.99
            full_tp = entry_price * 0.975
        label = -1
        for k in range(i + 1, j):
            low_val = df.loc[k, 'low']
            high_val = df.loc[k, 'high']
            if signal == 'BUY':
                if low_val <= stop_loss:
                    label = -1
                    break
                if high_val >= full_tp:
                    label = 1
                    break
                if high_val >= partial_tp:
                    label = 0.5
            else:
                if high_val >= stop_loss:
                    label = -1
                    break
                if low_val <= full_tp:
                    label = 1
                    break
                if low_val <= partial_tp:
                    label = 0.5
        
        df.loc[i, 'correct'] = label
    return df


def fetch_binance_data(symbol, days, timeframe, limit=100000):
    """
    Fetch historical 1-hour candle data from Binance for the past 'days'.

    Parameters:
    - symbol: The trading pair (e.g., "BTC/USDT")
    - days: Number of days of historical data to fetch
    - limit: Maximum number of candles per request (default: 1000)

    Returns:
    - DataFrame containing the historical data
    """
    exchange = ccxt.binance()
    end_time = int(pd.Timestamp.now().timestamp() * 1000)  # Current timestamp in milliseconds
    start_time = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)  # Start timestamp in milliseconds
    all_data = []

    while start_time < end_time:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_time, limit=limit)
        if not ohlcv:
            break  # Stop if no more data is returned

        # Convert to DataFrame
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        all_data.append(data)
        print("Total records fetched : " + str(len(all_data)) + "for " + str(start_time))
        # Update `start_time` to the timestamp of the last candle to fetch the next batch
        start_time = ohlcv[-1][0] + 1  # Move to the next timestamp

        # Break if the last fetched timestamp exceeds `end_time`
        if ohlcv[-1][0] >= end_time:
            break

    # Combine all fetched data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'], unit='ms')
    combined_data.set_index('timestamp', inplace=True)
    return combined_data

symbol = "BTCUSDT"
days = 400
timeframe = "5m"
file_name = symbol + "_" + timeframe + "_" + str(days) + "d"

data = fetch_binance_data(symbol, days, timeframe)
print(type(data))
print(data.head())
data = add_rsi(data, 14)
# data = add_ema(data, 9)
# data = add_ema(data, 21)
# data = add_ema(data, 50)
# data = add_ema(data, 200)
data = add_ema_features(data)
data = add_macd(data)
data = add_volume_filter(data, 16, 4)
data = add_choppiness_index(data, 14)
#data = add_cci(data)
#data = add_roc(data)
#data = add_williams_r(data)
#data = add_mfi(data)
#data = add_tsi(data)
#data = add_keltner_channels(data)
#data = add_adl(data)
#data = add_dmi_adx(data)
data = add_ut_bot(data, 2, 8)

data = mark_labels(data)

data.to_excel(file_name + ".xlsx")
print("File created with data == " + file_name + ".xlsx")