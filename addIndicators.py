import numpy as np
import pandas as pd

def add_rsi(df, period=14):
    # Calculate price differences
    delta = df['close'].diff()
    
    # Gains (positive differences) and losses (negative differences)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Overbought'] = (df['RSI'] > 60).astype(int)
    df['RSI_Oversold'] = (df['RSI'] < 40).astype(int)
    return df

def add_ema(df, period):
    ema_column = f'EMA_{period}'
    df[ema_column] = df['close'].ewm(span=period, adjust=False).mean()

    # Set first (period-1) rows to NaN since EMA requires 'period' data points
    df.iloc[:period-1, df.columns.get_loc(ema_column)] = None
    return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    # Compute MACD Line and Signal Line (not adding them to df)
    macd_line = df['close'].ewm(span=fast_period, adjust=False).mean() - df['close'].ewm(span=slow_period, adjust=False).mean()
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Compute MACD Histogram (Only storing this in df)
    df['MACD_Histogram'] = macd_line - macd_signal
    
    # Momentum Feature: Rate of Change in MACD Histogram
    df['MACD_Histogram_Change'] = df['MACD_Histogram'].diff()
    
    # Lagged Features
    df['MACD_Histogram_Lag_1'] = df['MACD_Histogram'].shift(1)
    return df

def add_volume_filter(data, slow_ma, fast_ma):
    if 'volume' not in data.columns:
        raise KeyError("Column 'volume' not found in the DataFrame")
    # Volume Moving Averages
    vma_fast = data['volume'].rolling(window=fast_ma).mean() 
    vma_slow = data['volume'].rolling(window=slow_ma).mean()  

    # Volume filter for buy/sell signals
    volume_condition = (vma_fast > vma_slow * 1.05) & (vma_fast < vma_slow * 2.2)
    data["volume_condition"] = volume_condition.astype(int)
    return data

def calculate_atr(high, low, close, period):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def add_ut_bot(data, key_value=2, atr_period=10):
    src = data['close']

    atr = calculate_atr(data['high'].values, data['low'].values, data['close'].values, atr_period)
    n_loss = key_value * atr

    x_atr_trailing_stop = pd.Series(index=data.index, dtype='float64')
    pos = np.zeros(len(data))

    for i in range(1, len(data)):
        prev_stop = x_atr_trailing_stop.iloc[i - 1]
        if src.iloc[i] > prev_stop and src.iloc[i - 1] > prev_stop:
            x_atr_trailing_stop.iloc[i] = max(prev_stop, src.iloc[i] - n_loss.iloc[i])
        elif src.iloc[i] < prev_stop and src.iloc[i - 1] < prev_stop:
            x_atr_trailing_stop.iloc[i] = min(prev_stop, src.iloc[i] + n_loss.iloc[i])
        else:
            x_atr_trailing_stop.iloc[i] = src.iloc[i] - n_loss.iloc[i] if src.iloc[i] > prev_stop else src.iloc[i] + n_loss.iloc[i]

        if src.iloc[i - 1] < prev_stop and src.iloc[i] > prev_stop:
            pos[i] = 1  # Buy signal
        elif src.iloc[i - 1] > prev_stop and src.iloc[i] < prev_stop:
            pos[i] = -1  # Sell signal
        else:
            pos[i] = pos[i - 1]

    ema = src.ewm(span=1, adjust=False).mean()
    above = (ema > x_atr_trailing_stop) & (ema.shift(1) <= x_atr_trailing_stop.shift(1))
    below = (ema < x_atr_trailing_stop) & (ema.shift(1) >= x_atr_trailing_stop.shift(1))

    buy = (src > x_atr_trailing_stop) & above
    sell = (src < x_atr_trailing_stop) & below  

    data['my_indicator'] = np.where(buy, 'BUY', np.where(sell, 'SELL', 'HOLD'))
    
    return data

def add_choppiness_index(df, length=14):
    prev_close = df['close'].shift(1)
    # Calculate True Range (TR)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - prev_close),
                               np.abs(df['low'] - prev_close)))
    
    # Compute rolling sum of TR and rolling range (max high - min low)
    tr_sum = tr.rolling(window=length).sum()
    rolling_range = df['high'].rolling(window=length).max() - df['low'].rolling(window=length).min()
    
    # Calculate CHOP and avoid division by zero
    chop = 100 * np.log10(tr_sum / rolling_range) / np.log10(length)
    chop = chop.where(rolling_range != 0, np.nan)
    
    df['CHOP'] = chop
    return df

def add_cci(df, period=20):
    # Commodity Channel Index in a single column
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (tp - sma) / (0.015 * mad)
    return df

def add_roc(df, period=12):
    # Rate of Change indicator as a single column
    df['ROC'] = df['close'].pct_change(periods=period) * 100
    return df

def add_williams_r(df, period=14):
    # Williams %R indicator in a single column
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    df['Williams_R'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    return df

def add_mfi(df, period=14):
    """Adds Money Flow Index (MFI) to the dataset."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    
    pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    pos_mf = pos_flow.rolling(window=period).sum()
    neg_mf = neg_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    df['MFI'] = mfi
    return df

def add_tsi(df, long_period=25, short_period=13):
    """Adds True Strength Index (TSI) to the dataset."""
    price_change = df['close'].diff()
    abs_change = price_change.abs()

    smoothed_price = price_change.ewm(span=short_period, adjust=False).mean()
    smoothed_price = smoothed_price.ewm(span=long_period, adjust=False).mean()

    smoothed_abs = abs_change.ewm(span=short_period, adjust=False).mean()
    smoothed_abs = smoothed_abs.ewm(span=long_period, adjust=False).mean()

    tsi = (smoothed_price / smoothed_abs) * 100
    df['TSI'] = tsi
    return df

def add_keltner_channels(df, period=20, atr_multiplier=2):
    """Adds Keltner Channels (Upper, Middle, Lower) to the dataset."""
    ema = df['close'].ewm(span=period, adjust=False).mean()
    atr = calculate_atr(df['high'], df['low'], df['close'], period)

    df['Keltner_Upper'] = ema + (atr_multiplier * atr)
    df['Keltner_Lower'] = ema - (atr_multiplier * atr)
    return df

def add_adl(df):
    """Adds Accumulation/Distribution Line (ADL) to the dataset."""
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_volume = mf_multiplier * df['volume']
    df['ADL'] = mf_volume.cumsum()
    return df

def add_dmi_adx(df, period=14):
    """Adds Directional Movement Index (DMI) and Average Directional Index (ADX) to the dataset."""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    atr = calculate_atr(df['high'], df['low'], df['close'], period)

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()

    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = adx
    return df

def add_ema_features(df, periods=[9, 21, 50, 200]):
    for period in periods:
        ema_col = f'EMA_{period}'
        df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()

        # Feature 1: Price distance from EMA
        df[f'Price_{ema_col}_Diff'] = df['close'] - df[ema_col]
    
    return df