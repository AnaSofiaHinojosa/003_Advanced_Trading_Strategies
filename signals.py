import ta
import pandas as pd


def momentum_indicators(df: pd.DataFrame,
                        rsi_window: int = 14,
                        rsi2_window: int = 10,
                        rsi3_window: int = 20,
                        kama_window: int = 30,
                        stoch_window: int = 14,
                        stoch2_window: int = 10,
                        stoch3_window: int = 20,
                        roc_window: int = 12,
                        roc2_window: int = 10,
                        roc3_window: int = 20,
                        williams_r_window: int = 14,
                        williams_r2_window: int = 7,
                        williams_r3_window: int = 21,
                        kama_pow1: int = 2,
                        kama_pow2: int = 30) -> pd.DataFrame:

    df = df.copy()

    # --- Momentum indicators ---
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()
    df['rsi2'] = ta.momentum.RSIIndicator(
        df['Close'], window=rsi2_window).rsi()
    df['rsi3'] = ta.momentum.RSIIndicator(
        df['Close'], window=rsi3_window).rsi()

    # KAMA
    df['kama'] = ta.momentum.KAMAIndicator(
        df['Close'], window=kama_window, pow1=kama_pow1, pow2=kama_pow2).kama()

    # Stochastic Oscillator
    df['stoch'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close'], window=stoch_window).stoch()
    df['stoch2'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close'], window=stoch2_window).stoch()
    df['stoch3'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close'], window=stoch3_window).stoch()

    # ROC
    df['roc'] = ta.momentum.ROCIndicator(df['Close'], window=roc_window).roc()
    df['roc2'] = ta.momentum.ROCIndicator(
        df['Close'], window=roc2_window).roc()
    df['roc3'] = ta.momentum.ROCIndicator(
        df['Close'], window=roc3_window).roc()

    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        df['High'], df['Low'], df['Close'], lbp=williams_r_window).williams_r()
    df['williams_r2'] = ta.momentum.WilliamsRIndicator(
        df['High'], df['Low'], df['Close'], lbp=williams_r2_window).williams_r()
    df['williams_r3'] = ta.momentum.WilliamsRIndicator(
        df['High'], df['Low'], df['Close'], lbp=williams_r3_window).williams_r()

    return df


def volatility_indicators(df: pd.DataFrame,
                          bb_window: int = 20,
                          donchian_window: int = 20,
                          kc_window: int = 20) -> pd.DataFrame:

    df = df.copy()

    # --- Volatility indicators ---

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=bb_window)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    # Keltner Channels
    kc = ta.volatility.KeltnerChannel(
        df['High'], df['Low'], df['Close'], window=kc_window)
    df['kc_upper'] = kc.keltner_channel_hband()
    df['kc_lower'] = kc.keltner_channel_lband()

    # Donchian Channels
    donchian = ta.volatility.DonchianChannel(
        df['High'], df['Low'], df['Close'], window=donchian_window)
    df['donchian_high'] = donchian.donchian_channel_hband()
    df['donchian_low'] = donchian.donchian_channel_lband()

    return df


def volume_indicators(df: pd.DataFrame,
                      mfi_window: int = 14,
                      eom_window: int = 14,
                      fi_window: int = 14) -> pd.DataFrame:
    # --- Volume indicators ---

    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        df['Close'], df['Volume']).on_balance_volume()

    # Accumulation/Distribution Line (A/D)
    df['ad'] = ta.volume.AccDistIndexIndicator(
        df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()

    # Chaikin Money Flow (CMF)
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
        df['High'], df['Low'], df['Close'], df['Volume'], window=mfi_window).chaikin_money_flow()

    # Ease of Movement (EOM)
    df['eom'] = ta.volume.EaseOfMovementIndicator(
        df['High'], df['Low'], df['Volume'], window=eom_window).ease_of_movement()

    # Force Index
    df['fi'] = ta.volume.ForceIndexIndicator(
        df['Close'], df['Volume'], window=fi_window).force_index()

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = momentum_indicators(df)
    df = volatility_indicators(df)
    df = volume_indicators(df)

    return df


def get_signals(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = normalize_indicators(df)

    df['price_shift'] = df['Close'].pct_change().dropna()
    df['final_signal'] = 0
    df.loc[df['price_shift'] > 0.01, 'final_signal'] = 1
    df.loc[df['price_shift'] < -0.01, 'final_signal'] = -1

    df.drop(columns=['price_shift'], inplace=True)

    return df

def normalize_indicators(df):
    df = df.copy()

    # --- RSI (0 to 100) → 0 to 1---
    for col in ['rsi', 'rsi2', 'rsi3']:
        df[col] = df[col] / 100

    # --- Stochastic Oscillator (0 to 100) → 0 to 1---
    for col in ['stoch', 'stoch2', 'stoch3']:
        df[col] = df[col] / 100

    # --- Williams %R (-100 to 0) → 0 to 1---
    for col in ['williams_r', 'williams_r2', 'williams_r3']:
        df[col] = (df[col] + 100) / 100

    # --- KAMA → relative to Close ---
    df['kama'] = df['Close'] / df['kama'] - 1

    # --- ROC (z-score) ---
    for col in ['roc', 'roc2', 'roc3']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # --- Bollinger Bands → position 0-1 ---
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # --- Keltner Channels → position 0-1 ---
    df['kc_position'] = (df['Close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])

    # --- Donchian Channels → position 0-1 ---
    df['donchian_position'] = (df['Close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'])

    # --- OBV, AD, EOM, FI (z-score) ---
    for col in ['obv', 'ad', 'eom', 'fi']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # --- CMF (-1 to 1) → 0 to 1 ---
    df['cmf'] = (df['cmf'] + 1) / 2

    return df
