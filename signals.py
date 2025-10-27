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
    """
    Calculate momentum indicators and add them to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with price data.
        rsi_window (int): Window size for RSI calculation.
        rsi2_window (int): Window size for second RSI calculation.
        rsi3_window (int): Window size for third RSI calculation.
        kama_window (int): Window size for KAMA calculation.
        stoch_window (int): Window size for Stochastic Oscillator calculation.
        stoch2_window (int): Window size for second Stochastic Oscillator calculation.
        stoch3_window (int): Window size for third Stochastic Oscillator calculation.
        roc_window (int): Window size for ROC calculation.
        roc2_window (int): Window size for second ROC calculation.
        roc3_window (int): Window size for third ROC calculation.
        williams_r_window (int): Window size for Williams %R calculation.
        williams_r2_window (int): Window size for second Williams %R calculation.
        williams_r3_window (int): Window size for third Williams %R calculation.
        kama_pow1 (int): Power 1 for KAMA calculation.
        kama_pow2 (int): Power 2 for KAMA calculation.

    Returns:
        pd.DataFrame: DataFrame with added momentum indicators.
    """

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
    """
    Calculate volatility indicators and add them to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with price data.
        bb_window (int): Window size for Bollinger Bands calculation.
        donchian_window (int): Window size for Donchian Channels calculation.
        kc_window (int): Window size for Keltner Channels calculation.

    Returns:
        pd.DataFrame: DataFrame with added volatility indicators.
    """

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
    """
    Calculate volume indicators and add them to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with price data.
        mfi_window (int): Window size for Money Flow Index calculation.
        eom_window (int): Window size for Ease of Movement calculation.
        fi_window (int): Window size for Force Index calculation.

    Returns:
        pd.DataFrame: DataFrame with added volume indicators.
    """

    df = df.copy()

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
    """
    Add all technical indicators to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with price data.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """

    df = df.copy()

    df = momentum_indicators(df)
    df = volatility_indicators(df)
    df = volume_indicators(df)

    return df


def get_signals(df: pd.DataFrame, alpha: float = 0.02) -> pd.DataFrame:
    """
    Generate trading signals based on future price movements.

    Parameters:
        df (pd.DataFrame): Input DataFrame with price data.
        alpha (float): Threshold for generating buy/sell signals.

    Returns:
        pd.DataFrame: DataFrame with added trading signals.
    """

    df = df.copy()

    # 7 day shifted column (future)
    df['future_price'] = df['Close'].shift(-10)

    # Initialize all signals to 0
    df['final_signal'] = 0

    # Buy signals
    df.loc[df[df.columns[0]] *
           (1+alpha) < df['future_price'], 'final_signal'] = 1

    # Sell signals
    df.loc[df[df.columns[0]] *
           (1-alpha) > df['future_price'], 'final_signal'] = -1

    df.drop(columns=['future_price'], inplace=True)

    return df
