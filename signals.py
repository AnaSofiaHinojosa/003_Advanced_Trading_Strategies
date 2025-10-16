import ta
import pandas as pd


def momentum_indicators(df: pd.DataFrame,
                        rsi_window: int = 14,
                        kama_window: int = 30,
                        stoch_window: int = 14,
                        roc_window: int = 12,
                        williams_r_window: int = 14,
                        kama_pow1: int = 2,
                        kama_pow2: int = 30) -> pd.DataFrame:

    df = df.copy()

    # --- Momentum indicators ---
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()

    # KAMA
    df['kama'] = ta.momentum.KAMAIndicator(
        df['Close'], window=kama_window, pow1=kama_pow1, pow2=kama_pow2).kama()

    # Stochastic Oscillator
    df['stoch'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close'], window=stoch_window).stoch()

    # ROC
    df['roc'] = ta.momentum.ROCIndicator(df['Close'], window=roc_window).roc()

    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        df['High'], df['Low'], df['Close'], lbp=williams_r_window).williams_r()

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


def get_signals(df: pd.DataFrame,
                rsi_buy: float = 30,
                rsi_sell: float = 70,
                stoch_buy: float = 20,
                stoch_sell: float = 80,
                roc_sma_window: int = 5,
                williams_buy: float = -80,
                williams_sell: float = -20,
                obv_percentage: float = 0.1,
                ad_ma_window: int = 20,
                cmf_buy: float = 30,
                cmf_sell: float = 70,
                eom_window: int = 20,
                fi_percentage: float = 0.1,
                number_agreement_buy: int = 5,
                number_agreement_sell: int = 5) -> pd.DataFrame:

    df = df.copy()

    df['buy_signal_rsi'] = df['rsi'] < rsi_buy
    df['sell_signal_rsi'] = df['rsi'] > rsi_sell

    df['buy_signal_kama'] = df['Close'] > df['kama']
    df['sell_signal_kama'] = df['Close'] < df['kama']

    df['buy_signal_stoch'] = df['stoch'] < stoch_buy
    df['sell_signal_stoch'] = df['stoch'] > stoch_sell

    df['roc_sma'] = df['roc'].rolling(roc_sma_window).mean()
    df['buy_signal_roc'] = df['roc_sma'] > 0
    df['sell_signal_roc'] = df['roc_sma'] < 0

    df['buy_signal_williams_r'] = df['williams_r'] < williams_buy
    df['sell_signal_williams_r'] = df['williams_r'] > williams_sell

    df['buy_signal_bb'] = df['Close'] < df['bb_lower']
    df['sell_signal_bb'] = df['Close'] > df['bb_upper']

    df['buy_signal_kc'] = df['Close'] < df['kc_lower']
    df['sell_signal_kc'] = df['Close'] > df['kc_upper']

    df['buy_signal_donchian'] = df['Close'] < df['donchian_low']
    df['sell_signal_donchian'] = df['Close'] > df['donchian_high']

    obv_diff = df['obv'] - df['obv'].shift(1)
    obv_threshold = df['obv'].std(skipna=True) * obv_percentage
    df['buy_signal_obv'] = obv_diff > obv_threshold
    df['sell_signal_obv'] = obv_diff < -obv_threshold

    df['ad_ma'] = df['ad'].rolling(ad_ma_window).mean()
    df['buy_signal_ad'] = df['ad'] > df['ad_ma']
    df['sell_signal_ad'] = df['ad'] < df['ad_ma']

    df['buy_signal_cmf'] = df['cmf'] < cmf_buy
    df['sell_signal_cmf'] = df['cmf'] > cmf_sell

    df['eom_ma'] = df['eom'].rolling(eom_window).mean()
    df['buy_signal_eom'] = df['eom'] > df['eom_ma']
    df['sell_signal_eom'] = df['eom'] < df['eom_ma']

    fi_threshold = df['fi'].std() * fi_percentage
    df['buy_signal_fi'] = df['fi'] > fi_threshold
    df['sell_signal_fi'] = df['fi'] < -fi_threshold

    buy_cols = [col for col in df.columns if 'buy_signal' in col]
    sell_cols = [col for col in df.columns if 'sell_signal' in col]

    df['buy_count'] = df[buy_cols].sum(axis=1)
    df['sell_count'] = df[sell_cols].sum(axis=1)

    df['buy_signal'] = df['buy_count'] >= number_agreement_buy
    df['sell_signal'] = df['sell_count'] >= number_agreement_sell

    return df
