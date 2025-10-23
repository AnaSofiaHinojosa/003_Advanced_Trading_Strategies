import pandas as pd
from typing import Dict, Tuple

def normalize_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()

    params = {}

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
    params['min_kama'] = df['kama'].min()
    params['max_kama'] = df['kama'].max()
    df['kama'] = (df['kama'] - params['min_kama']) / (params['max_kama'] - params['min_kama'])

    # --- ROC (z-score) ---
    for col in ['roc', 'roc2', 'roc3']:
        params[f'mean_{col}'] = df[col].mean()
        params[f'std_{col}'] = df[col].std()
        df[col] = (df[col] - params[f'mean_{col}']) / params[f'std_{col}']

    # --- Bollinger Bands → position 0-1 ---
    df['bb_position'] = (df['Close'] - df['bb_lower']) / \
        (df['bb_upper'] - df['bb_lower'])

    # --- Keltner Channels → position 0-1 ---
    df['kc_position'] = (df['Close'] - df['kc_lower']) / \
        (df['kc_upper'] - df['kc_lower'])

    # --- Donchian Channels → position 0-1 ---
    df['donchian_position'] = (
        df['Close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'])

    # --- OBV, AD, EOM, FI (z-score) ---
    for col in ['obv', 'ad', 'eom', 'fi']:
        params[f'mean_{col}'] = df[col].mean()
        params[f'std_{col}'] = df[col].std()
        df[col] = (df[col] - params[f'mean_{col}']) / params[f'std_{col}']

    # --- CMF (-1 to 1) → 0 to 1 ---
    params['min_cmf'] = df['cmf'].min()
    params['max_cmf'] = df['cmf'].max()
    df['cmf'] = (df['cmf'] - params['min_cmf']) / (params['max_cmf'] - params['min_cmf'])

    # --- Close ---
    params['mean_close'] = df['Close'].mean()
    params['std_close'] = df['Close'].std()
    df['Close'] = (df['Close'] - params['mean_close']) / params['std_close']

    df = df.drop(columns=['bb_upper', 'bb_lower',
                          'kc_upper', 'kc_lower',
                          'donchian_high', 'donchian_low'])
    return df, params

def normalize_new_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
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
    df['kama'] = (df['kama'] - params['min_kama']) / (params['max_kama'] - params['min_kama'])

    # --- ROC (z-score) ---
    for col in ['roc', 'roc2', 'roc3']:
        df[col] = (df[col] - params[f'mean_{col}']) / params[f'std_{col}']

    # --- Bollinger Bands → position 0-1 ---
    df['bb_position'] = (df['Close'] - df['bb_lower']) / \
        (df['bb_upper'] - df['bb_lower'])

    # --- Keltner Channels → position 0-1 ---
    df['kc_position'] = (df['Close'] - df['kc_lower']) / \
        (df['kc_upper'] - df['kc_lower'])

    # --- Donchian Channels → position 0-1 ---
    df['donchian_position'] = (
        df['Close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'])

    # --- OBV, AD, EOM, FI (z-score) ---
    for col in ['obv', 'ad', 'eom', 'fi']:
        df[col] = (df[col] - params[f'mean_{col}']) / params[f'std_{col}']

    # --- CMF (-1 to 1) → 0 to 1 ---
    df['cmf'] = (df['cmf'] - params['min_cmf']) / (params['max_cmf'] - params['min_cmf'])

    # --- Close ---
    df['Close'] = (df['Close'] - params['mean_close']) / params['std_close']

    df = df.drop(columns=['bb_upper', 'bb_lower',
                        'kc_upper', 'kc_lower',
                        'donchian_high', 'donchian_low'])

    return df
