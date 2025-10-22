import pandas as pd
from models import Operation
from datadrift import calculate_drift_metrics, get_most_drifted_features


def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation],
                        current_price: float, n_shares: int) -> float:
    """Calculate total portfolio value from cash + open positions."""
    val = cash

    # Long positions
    for pos in long_ops:
        pnl = current_price * pos.n_shares
        val += pnl

    # Short positions
    for pos in short_ops:
        pnl = (pos.price - current_price) * pos.n_shares
        val += pnl

    return val


def backtest(data, reference_features=None, window_size: int = 100, slide_size: int = 50):
    """Run a backtest with KS-based data drift detection."""
    # --- Trading parameters ---
    stop_loss = 0.07
    take_profit = 0.14
    n_shares = 100
    SL, TP = stop_loss, take_profit
    COM = 0.125 / 100
    BORROW_RATE = 0.25 / 100
    INTERVALS = 252  # trading days per year
    BORROW_RATE_DAILY = BORROW_RATE / INTERVALS
    cash = 1_000_000

    # --- Prepare data ---
    historic = data.copy().dropna()

    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    portfolio_value = []
    positive_trades = 0
    negative_trades = 0
    buy = 0
    sell = 0

    # --- Drift tracking ---
    drift_results = []
    full_drift_df = pd.DataFrame()

    # --- Main backtest loop ---
    for i, row in enumerate(historic.itertuples(index=True)):
        # --- Close long positions ---
        for position in active_long_positions.copy():
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
                cash += row.Close * position.n_shares * (1 - COM)
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_long_positions.remove(position)

        # --- Borrow cost for short positions ---
        for position in active_short_positions.copy():
            cover_cost = row.Close * position.n_shares
            borrow_cost = cover_cost * BORROW_RATE_DAILY
            cash -= borrow_cost

        # --- Close short positions ---
        for position in active_short_positions.copy():
            if row.Close < position.take_profit or row.Close > position.stop_loss:
                pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
                cash += pnl
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_short_positions.remove(position)

        # --- Open long position (BUY) ---
        if row.final_signal == 1:
            position_value = row.Close * n_shares * (1 + COM)
            if cash > position_value:
                cash -= position_value
                buy += 1
                active_long_positions.append(
                    Operation(
                        time=row.Index,
                        price=row.Close,
                        take_profit=row.Close * (1 + TP),
                        stop_loss=row.Close * (1 - SL),
                        n_shares=n_shares,
                        type="LONG"
                    )
                )

        # --- Open short position (SELL) ---
        if row.final_signal == -1:
            position_value = row.Close * n_shares
            short_cost = position_value * COM
            if cash > short_cost:
                cash -= short_cost
                sell += 1
                active_short_positions.append(
                    Operation(
                        time=row.Index,
                        price=row.Close,
                        take_profit=row.Close * (1 - TP),
                        stop_loss=row.Close * (1 + SL),
                        n_shares=n_shares,
                        type="SHORT"
                    )
                )

        # --- Portfolio value update ---
        portfolio_value.append(
            get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, n_shares)
        )

        # --- Drift detection (KS test) ---
        if reference_features is not None and i + 1 >= window_size + slide_size:
            if (i + 1 - window_size) % slide_size == 0:
                current_window = historic.iloc[i + 1 - window_size:i + 1][reference_features.columns]
                drift_metrics = calculate_drift_metrics(
                    reference_df=reference_features,
                    new_df=current_window,
                    alpha=0.05
                )
                drift_metrics['start_idx'] = i + 1 - window_size
                drift_metrics['end_idx'] = i + 1
                drift_results.append(drift_metrics)

    # --- Combine drift results ---
    if drift_results:
        full_drift_df = pd.DataFrame(drift_results)
        full_drift_df = full_drift_df.drop_duplicates(subset=['start_idx', 'end_idx']).reset_index(drop=True)
        most_drifted_features = get_most_drifted_features(full_drift_df)
        print(f"Most drifted features: {most_drifted_features}")
        print(f"Values of most drifted features")
    else:
        most_drifted_features = []

    # --- Close any remaining positions ---
    for position in active_long_positions:
        pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
        cash += row.Close * position.n_shares * (1 - COM)
        if pnl >= 0:
            positive_trades += 1
        else:
            negative_trades += 1

    for position in active_short_positions:
        pnl = (position.price - row.Close) * position.n_shares
        short_com = row.Close * position.n_shares * COM
        cash += pnl - short_com
        if pnl >= 0:
            positive_trades += 1
        else:
            negative_trades += 1

    # --- Metrics ---
    win_rate = positive_trades / (positive_trades + negative_trades) if (positive_trades + negative_trades) > 0 else 0
    total_trades = positive_trades + negative_trades

    return cash, portfolio_value, win_rate, buy, sell, total_trades, full_drift_df
