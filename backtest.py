from models import Operation
from datadrift import calculate_drift_metrics, calculate_drift_pvalues

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price:float, n_shares: int) -> float:
    """
    Calculate the total portfolio value including cash and open positions.

    Parameters:
        cash (float): Current cash available.
        long_ops (list[Operation]): List of active long positions.
        short_ops (list[Operation]): List of active short positions.
        current_price (float): Current price of the asset.
        n_shares (int): Number of shares per operation.
    
    Returns:
        float: Total portfolio value.
    """
    
    val = cash

    # Add long positions value
    for pos in long_ops:
        pnl = current_price * pos.n_shares
        val += pnl

    # Add short positions value
    for pos in short_ops:
        pnl = (pos.price - current_price) * pos.n_shares
        val += pnl

    return val

def backtest(data, reference_features=None, compare_features=None):
    """
    Backtest trading strategy based on generated signals.
    
    Parameters:
        data (DataFrame): DataFrame containing historical price data and signals.
        reference_features (DataFrame, optional): Reference features for drift detection.
        compare_features (DataFrame, optional): Features to compare for drift detection.

    Returns:
        tuple: Final cash, portfolio value over time, win rate, buy count, sell count, hold count, total trades,
               data drift results, p-values results.
    """

    # Trade params
    stop_loss = 0.07
    take_profit = 0.14
    n_shares = 100

    # Signals
    historic = data.copy()
    historic = historic.dropna()

    # Params
    SL = stop_loss
    TP = take_profit

    COM = 0.125 / 100
    BORROW_RATE = 0.25 / 100

    cash = 1_000_000

    INTERVALS = 252 # daily intervals
    BORROW_RATE_DAILY = BORROW_RATE / INTERVALS

    # Backtest logic
    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    portfolio_value = []

    positive_trades = 0
    negative_trades = 0

    buy = 0
    sell = 0
    hold = 0

    # Drift detection params
    window_size = 90
    window_step = 30

    data_drift_results: list[dict] = []
    p_values_results: list[dict] = []  

    for i, row in enumerate(historic.itertuples(index=True)):
        # Close positions
        for position in active_long_positions.copy():
            # Check take profit or stop loss
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
                cash += row.Close * position.n_shares * (1 - COM)
                # Add to win/loss count
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_long_positions.remove(position)

        # Substract borrow cost
        for position in active_short_positions.copy():
            # Check take profit or stop loss
            cover_cost = row.Close * position.n_shares
            borrow_cost = cover_cost * BORROW_RATE_DAILY
            cash -= borrow_cost
        
        # Close positions
        for position in active_short_positions.copy():
            if row.Close < position.take_profit or row.Close > position.stop_loss:
                pnl = (position.price - row.Close) * position.n_shares
                short_com = row.Close * position.n_shares * COM
                cash += pnl - short_com
                # Add to win/loss count
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_short_positions.remove(position)
                continue

        # --- BUY ---
        # Check signal
        if row.final_signal == 1:
            position_value = row.Close * n_shares * (1 + COM)
            # Do we have enough cash?
            if cash > position_value:
                # Discount the cost
                cash -= position_value
                # Count the buy
                buy += 1
                # Save the operation as active position
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

        # --- SELL ---
        # Check signal
        if row.final_signal == -1:
            position_value = row.Close * n_shares
            short_cost = position_value * COM
            # Do we have enough cash?
            if cash > short_cost:
                # Discount the cost
                cash -= short_cost
                # Count the sell
                sell += 1
                # Save the operation as active position
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
        else:
            # Count the hold
            hold += 1        
                
        portfolio_value.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, n_shares))

        j = i + 1
        # Drift detection
        if reference_features is not None and compare_features is not None:
            if j >= window_size and (j - window_size) % window_step == 0:
                start_window = j - window_size
                end_window = j

                # Add dataframe with established window
                df_with_window = compare_features.iloc[start_window:end_window]

                # Calculate drift metrics
                drift_metrics = calculate_drift_metrics(reference_features, df_with_window)

                # Calculate p-values
                p_values = calculate_drift_pvalues(reference_features, df_with_window)

                feature_snapshot = {
                    "train": reference_features.to_dict(orient="records"),
                    "test":  compare_features.to_dict(orient="records"),
                    "val":   df_with_window.to_dict(orient="records")
                }
                
                # Store results
                data_drift_results.append(drift_metrics)
                p_values_results.append(p_values)
                dashboard_snapshot = feature_snapshot

    # Close long positions        
    for position in active_long_positions:
        pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
        cash += row.Close * position.n_shares * (1 - COM)
        # Add to win/loss count
        if pnl >= 0:
            positive_trades += 1
        else:
            negative_trades += 1
    
    for position in active_short_positions:
        pnl = (position.price - row.Close) * position.n_shares
        short_com = row.Close * position.n_shares * COM
        cash += pnl - short_com
        # Add to win/loss count
        if pnl >= 0:
            positive_trades += 1
        else:
            negative_trades += 1

    active_long_positions = []
    active_short_positions = []

    # Calculate win rate
    win_rate = positive_trades / (positive_trades + negative_trades) if (positive_trades + negative_trades) > 0 else 0
    total_trades = positive_trades + negative_trades

    return cash, portfolio_value, win_rate, buy, sell, hold, total_trades, data_drift_results, p_values_results
