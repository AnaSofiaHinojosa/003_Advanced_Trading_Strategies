from models import Operation

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price:float, n_shares: int, COM: float) -> float:
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

def backtest(data):
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
                pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
                cash += pnl
                # Add to win/loss count
                if pnl >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_short_positions.remove(position)
                continue

        # --- BUY ---
        # Check signal
        if row.buy_signal:
            position_value = row.Close * n_shares * (1 + COM)
            # Do we have enough cash?
            if cash > position_value:
                # Discount the cost
                cash -= position_value
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
        if row.sell_signal:
            position_value = row.Close * n_shares
            short_cost = position_value * COM
            # Do we have enough cash?
            if cash > short_cost:
                # Discount the cost
                cash -= short_cost
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
                
        portfolio_value.append(get_portfolio_value(cash, active_long_positions, active_short_positions, row.Close, n_shares, COM))

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
        pnl = (position.price - row.Close) * position.n_shares * (1 - COM)
        cash += pnl
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

    return cash, portfolio_value, win_rate, total_trades