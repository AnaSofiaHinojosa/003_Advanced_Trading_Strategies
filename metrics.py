import numpy as np
import pandas as pd


def sharpe_ratio(portfolio_values: pd.Series) -> float:
    """
    Calculate the Sharpe ratio of a portfolio.

    Parameters:
        portfolio_values (pd.Series): Series of portfolio values over time.

    Returns:
        float: Sharpe ratio of the portfolio.
    """

    # Daily
    returns = portfolio_values.pct_change().dropna()
    mean = returns.mean()
    std = returns.std()

    # Annualized
    intervals = 252
    annual_rets = mean * intervals
    annual_std = std * np.sqrt(intervals)

    return annual_rets / annual_std if annual_std > 0 else 0


def sortino_ratio(portfolio_values: pd.Series) -> float:
    """
    Calculate the Sortino ratio of a portfolio.

    Parameters:
        portfolio_values (pd.Series): Series of portfolio values over time.

    Returns:
        float: Sortino ratio of the portfolio.
    """

    # Daily
    returns = portfolio_values.pct_change().dropna()
    mean = returns.mean()
    downside = np.minimum(returns, 0).std()

    # Annualized
    intervals = 252
    annual_rets = mean * intervals
    annual_downside = downside * np.sqrt(intervals)

    return annual_rets / annual_downside if annual_downside > 0 else 0


def max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a portfolio.

    Parameters:
        portfolio_values (pd.Series): Series of portfolio values over time.

    Returns:
        float: Maximum drawdown of the portfolio.
    """

    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    return abs(max_dd)


def calmar_ratio(portfolio_values: pd.Series) -> float:
    """
    Calculate the Calmar ratio of a portfolio.

    Parameters:
        portfolio_values (pd.Series): Series of portfolio values over time.

    Returns:
        float: Calmar ratio of the portfolio.
    """

    # Daily
    returns = portfolio_values.pct_change().dropna()
    mean = returns.mean()

    # Annualized
    intervals = 252
    annual_rets = mean * intervals

    # Max Drawdown
    mdd = max_drawdown(portfolio_values)

    return annual_rets / mdd if mdd > 0 else 0


def evaluate_metrics(portfolio_values: pd.Series) -> pd.DataFrame:
    """
    Evaluate key performance metrics of a portfolio.

    Parameters:
        portfolio_values (pd.Series): Series of portfolio values over time.

    Returns:
        pd.DataFrame: DataFrame containing Sharpe ratio, Sortino ratio, Max drawdown, and Calmar ratio.
    """

    metrics = {
        'Sharpe ratio': sharpe_ratio(portfolio_values),
        'Sortino ratio': sortino_ratio(portfolio_values),
        'Max drawdown': max_drawdown(portfolio_values),
        'Calmar ratio': calmar_ratio(portfolio_values)
    }

    return pd.DataFrame([metrics], index=['Value'])


def win_rate(win_rate_test, total_trades_test, win_rate_validation, total_trades_validation) -> float:
    """
    Calculate the total win rate from test and validation win rates and total trades.

    Parameters:
        win_rate_test (float): Win rate from the test set.
        total_trades_test (int): Total trades from the test set.
        win_rate_validation (float): Win rate from the validation set.
        total_trades_validation (int): Total trades from the validation set.

    Returns:
        float: Combined win rate.
    """

    total_wins = (win_rate_test * total_trades_test) + \
        (win_rate_validation * total_trades_validation)
    total_trades = total_trades_test + total_trades_validation
    total_win_rate = total_wins / total_trades if total_trades > 0 else 0

    return total_win_rate
