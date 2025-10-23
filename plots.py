import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_value: list, section: str) -> None:
    """
    Plot the portfolio value over time.

    Parameters:
        portfolio_value (list): List of portfolio values over time.
    """

    colors = {'train': 'palevioletred', 'test': 'cadetblue', 'val': 'steelblue'}
    color = colors[section]

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, color=color)
    plt.title(f'Portfolio value over time ({section})')
    plt.xlabel("Time")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()

def plot_trade_distribution(buy: int, sell: int, hold: int, section: str) -> None:
    """
    Plot the distribution of buy and sell trades.

    Parameters:
        buy (int): Number of buy trades.
        sell (int): Number of sell trades.
        hold (int): Number of hold trades.
        section (str): Section name (train, test, val).
    """

    colors = ['palevioletred', 'cadetblue', 'steelblue']

    labels = ['Buy', 'Sell', 'Hold']
    values = [buy, sell, hold]
    explode = (0.02, 0.02, 0.02)

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140, explode=explode)
    plt.title(f'Trade Distribution ({section})')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()