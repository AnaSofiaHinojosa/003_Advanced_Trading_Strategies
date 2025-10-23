import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_value: list, section) -> None:
    """
    Plot the portfolio value over time.

    Parameters:
        portfolio_value (list): List of portfolio values over time.
    """

    colors = {'train': 'palevioletred', 'test': 'cadetblue', 'validation': 'mediumpurple'}
    color = colors[section]

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, color=color)
    plt.title(f'Portfolio value over time ({section})')
    plt.xlabel("Time")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()
