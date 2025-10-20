import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_value: list) -> None:
    """
    Plot the portfolio value over time.

    Parameters:
        portfolio_value (list): List of portfolio values over time.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label="Portfolio value", color="palevioletred")
    plt.title("Portfolio value over time")
    plt.xlabel("Time")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()
