from dataclasses import dataclass

@dataclass
class Operation:
    """
    A class to represent a trading operation.
    """
    
    time: str
    price: float
    stop_loss: float
    take_profit: float
    n_shares: int
    type: str
