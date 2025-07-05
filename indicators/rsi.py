import pandas as pd

def rsi(data: pd.Series, window = int) -> pd.Series:

    delta = data.diff()

    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0 
    down[down < 0] = 0

    _gain = up.ewm(alpha=1.0 / window, adjust=True).mean()
    _loss = down.abs().ewm(alpha=1.0 / window, adjust=True).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)))



