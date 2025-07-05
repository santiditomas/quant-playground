import pandas as pd

def add_sma(data: pd.Series, window = int) -> pd.Series:
    return data.rolling(window=window).mean()
