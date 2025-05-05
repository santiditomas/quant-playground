import pandas as pd

def moving_average(data: pd.Series, window = int) -> pd.Series:
    return data.rolling(window=window).mean()
