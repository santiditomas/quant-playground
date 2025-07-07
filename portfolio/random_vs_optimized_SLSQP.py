import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def random_portfolio(
        tickers: list[str],
        start: str,
        end: str,
        n_portfolios: int
        ):
    """
    Generates a set of random portfolios and computes their annual return,
    volatility and Sharpe ratio.

    Parameters:
    - tickers: list of stock tickers
    - start: start date for historical data
    - end: end date for historical data
    - n_portfolios: number of random portfolios to generate

    Returns:
    - results: numpy array with shape (3, n_portfolios)
        - results[0]: annual returns
        - results[1]: volatilities
        - results[2]: Sharpe ratios
    """
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"] 
    returns = np.log(data / data.shift(1)).dropna() # Log returns are additive: log(p2/p1) + log(p3/p2) = log((p2/p1) + (p3/p2)) = log(p3/p1) ; simple returns are not
    mean_annual_returns = returns.mean()*252 # 252 = trading days in a year
    annual_cov_matrix = returns.cov() * 252

    results = np.zeros((3, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.random((len(tickers))) # random weights for each active
        weights /= np.sum(weights) # normalize so the sum of the weights equals to 1

        portfolio_return = np.dot(weights, mean_annual_returns) # = w_1 * r1 + w_2 * r2 + ... + w_n * rn
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
    
    return results


def optimize_portfolio_slsqp(
        tickers: list[str],
        start: str,
        end: str
        ):
    """
    Optimizes portfolio weights to maximize Sharpe ratio using SLSQP method.

    Parameters:
    - tickers: list of stock tickers
    - start: start date for historical data
    - end: end date for historical data

    Returns:
    - optimal_weights: numpy array of weights that maximize Sharpe ratio
    - optimal_return: expected annual return of the optimal portfolio
    - optimal_volatility: annualized volatility of the optimal portfolio
    - optimal_sharpe: Sharpe ratio of the optimal portfolio
    """

    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"] 
    returns = np.log(data / data.shift(1)).dropna() # Log returns are additive: log(p2/p1) + log(p3/p2) = log((p2/p1) + (p3/p2)) = log(p3/p1) ; simple returns are not
    mean_annual_returns = returns.mean()*252 # 252 = trading days in a year
    annual_cov_matrix = returns.cov() * 252

    def negative_sharpe(weights): # the function we want to minimize (that's why we define it negative)
        ret = np.dot(weights, mean_annual_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
        return -ret / vol
    
    initial_guess = np.array([1/len(tickers)]*len(tickers)) # just learned you can extend the length of a list n times by multiplying it by n 
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1} # the constraint we pass to the SQLSP (Sequential Least Squares Programming)
                                                                 # "eq" stands for equality in the function we pass for the constraint
    bounds = tuple((0, 1) for _ in range(len(tickers))) # each weight must be between 0 and 1

    result = minimize(
        fun = negative_sharpe,
        x0 = initial_guess,
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints
    )
    
    optimal_weights = result.x
    optimal_return = np.dot(optimal_weights, mean_annual_returns)
    optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(annual_cov_matrix, optimal_weights)))
    optimal_sharpe = optimal_return / optimal_volatility

    return optimal_weights, optimal_return, optimal_volatility, optimal_sharpe
