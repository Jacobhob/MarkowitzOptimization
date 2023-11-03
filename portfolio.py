"""
portfolio.py
Author: Jacob Lu
Date: 2022.8.28
"""

import pandas as pd


class Portfolio():
    """ Handle portfolio statistics. """
    def __init__(self, init_capital: float, universe: list) -> None:
        self._capital = init_capital  # Portfolio initial capital
        universe.sort()
        self._universe = universe  # Portfolio stocks universe
        self._stkvalue = pd.Series()  # Portfolio stocks market value

    def construct(self, weight: pd.Series) -> None:
        """ Construct portfolio from stocks weight series. """
        stock_value = (weight * self._capital).sort_index()
        # Optimizater has certain flexibility to return a negative weight, but small engough to ignore
        stock_value = stock_value.mask(stock_value.abs() < 1e-2, 0)
        # Round all value to integer
        stock_value = stock_value.round(decimals=0)
        # Store as stock value
        self._stkvalue = stock_value

    @property
    def universe(self) -> list:
        return self._universe

    @property
    def wgt(self) -> pd.Series:
        return self._stkvalue / self._stkvalue.sum()

    @property
    def val(self) -> pd.Series:
        return self._stkvalue
