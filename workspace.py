"""
workspace.py
Author: Jacob Lu
Date: 2022.8.28
"""

import cvxpy as cp
import numpy as np
import pandas as pd

import data_util as dutil
from optimizer import Optimizer
from portfolio import Portfolio


class MarkowitzOptimizationWorkspace():
    """ Workspace for Markoxitz optimization portfolio. """
    def __init__(self) -> None:
        self._optimizer = Optimizer()
        self._port = None

    def initialize_port(self) -> None:
        """ Construct initial portfolio. """
        csi300_components = dutil.get_csi300_components()
        self._port = Portfolio(init_capital=10000000, universe=csi300_components.index.tolist())
        self._port.construct(csi300_components.assign(Weight=1/300)['Weight'])
        print(' - Portfolio initialized.')

    def optimize(self):
        """ Solve Markowitz optimization problem. """
        # Import alpha and covariance matrix
        alpha = dutil.get_csi300_alpha().to_numpy()
        stk_cov = dutil.get_csi300_covariance().to_numpy()

        # -------------------------------------------------------------------------------------------------------------------
        # We are trying to construct a Markowitz optimized portfolio.
        # Using stock covariance to measure risk, our objective is simplified to maximize the sharpe ratio.
        #     Sharpe ratio = (E[portfolio return] - risk free rate) / std(portfolio return)
        # Define w as the weight vector of stocks, u as the expected return vector of stocks, rf as the risk free rate,
        # Z as the covariance matrix of stocks.
        #     Sharpe ratio = (u.T @ w - rf) / (w.T @ Z @ w)^0.5
        # Our objective is:
        #     Maximize (u.T @ w - rf) / (w.T @ Z @ w)^0.5
        #         s.t. sum(w) = 1,
        #              0 <= w <= 0.03,
        #              sum(abs(w - 1/300)) <= 0.15
        # Clearly this objective is not a standard convex optimization, we need further deduction.
        # Define
        #     f(w) = (u.T @ w - rf) / (w.T @ Z @ w)^0.5
        #          = (u.T @ w - rf * sum(w)) / (w.T @ Z @ w)^0.5
        # Let v be the vector of expected return minus risk free rate.
        #     f(w) = v.T @ w / (w.T @ Z @ w)^0.5
        # Consider vector x = a * w, where scalar a > 0.
        #     f(x) = f(a * w) = a * (v.T @ w) / (a * (w.T @ Z @ w)^0.5)
        #                     = v.T @ w / (w.T @ Z @ w)^0.5 = f(w)
        # If we let v.T @ x = 1 for some positive scalr a,
        #     f(w) = f(x) = v.T @ x / (x.T @ Z @ x)^0.5 = 1 / (x.T @ Z @ x)^0.5
        # Our objective is equavalent to:
        #     Minimize x.T @ Z @ x
        #         s.t. v.T @ x = 1,
        #              0 <= x <= 0.03a,
        #              sum(abs(x - a * 1/300)) <= 0.15a
        # , which is a standard quadratic program.
        # Suppose that x* is the optimal solution,
        #     w* = x* / a,
        #     sum(w*) = 1, sum(x*) = a * sum(w*) = a
        #     w* = x* / sum(x*)
        # Hence, w* is the optimal solution to our sharpe ratio.
        # -------------------------------------------------------------------------------------------------------------------

        # Define transformed stock weight as variable
        stk_wgt_transfrom = cp.Variable(shape=len(self._port.universe), name='stk_wgt_transfrom')
        self._optimizer.add_variable('stk_wgt_transfrom', stk_wgt_transfrom)
        end_stkweight_sum = cp.sum(stk_wgt_transfrom)
        cov_matrix = cp.Parameter(shape=stk_cov.shape, name='cov', value=stk_cov, PSD=True)
        sharpe_transfrom = cp.quad_form(stk_wgt_transfrom, cov_matrix)

        # Add constriants
        self._optimizer.add_constraint('validate_weight', [
            stk_wgt_transfrom >= 0,
            stk_wgt_transfrom <= 0.03 * end_stkweight_sum,
        ])
        self._optimizer.add_constraint('weight_limit', [
            cp.sum(cp.hstack(stk_wgt_transfrom) @ (alpha - dutil.RISK_FREE)) == 1,
        ])
        self._optimizer.add_constraint('turnover', [
            cp.sum(cp.abs(stk_wgt_transfrom - self._port.wgt.to_numpy() * end_stkweight_sum)) <= 0.15 * end_stkweight_sum,
        ])
        # Add objective
        self._optimizer.create_objective(sharpe_transfrom, maximize=False)
        # Solve problem
        self._optimizer.solve()

    def backcasting(self):
        """ Backcasting optimal solution to original stock weight """
        result = self._optimizer.output()
        stk_wgt_transfrom = result['stk_wgt_transfrom']
        stk_wgt = stk_wgt_transfrom / np.sum(stk_wgt_transfrom)
        # Update portfolio weight
        self._port.construct(pd.Series(stk_wgt, index=self._port.universe))
        print(f' - Portfolio weight: {self._port.val}')
        # Calculate sharpe ratio
        sharpe_ratio = 1 / result['objective']**0.5
        print(f' - Sharpe ratio: {sharpe_ratio:.4f}')

    def write_port(self):
        """ write to csv """
        self._port.val.to_csv('optimized_port.csv')
