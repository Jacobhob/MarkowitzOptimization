"""
optimizer.py
Author: Jacob Lu
Date: 2022.8.28
"""

import itertools
import traceback

import cvxpy as cp


class Optimizer():
    """ Optimizer wrapper. """
    def __init__(self) -> None:
        self._prob = None
        self._variable = {}
        self._constraints = {}
        self._objective = None

    def add_variable(self, name, var) -> None:
        """ Store optimization variable. """
        self._variable[name] = var

    def add_constraint(self, name, cons) -> None:
        """ Add optimization constriants. """
        self._constraints[name] = cons

    def create_objective(self, func, maximize=True):
        """ Construct objective function for the optimization problem. """
        self._objective = cp.Maximize(func) if maximize else cp.Minimize(func)

    def solve(self, **kwargs) -> None:
        """ Solver wrapper. """
        max_iter = kwargs.get('max_iter', 1000000)
        check_termination = kwargs.get('check_termination', 500)
        eps_abs = kwargs.get('eps_abs', 1e-5)
        eps_rel = kwargs.get('eps_rel', 1e-5)
        rho = kwargs.get('rho', 1e-6)
        sigma = kwargs.get('sigma', 1e-12)
        alpha = kwargs.get('alpha', 1.6)
        verbose = kwargs.get('verbose', False)

        solved = False
        count = 0
        while not solved and count < 3:
            # Construct the problem
            constraints = list(itertools.chain.from_iterable(self._constraints.values()))
            prob = cp.Problem(self._objective, constraints)
            print(f' - Solving problem for objective {self._objective}, with constraints {[*self._constraints.keys()]}.')
            try:
                prob.solve(solver='OSQP', linsys_solver='qdldl', max_iter=max_iter, check_termination=check_termination,
                    eps_abs=eps_abs, eps_rel=eps_rel, rho=rho, sigma=sigma, alpha=alpha, verbose=verbose, **kwargs)
                self._prob = prob
            except Exception:
                traceback.print_exc()
            else:
                if prob.status != cp.OPTIMAL:
                    count += 1
                else:
                    solved = True
                    print(' - Problem solved.')

    def output(self) -> dict:
        """ Post optimization results in dictionary. """
        if self._prob is None or self._prob.status != cp.OPTIMAL:
            return {}
        output_dict = {k: v.value for k, v in self._variable.items()}
        output_dict['objective'] = self._prob.value
        return output_dict
