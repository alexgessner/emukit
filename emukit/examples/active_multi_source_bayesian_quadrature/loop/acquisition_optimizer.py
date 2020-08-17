import numpy as np
from scipy.optimize import minimize, Bounds

from emukit.core.optimization import AcquisitionOptimizerBase

from emukit.examples.active_multi_source_bayesian_quadrature.integral_bounds import IntegralBounds


# this class could inherit from emukit.core.optimization.AcquisitionOptimizer
class MultiSourceAcquisitionOptimizer:
    """
    Class that handles the optimization of an acquisition function for the multi-source setting.
    Do not confound with the MultiSourceAcquisitionOptimizer in emukit.core.optimization.
    This one is merely a modified version that takes integral bounds as defined in this particular example
    instead of the parameter space defined through emukit.
    """

    def __init__(self, space: IntegralBounds) -> None:
        """
        class that handles the optimization of an acquisition function (GPyOpt independent)

        :param space: domain specifier
        :type space: IntegralBounds
        """
        lb, ub = space.get_bounds()
        self.bounds = Bounds(lb[0,:], ub[0,:])

    def optimize(self, acquisition, level, method='L-BFGS-B', n_restarts=10):
        """
        optimize the acquisition function

        :param acquisition: Acquisition function
        :param level: source to evaluate
        :param method: which scipy optimizer to use
        :param n_restarts: best result of n_restarts is accepted
        :return: a tuple containing the location of the minimum and the value of the acquisition at that point
        """

        def f(x):
            return -acquisition.evaluate(x[None, :], levels=[level])

        def f_df(x):
            f_value, df_value = acquisition.evaluate_with_gradients(x[None, :], levels=[level])
            return -f_value, -df_value

        if acquisition.has_gradients:
            return self._optimize_with_restarts(f_df, method=method, jac=True, n_restarts=n_restarts)
        else:
            return self._optimize_with_restarts(f, method=method, jac=None, n_restarts=n_restarts)


    def _optimize_with_restarts(self, f, method, jac, n_restarts):
        """
        Helper to restart the optimization n_restart times from random initialisations. Of n_restart optimization runs,
        it returns the result with the lowest value of the negative acquisition function
        :param f: function handle for the function to be optimized.
        :param method: optimization method, see scipy.opimize.minimize
        :param jac: boolean, if False, f returns a function value, if True a tuple of (function value, gradient)
        :param n_restarts: number of optimization runs the best one of which is returned
        :return: Tuple of the location of a minimum and the value of acquisition at that point
        """

        best_result = None

        for i in range(n_restarts):

            # draw a random starting point from domain
            x0 = np.random.uniform(self.bounds.lb, self.bounds.ub)

            result = minimize(f, x0, method=method, bounds=self.bounds, jac=jac)

            if best_result is None:
                best_result = result
            elif result.fun < best_result.fun:
                best_result = result

        return best_result.x, best_result.fun