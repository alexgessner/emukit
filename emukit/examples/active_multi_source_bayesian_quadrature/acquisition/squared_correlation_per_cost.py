import numpy as np
from .squared_correlation import MultiSourceSquaredCorrelation


class MultiSourceSquaredCorrelationPerCost(MultiSourceSquaredCorrelation):
    """
    Acquisition for the reduction of integrated variance of the highest fidelity per cost of the to-be-evaluated
    fidelity level, for kernels where the integral is available in closed form.
    (equivalent to maximizing correlation)
    Note that this class is entirely identical to IntegratedVarianceReduction. They are both kept for sanity checks.
    """

    def __init__(self, model, cost_functions):
        """
        :param model: multi-fidelity models (based on a GP)
        :param cost_functions: list of cost functions per fidelity level
        """
        super(MultiSourceSquaredCorrelationPerCost, self).__init__(model)
        self.cost_functions = cost_functions

    def evaluate(self, location, levels=None):
        """
        Evaluate the acquisition at location 'location' and given fidelity levels 'levels'
        It is scaled by the current variance of the high fidelity integral to avoid very small values of acquisition.

        :param location: location where to evalute shape = (number of evaluations, input_dim-1)
        :param levels: list of fidelity levels, defaults to all
        :return: the acquisition function evaluated at x
        """
        x, levels = self._x_from_loclev(location, levels)
        return np.reshape(self._corr2(x=x), (-1, levels.size), order='F') / \
               self.cost_functions.evaluate(location=location, levels=levels)

    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, location, levels=None):
        """
        Evaluate the acquisition function with gradient

        :param x: location and fidelity
        :return: the acquisition function and its gradient evaluated at x
        """
        grad = self._compute_gradient_w_cost(location, levels)
        return self.evaluate(location=location, levels=levels), grad

        # helpers
    def _compute_gradient_w_cost(self, location, levels):
        """ Compute the gradient including the gradient of the cost """
        x, levels = self._x_from_loclev(location, levels)

        c = self.cost_functions.evaluate(location=location, levels=levels)
        dc = self.cost_functions.evaluate_gradient(location=location, levels=levels)

        rho2 = self._corr2(x)
        drho2 = self._corr2_gradient(x).reshape(self.model.input_dim - 1, location.shape[0], len(levels))

        grad = (c * drho2 - dc * rho2) / c**2

        return grad
