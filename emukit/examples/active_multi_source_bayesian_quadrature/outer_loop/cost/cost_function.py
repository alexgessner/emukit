import abc
import numpy as np


class CostFunctionsBase(abc.ABC):
    """ Cost functions for multi-fidelity """
    def __init__(self, nlevels):
        """
        :param nlevels: total number of fidelity levels
        """
        self.nlevels = nlevels

    @abc.abstractmethod
    def evaluate(self, location, levels):
        """
        Evaluate the cost functions at locations x
        :param location: locations at which to evaluate
        :param levels: List of fidelity levels where to evaluate. Defaults to all levels
        :return: np.ndarray of cost function evaluated at levels.
        """
        pass

    def has_gradients(self) -> bool:
        pass

    def evaluate_gradient(self, location, levels):
        """
        Evaluate the jacobian of the cost functions at locations x
        :param location: locations at which to evaluate
        :param levels: List of fidelity levels where to evaluate. Defaults to all levels
        :return: np.ndarray of cost function gradients evaluated at levels.
        """
        pass


class ConstantCostFunctions(CostFunctionsBase):
    """ Cost functions that do not depend on the location, only on the fidelity level """
    def __init__(self, nlevels, cost_per_level):
        """
        :param nlevels: number of fidelities
        :param cost_per_level: list of costs per fidelity in descending order of cost
        """
        self.cost_per_level = cost_per_level
        super(ConstantCostFunctions, self).__init__(nlevels=nlevels)

    def evaluate(self, location, levels):
        """ Evaluate the cost function at location x and fidelity levels
        default is same cost for every level
        :param location: (dummy) location where to evaluate
        :param levels: list of fidelity levels evaluated

        :returns: cost vector (independent of x)
        """
        return (np.asarray(self.cost_per_level)[levels])[None,:]

    def has_gradients(self):
        return True

    def evaluate_gradient(self, location, levels):
        """
        Evaluate the jacobian of the cost functions at locations x
        :param location: (dummy) location where to evaluate
        :param levels: List of fidelity levels where to evaluate
        :return: np.ndarray of cost function gradients evaluated at levels.
        """
        return np.zeros(shape=(location.shape[1], 1, len(levels)))


class CostFunctions(CostFunctionsBase):
    """ Cost functions that depend on the location and the source """
    def __init__(self, nlevels, cost_functions, cost_gradients=None):
        """
        :param nlevels: number of fidelities
        :param cost_functions: list of function handles to cost, len(cost_functions) = nlevels,
                               the shape of the return values of the function values should be (number of points, 1)
        :param cost_gradients: list of function handles to the gradients of the cost functions
                               the shape of the return values of the function values should be (dim, number of points)
        """
        super(CostFunctions, self).__init__(nlevels=nlevels)
        self.cost_functions = cost_functions
        self.cost_gradients = cost_gradients

    def evaluate(self, location, levels):
        """ Evaluate the cost function at location x and fidelity levels
        default is same cost for every level
        :param location: location where to evaluate
        :param levels: list of source levels evaluated

        :returns: cost vector
        """
        costs = np.zeros(shape=(location.shape[0], len(levels)))
        for i, l in enumerate(levels):
            costs[:,i] = self.cost_functions[l](location)[:,0]
        return costs

    def has_gradients(self):
        if cost_gradients is None:
            return False
        return True

    def evaluate_gradient(self, location, levels):
        """
        Evaluate the jacobian of the cost functions at locations x
        :param location: location where to evaluate
        :param levels: List of source levels where to evaluate
        :return: np.ndarray of cost function gradients evaluated at levels.
        """
        gradients = np.zeros(shape=(location.shape[1], location.shape[0], len(levels)))
        for i, l in enumerate(levels):
            gradients[:,:,i] = self.cost_gradients[l](location)
        return gradients
