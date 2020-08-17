
import numpy as np
from emukit.core.loop.candidate_point_calculators import SequentialPointCalculator

from ..optimization import MultiSourceAcquisitionOptimizer


# this class could inherit from emukit.core.loop.candidate_point_calculators.Sequential
class MultiSourceSequentialPointSelector(SequentialPointCalculator):
    """
    This candidate point calculator optimizes the acquisition function separately for each fidelity,
    then chooses the fidelity with the highest acquisition value
    """

    def __init__(self, acquisition, space, n_levels, acquisition_optimizer=None):
        """
        Sequestially suggest a new candidate point and fidelity level to be evaluated

        :param acquisition: Acquisition function to find maximum of
        :param space: Domain to search for maximum over, IntegralBounds object
        :param n_levels: Number of sources
        :param acquisition_optimizer: Optimizer of the acquisition function, if None, use default
        """
        if acquisition_optimizer is None:
            self.acquisition_optimizer = MultiSourceAcquisitionOptimizer(self.space)
        else:
            self.acquisition_optimizer = acquisition_optimizer

        super(MultiSourceSequentialPointSelector, self).__init__(acquisition=acquisition,
                                                                 acquisition_optimizer=acquisition_optimizer)
        self.space = space
        self.n_levels = n_levels

    def compute_next_points(self, loop_state=None):
        """
        TODO: Signature match! No `context` variable here.
        TODO: use UserFunction interface instead of (flawed) target function
        :param loop_state: Unused
        :return: An np.ndarray with the next input location to query
        """
        f_mins = np.zeros(self.n_levels)
        x_opts = []

        # Optimize acquisition for each information source
        for i in range(self.n_levels):
            x, f_mins[i] = self.acquisition_optimizer.optimize(self.acquisition, level=i)
            x_opts.append(x)
        best_fidelity = np.argmin(f_mins)
        return np.append(np.atleast_2d(x_opts[best_fidelity]), np.asarray([[best_fidelity]]), axis=1)
