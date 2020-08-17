# emukit imports
from emukit.core.acquisition import Acquisition
from emukit.core.loop.outer_loop import OuterLoop
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop.model_updaters import FixedIntervalUpdater

# multi-source specific objects
from ..integral_bounds import IntegralBounds
from ..multi_source_bq import MultiSourceBayesianQuadrature
from ..acquisition import MultiSourceIntegratedVarianceReductionPerCost, CostFunctionsBase
from .candidate_point_calculators import MultiSourceSequentialPointSelector


class MultiSourceQuadratureLoop(OuterLoop):
        def __init__(self, integral_bounds: IntegralBounds, n_levels: int, model: MultiSourceBayesianQuadrature,
                     cost_functions: CostFunctionsBase, acquisition: Acquisition=None, update_interval: int=1):
            """
            An outer loop class for use with multi-source Bayesian quadrature

            :param integral_bounds: Definition of domain bounds to collect points within
            :param n_levels: number of fidelity levels
            :param model: The model that approximates the underlying function
            :param acquisition: acquisition function object
            :param update_interval: How many iterations between optimizing the model
            """

            if acquisition is None:
                acquisition = MultiSourceIntegratedVarianceReductionPerCost(model, cost_functions)
            else:
                acquisition = acquisition

            candidate_point_calculator = MultiSourceSequentialPointSelector(acquisition=acquisition,
                                                                            space=integral_bounds, n_levels=n_levels)
            model_updater = FixedIntervalUpdater(model, update_interval)
            loop_state = create_loop_state(model.X, model.Y)
            super(MultiSourceQuadratureLoop).__init__(candidate_point_calculator, model_updater, loop_state)

        def advance_loop_by_one(self, user_functions: MultiSourceFunctionWrapper):
            """ Run one step of the loop only """
            new_x = self.candidate_point_calculator.compute_next_points(self.loop_state)
            results = user_functions.evaluate(new_x)
            self.loop_state.update(results)
            self.model_updater.update(self.loop_state)