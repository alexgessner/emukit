# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple

from ...core.acquisition import Acquisition
from ...quadrature.methods import VanillaBayesianQuadrature


class ModelVariance(Acquisition):
    """
    This acquisition selects the point in the domain where the predictive variance is the highest
    .. math::
        PKM(x) =  k(x,x) - k(x,X) K_{XX}^{-1} k(X, x)
    """

    def __init__(self, model: VanillaBayesianQuadrature):
        """
        :param model: The vanilla Bayesian quadrature model
        """
        self.model = model

    def has_gradients(self) -> bool:
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the acquisition function at x.

        :param x: (n_points, input_dim) locations where to evaluate
        :return: (n_points, 1) the acquisition function value at x
        """
        _, variance = self.model.predict(x)
        return variance

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the acquisition function with gradient

        :param x: (n_points, input_dim) locations where to evaluate
        :return: acquisition value and corresponding gradient at x, shapes (n_points, 1) and (n_points, input_dim)
        """
        # value
        variance = self.evaluate(x)

        # gradient #TODO: Should go into model, which should inherit from "IDifferentiable"
        dvariance_dx = - 2 * np.dot(self.model.base_gp.kern.dK_dx1(x, self.model.base_gp.X), self._graminv_k(x))
        return variance, dvariance_dx

    def _graminv_k(self, x: np.ndarray):
        """
        Inverse kernel mean multiplied with inverse kernel Gram matrix, all evaluated at training locations.
        :param x: (n_points, input_dim) locations where to evaluate

        .. math::
             [k(X, X) + \sigma^2 I]^{-1} k(X, x)

        :return: weights of shape (1, n_train_points)
        """
        lower_chol = self.model.base_gp.gram_chol()
        k = self.model.base_gp.kern.K(self.model.base_gp.X, x)
        graminv_k = lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, k, lower=1)[0]), lower=0)[0]
        return graminv_k