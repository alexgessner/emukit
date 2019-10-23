# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple

from ...core.acquisition import Acquisition
from ...quadrature.methods import VanillaBayesianQuadrature


class PosteriorKernelMean(Acquisition):
    """
    This acquisition function is the posterior kernel mean under a GP-model, i.e. the once integrated posterior mean.
    Applies to vanilla-BQ only.

    .. math::
        PKM(x) = \int k(x,x') - k(x,X) K_{XX}^{-1} k(X, x') dx'
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
        k_mean_x = self.model.base_gp.kern.Kq(x)
        return k_mean_x - np.dot(self.model.base_gp.kern.K(x, self.model.base_gp.X), self._graminv_Kq())

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the acquisition function with gradient

        :param x: (n_points, input_dim) locations where to evaluate
        :return: acquisition value and corresponding gradient at x, shapes (n_points, 1) and (n_points, input_dim)
        """
        # value
        posterior_kmean = self.evaluate(x)

        # gradient
        posterior_kmean_gradient = self.model.base_gp.kern.dKq_dx(x) - \
                                   np.dot(self.model.base_gp.kern.dK_dx1(x, self.model.base_gp.X), self._graminv_Kq())
        return posterior_kmean, posterior_kmean_gradient


    def _graminv_Kq(self):
        """
        Inverse kernel mean multiplied with inverse kernel Gram matrix, all evaluated at training locations.

        .. math::
            \int k(x, X)\mathrm{d}x [k(X, X) + \sigma^2 I]^{-1}

        :return: weights of shape (1, n_train_points)
        """
        lower_chol = self.model.base_gp.gram_chol()
        Kq = self.model.base_gp.kern.Kq(self.model.base_gp.X)
        graminv_Kq = lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, Kq, lower=1)[0]), lower=0)[0]
        return graminv_Kq