# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...quadrature.methods import VanillaBayesianQuadrature

class VanillaMultiSourceBayesianQuadrature(VanillaBayesianQuadrature):
    """
    Vanilla multi-source Bayesian quadrature method


    """

    def __init__(self, ms_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray):
        """
        :param ms_gp: a multi-source GP
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        """
        super(VanillaMultiSourceBayesianQuadrature, self).__init__(base_gp=ms_gp, X=X, Y=Y)

    def integrate(self, levels=None):
        """
        Computes the estimator for the integral and a variance
        """
        return self._compute_integral_mean_and_variance(levels=levels)

    # helpers
    def _compute_integral_mean(self, levels=None):
        """
        Computes the mean of the integral at given multi-fidelity levels
        :param levels: levels (indices) indicating the outputs we want to compute the kernel mean for. Default: all levels
        :type levels: np.ndarray of shape (number of levels, 1) or (number of levels,)

        :return: mean of the integral (np.ndarray)
        """
        return np.dot(self.kern.qK(self.X, levels=levels), self.posterior.woodbury_vector).squeeze()

    def _compute_integral_variance(self, levels=None):
        """
        Computes the variance of the integral at given multi-fidelity levels
        :param levels: levels (indices) indicating the outputs we want to compute the kernel mean for. Default: all levels
        :type levels: np.ndarray of shape (number of levels, 1) or (number of levels,)

        :return: variance of the integral (np.ndarray)
        """
        return self.kern.qKq(levels=levels) - \
               np.square(lapack.dtrtrs(self.posterior.woodbury_chol,
                                       self.kern.Kq(X=self.X, levels=levels), lower=1)[0]).sum(axis=0, keepdims=True).T

    def _compute_integral_mean_and_variance(self, levels=None):
        """
        Compute both integral mean and variance
        :return: integral mean and variance
        """
        return self._compute_integral_mean(levels=levels), self._compute_integral_variance(levels=levels)