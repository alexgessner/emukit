# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from GPy.core.gp import GP
from GPy.likelihoods import Gaussian
from scipy.linalg import lapack


class MultiSourceBayesianQuadrature(GP):
    """
    Adds Vanilla Bayesian Quadrature to GPy GP

    Takes a separable kernel k(l,l',x,x') that models correlation between different levels of fidelity,
    allows to do GP regression on observations and integrates the kernel in x-space to return the mean and the variance
    of the highest fidelity function.

    The last axis contains the fidelity level associated with the observation
    """

    def __init__(self, X, Y, kernel, noise_var=1, **kwargs):
        """
        Quadrature model for a multi-source model
        :param X: Input of training data, shape (n_eval, dim+1), last dimension identifies the source
        :param Y: Output of training data, shape (n_eval, 1)
        :param kernel: instance of :class:IntegrableTensorProductKernel
        :param noise_var: Noise variance (assumed same across sources)
        """

        # for GP regression
        likelihood = Gaussian(variance=noise_var)
        super(MultiSourceBayesianQuadrature, self).__init__(X, Y, kernel, likelihood, name="Vanilla AMS-BQ", **kwargs)

    def integrate(self, levels=None):
        """
        Computes the estimator for the integral and a variance
        """
        return self._compute_integral_mean_and_variance(levels=levels)

    def update_data(self, X, Y):
        """ Wrapper for emukit style model update in terms of GPy model update """
        self.set_XY(X, Y)

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