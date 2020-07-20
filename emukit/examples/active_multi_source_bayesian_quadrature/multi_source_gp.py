# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from GPy.core.gp import GP
from GPy.likelihoods import Gaussian
from scipy.linalg import lapack


class MultiSourceGaussianProcess(GP):
    """
    A multi-output GP

    Takes a separable kernel k(l,l',x,x') that models correlation between different "levels of fidelity" or simply
    different sources.

    The last axis contains the fidelity level associated with the observation
    """

    def __init__(self, X, Y, kernel, Y_metadata=None, normalizer=None, noise_var=1, mean_function=None):

        # for GP regression
        likelihood = Gaussian(variance=noise_var)
        super(MultiSourceGaussianProcess, self).__init__(X, Y, kernel, likelihood, name="MS-GP", Y_metadata=Y_metadata,
                                                         normalizer=normalizer, mean_function=mean_function)

    def update_data(self, X, Y):
        """ add new data to the model """
        self.set_XY(X, Y)