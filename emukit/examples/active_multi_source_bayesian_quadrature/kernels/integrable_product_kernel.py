# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from GPy.kern.src.prod import Prod


class IntegrableTensorProductKernel(Prod):
    """
    Tensor product kernel between an integrable kernel in x-space and a coregionalization kernel in level-space

    :param kernels: (k1, k2) tuple of kernels;
                    k1 is the one in x-space and needs to be integrable
                    k2 is in "level"-space, and has input_dim=1
    :param integral_bounds: (lower_bounds, upper_bounds) tuple of integral lower and upper bounds.
                            Both need to have the size (k1.input_dim, 1) and will be passed to the integrable kernel

    This class inherits from parent class GPy.kern.Prod

    Note that this subclass is more restrictive than its parent in that it cannot accommodate the product of several
    kernels in x-space, since the integral would change.

    TODO: The methods contained in this class could be added to GPy.kern.Prod. These methods only work if the first kernel provided with the tensor product is integrable.
    """

    def __init__(self, kernels): # TODO: two inputs

        # code rewritten from Kern.__pow__ to ensure tensor product
        assert np.all(kernels[0]._all_dims_active == range(kernels[0].input_dim)), \
            "Can only use kernels, which have their input_dims defined from 0"
        assert np.all(kernels[1]._all_dims_active == range(kernels[1].input_dim)), \
            "Can only use kernels, which have their input_dims defined from 0"
        kernels[1]._all_dims_active += kernels[0].input_dim

        assert len(kernels) == 2
        assert kernels[1].input_dim == 1

        super(IntegrableTensorProductKernel, self).__init__(kernels, name='integrable_product_kernel')


    def qK(self, X, levels=None):
        """
        Kernel mean of tensor product kernel is the spatial kernel mean multiplied with the kernel value in level space

        :param X: Points at which to evaluate the kernel mean
        :param levels: levels (indices) indicating the outputs we want to compute the kernel mean for. Default: all levels
        :type levels: list of fidelity levels to be evaluated

        :return: the kernel mean evaluated at X
        """
        if levels is None:
            levels = self._default_levels()

        # dummy array for passing levels to coregionalization kernel
        dummy_X = self._dummy_X(levels=levels)
        # TODO: how does GPy manage to pass the entire X to the kernel routine?
        return self.parts[0].qK(X[:, :-1]) * self.parts[1].K(dummy_X, X)


    def Kq(self, X, levels=None):
        """
        Transpose of qK
        """
        return self.qK(X, levels=levels).T


    def qKq(self, levels=None):
        """
        Tensor product kernel integrated over both spatial dimensions (aka. initial integral error)

        :param levels: list of levels (indices) indicating the outputs we want to compute the kernel mean for. Default: all levels

        :return: the initial integral error
        """
        if levels is None:
            levels = self._default_levels()

        # dummy array for passing levels to coregionalization kernel
        dummy_X = self._dummy_X(levels=levels)
        return (np.multiply(self.parts[0].qKq(), self.parts[1].Kdiag(dummy_X))).reshape(-1, 1)


    def dK_dx(self, X, X2):
        """
        gradient of the kernel wrt x

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return self.parts[0].dK_dx(X[:,:-1], X2[:,:-1]) * self.parts[1].K(X, X2)[None, :, :]


    def dqK_dx(self, X, levels=None):
        """
        gradient of the kernel mean evaluated at x
        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)

        :return: the gradient with shape (input_dim, N, num_levels)
        """
        if levels is None:
            levels = self._default_levels()

        # dummy array for passing levels to coregionalization kernel
        dummy_X = self._dummy_X(levels=levels)

        return self.parts[0].dqK_dx(X[:,:-1])[:, :, None] * (self.parts[1].K(dummy_X, X).T)[None,:,:]


    def dKq_dx(self, x, levels=None):
        """
        gradient of the transposed kernel mean evaluated at x
        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)

        :return: the gradient with shape (N, input_dim, num_levels)
        """
        return np.swap_axis(self.dqK_dx(x, levels=levels), 0, 1)

    # helpers
    def _dummy_X(self, levels):
        """
        :param levels: list of levels (indices). Default: all levels

        :return: some dummy X to pass to the product kernel
        """
        return np.hstack((np.zeros(shape=(len(levels), self.parts[0].input_dim)), np.asarray(levels).reshape(-1, 1)))


    def _default_levels(self):
        return [i for i in range(self.parts[1].output_dim)]

