# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class IntegralBounds():
    def __init__(self, lower_bounds, upper_bounds, input_dim):
        """
        Defines the parameter space by specifying the integration bounds
        :param lower_bounds: Lower bounds of the integral
        :type lower_bounds: np.ndarray with shape (1, input_dim) or float if integration is over a cube
        :param upper_bounds: Upper bounds of the integral
        :type upper_bounds: np.ndarray with shape (1, input_dim) or float if integration is over a cube
        :param input_dim: input dimension
        """

        if isinstance(lower_bounds, float) and isinstance(lower_bounds, float):
            lower_bounds = lower_bounds * np.ones(shape=(1, input_dim))
            upper_bounds = upper_bounds * np.ones(shape=(1, input_dim))

        elif isinstance(lower_bounds, np.ndarray) and isinstance(upper_bounds, np.ndarray):
            assert lower_bounds.shape == (1, input_dim) and upper_bounds.shape == (1, input_dim)
            assert np.all(upper_bounds - lower_bounds >= 0.)

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def check_in_domain(self, x):
        """
        Checks if the points in x lie between the min and max allowed values
        :param x: locations (n_points, input_dim)
        :return: An array (n_points, input_dim) which contains a boolean indicating whether each point is in domain
        """
        return np.all([(self.lower_bounds < x), (self.upper_bounds > x)], axis=0)

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds
