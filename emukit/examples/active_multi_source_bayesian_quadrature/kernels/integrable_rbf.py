
import numpy as np
from scipy.special import erf
from GPy.kern.src.rbf import RBF

class IntegrableRBF(RBF):
    """
    RBF kernel with integrals

    :param input_dim: Integer specifying the input dimension the kernel works on
    :param integral_bounds: (lower_bounds, upper_bounds) tuple of integral lower and upper bounds.
                            Both need to have the size (1, input_dim)
    For other inputs check GPy RBF.
    """
    def __init__(self, input_dim, integral_bounds, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf_with_integral', useGPU=False):
        super(IntegrableRBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)

        # integral bounds
        self.lower_bounds, self.upper_bounds = integral_bounds.get_bounds()

        # checks for the integral bounds (integration only happens in x-space)
        assert self.lower_bounds.shape == (1, input_dim) and self.upper_bounds.shape == (1, input_dim)
        assert np.all(self.upper_bounds - self.lower_bounds >= 0.)


    def qK(self, x):
        """
        RBF kernel mean, i.e. the kernel integrated over the first argument as a function of its second argument

        :param x: N points at which kernel mean is evaluated, np.ndarray with x.shape = (N, input_dim)

        :returns: the kernel mean evaluated at N input points x, np.ndarray with shape (1, N)
        """
        erf_lo = erf(self._scaled_vectordiff(self.lower_bounds, x))
        erf_up = erf(self._scaled_vectordiff(self.upper_bounds, x))
        kernel_mean = self.variance * (self.lengthscale * np.sqrt(np.pi/2.) * (erf_up - erf_lo)).prod(axis=1)

        return kernel_mean.reshape(1, -1)


    def Kq(self, x):
        """
        RBF transposed kernel mean,
        i.e. the kernel integrated over the second argument as a function of its first argument

        :param x: N points at which kernel mean is evaluated, np.ndarray with x.shape = (N, input_dim)

        :returns: the kernel mean evaluated at N input points x, np.ndarray with shape (N,1)
        """
        return self.qK(x).T


    def qKq(self):
        """
        RBF kernel integrated over both parameters

        :returns: scalar value for the double integral of the kernel
        """

        prefac = self.variance * (2.* self.lengthscale**2)**self.input_dim
        diff_bounds_scaled = self._scaled_vectordiff(self.upper_bounds, self.lower_bounds)
        exp_term = np.exp(-diff_bounds_scaled**2) - 1.
        erf_term = erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(np.pi)

        return prefac * (exp_term + erf_term).prod()


    def dK_dx(self, x, x2):
        """
        gradient of the kernel wrt x

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return (self.dK_dr_via_X(x, x2)[None, ...] * self._dr_dx(x, x2))


    def dqK_dx(self, x):
        """
        gradient of the kernel mean evaluated at x
        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)

        :return: the gradient with shape (input_dim, N)
        """

        exp_lo = np.exp(- self._scaled_vectordiff(x, self.lower_bounds) ** 2)
        exp_up = np.exp(- self._scaled_vectordiff(x, self.upper_bounds) ** 2)
        erf_lo = erf(self._scaled_vectordiff(self.lower_bounds, x))
        erf_up = erf(self._scaled_vectordiff(self.upper_bounds, x))

        fraction = ((exp_lo - exp_up)/(self.lengthscale * np.sqrt(np.pi/2.) * (erf_up - erf_lo))).T

        return self.qK(x) * fraction


    def dKq_dx(self, x):
        """
        gradient of the transposed kernel mean evaluated at x
        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)

        :return: the gradient with shape (N, input_dim)
        """
        return self.dqK_dx(x).T


    # helpers
    def _scaled_vectordiff(self, v1, v2):
        """
        Scaled element-wise vector difference between vectors v1 and v2

        .. math::
            \frac{v_1 - v_2}{\lambda \sqrt{2}}

        name mapping:
            \lambda: self.lengthscale

        :param v1: first vector
        :param v2: second vector
        :return: scaled difference between v1 and v2
        """
        return (v1 - v2) / (self.lengthscale * np.sqrt(2))

    def _dr_dx(self, x, x2):
        """
        Derivative of the radius

        .. math::
            r = \sqrt{ \frac{||x - x_2||^2}{\lambda^2} }

        name mapping:
            \lambda: self.lengthscale

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return (x[None, :, :] - x2[:, None, :]).T / (self.lengthscale ** 2 * self._scaled_dist(x, x2)[None,...])