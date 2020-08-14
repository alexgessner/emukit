import numpy as np
from scipy.linalg import lapack
from emukit.core.acquisition import Acquisition


class MultiSourceSquaredCorrelation(Acquisition):
    """
    Acquisition for the maximizing correlation of the integral and the new points, independent of cost
    """

    def __init__(self, model):
        """
        :param model: multi-fidelity models (based on a GP)
        :param cost_functions: list of cost functions per fidelity level
        """
        self.model = model

    def has_gradients(self):
        return True

    def evaluate(self, location, levels=None):
        """
        Evaluate the acquisition at location 'location' and given fidelity levels 'levels'
        It is scaled by the current variance of the high fidelity integral to avoid very small values of acquisition.

        :param location: location where to evalute shape = (number of evaluations, input_dim-1)
        :param levels: list of fidelity levels, defaults to all
        :return: the acquisition function evaluated at x
        """
        x, levels = self._x_from_loclev(location, levels)
        return np.reshape(self._corr2(x=x), (-1, levels.size), order='F')

    def evaluate_with_gradients(self, location, levels=None):
        """
        Evaluate the acquisition function with gradient

        :param x: location and fidelity
        :return: the acquisition function and its gradient evaluated at x
        """
        x, levels = self._x_from_loclev(location, levels)

        grad = self._corr2_gradient(x).reshape(self.model.input_dim - 1, location.shape[0], len(levels))
        return self.evaluate(location=location, levels=levels), grad

    # helpers
    def _corr2(self, x):
        """
        Helper to compute the variance reduction of the integrated GP

        :param x: new candidate point x, contains the location and the fidelity to be evaluated
        :return: proportional to variance reduction of high fidelity integral for evaluating at x
        """
        int_var, pred_var, int_pred_cov = self._corr2_terms(x)
        return int_pred_cov**2 / (int_var * pred_var)

    def _corr2_terms(self, x):
        """
        Helper to compute the terms needed for the variance reduction of the integrated GP

        :param x: new candidate point x, contains the location and the fidelity to be evaluated
        :return: integral variance, predictive variance + noise, once integrated predictive covariance
        """
        int_var = self.model.integrate()[1][0,0]
        pred_var = (self.model.predict(x, full_cov=False)[1] + self.model.Gaussian_noise[0])
        int_pred_cov = self._high_fidelity_kernel_mean(x=x).T \
                       - np.dot(self._high_fidelity_kernel_mean(x=self.model.X), self._Kinv_Kx(x)).T

        return int_var, pred_var, int_pred_cov

    def _corr2_gradient(self, x):
        """
        Computes the gradient of the acquisition function

        :param x: location at which to evaluate the gradient
        :return: the gradient at x
        """
        int_var, pred_var, int_pred_cov = self._corr2_terms(x)
        dpredvar_dx, dintpredcov_dx = self._corr2_gradient_terms(x)

        return (2. * int_pred_cov * dintpredcov_dx  - 1./pred_var * dpredvar_dx * int_pred_cov**2) / (int_var * pred_var)

    def _corr2_gradient_terms(self, x):
        """
        Compute the terms needed for the gradient of the squared correlation (see _corr2_terms)

        :param x: location at which to evaluate the gradient, contains fidelity levels
        :return: the gradient of (pred_var, int_pred_cov) at x
        """
        dpredvar_dx = -2. * (self.model.kern.dK_dx(x, self.model.X) *
                             (self._Kinv_Kx(x).T)[None, ...]).sum(axis=2, keepdims=True)

        dintpredcov_dx = self.model.kern.dqK_dx(x, levels=[0]) \
                         - np.dot(self.model.kern.dK_dx(x, self.model.X), self._Kinv_Kq())

        return dpredvar_dx, dintpredcov_dx


    def _Kinv_Kx(self, x):
        """
        Inverse kernel Gram matrix multiplied with kernel at self.models.X and x
        .. math::
            K(X, X)^{-1} K (X, x)

        :param x: N locations at which to evaluate
        :return: K(X,X)^-1 K(X, x) with shape (self.models.X.shape[0], N)
        """
        L_chol = self.model.posterior.woodbury_chol
        return lapack.dtrtrs(L_chol.T, (lapack.dtrtrs(L_chol,
                                                      self.model.kern.K(self.model.X, x), lower=1)[0]), lower=0)[0]

    def _Kinv_Kq(self):
        """
        Inverse kernel Gram matrix multiplied with kernel mean at self.models.X and high fidelity
        .. math::
            K(X, X)^{-1} \int K (X, x) dx

        :param x: N locations at which to evaluate
        :return: K(X,X)^-1 K(X, x) with shape (self.models.X.shape[0], N)
        """
        L_chol = self.model.posterior.woodbury_chol
        qK = self._high_fidelity_kernel_mean(x=self.model.X)
        return lapack.dtrtrs(L_chol.T, (lapack.dtrtrs(L_chol, qK.T, lower=1)[0]), lower=0)[0]

    def _high_fidelity_kernel_mean(self, x):
        """ computes the kernel mean at x only for highest fidelity """
        return self.model.kern.qK(x, levels=[0])

    def _x_from_loclev(self, location, levels):
        """
        augments location vector by levels
        :param location: location where to evaluate; shape = (number of evaluations, input_dim-1)
        :param levels: list of fidelity levels, defaults to all

        :return: new location array to with levels at the end, levels (might be modified)
        """
        if levels is None:
            levels = [i for i in range(self.model.kern.parts[1].output_dim)]

        levels = np.asarray(levels)
        assert location.shape[1] == self.model.input_dim - 1

        x = np.concatenate((np.tile(location, (levels.size, 1)), np.repeat(levels, location.shape[0])[:, None]), axis=1)

        return x, levels