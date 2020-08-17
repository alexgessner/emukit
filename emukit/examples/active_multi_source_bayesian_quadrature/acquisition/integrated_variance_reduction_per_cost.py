import numpy as np
from scipy.linalg import lapack
from emukit.core.acquisition import Acquisition


class MultiSourceIntegratedVarianceReductionPerCost(Acquisition):
    """
    Acquisition for the reduction of integrated variance of the highest fidelity per cost of the to-be-evaluated
    fidelity level, for kernels where the integral is available in closed form.

    NOTE: Inherited directly from `Acquisition` because we are NOT dealing with a VanillaBayesianQuadrature model.
    """

    def __init__(self, model, cost_functions):
        """
        :param model: multi-fidelity models (based on a GP)
        :param cost_functions: list of cost functions per fidelity level
        """
        self.model = model
        self.cost_functions = cost_functions

    def evaluate(self, location, levels=None):
        """
        Evaluate the acquisition at location 'location' and given fidelity levels 'levels'
        It is scaled by the current variance of the high fidelity integral to avoid very small values of acquisition.

        :param location: location where to evalute shape = (number of evaluations, input_dim-1)
        :param levels: list of fidelity levels, defaults to all
        :return: the acquisition function evaluated at x
        """

        if levels is None:
            levels = [i for i in range(self.model.kern.parts[1].output_dim)]

        levels = np.asarray(levels)
        assert location.shape[1] == self.model.input_dim - 1

        # make new array to with levels at the end
        x = np.concatenate((np.tile(location, (levels.size, 1)), np.repeat(levels, location.shape[0])[:, None]), axis=1)

        # compute acquisition function
        acq =  np.reshape(self._integral_variance_reduction(x=x), (-1, levels.size), order='F') / \
               self.cost_functions.evaluate(location=location, levels=levels) / self.model.integrate()[1][0,0]

        return acq

    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, location, levels=None):
        """
        Evaluate the acquisition function with gradient

        :param x: location and fidelity
        :return: the acquisition function and its gradient evaluated at x
        """
        grad = self._acquisition_gradient(location=location, levels=levels) / self.model.integrate()[1][0,0]
        return self.evaluate(location=location, levels=levels), grad

    # helpers
    def _integral_variance_reduction(self, x):
        """
        Helper to compute the variance reduction of the integrated GP

        :param x: new candidate point x, contains the location and the fidelity to be evaluated
        :return: proportional to variance reduction of high fidelity integral for evaluating at x
        """
        pred_var, data_term, kmean_at_x = self._integral_variance_reduction_terms(x)

        return 1./pred_var * (data_term - kmean_at_x)**2

    def _integral_variance_reduction_terms(self, x):
        """
        Helper to compute the terms needed for the variance reduction of the integrated GP

        :param x: new candidate point x, contains the location and the fidelity to be evaluated
        :return: pred_var, data_term, kmean_at_x so that the
            integrated variance reduction = pred_var * (data_term - kmean_at_x) ** 2
        """
        pred_var = (self.model.predict(x, full_cov=False)[1] + self.model.Gaussian_noise[0])

        kmean_at_x = self._high_fidelity_kernel_mean(x=x).T

        data_term = np.dot(self._high_fidelity_kernel_mean(x=self.model.X), self._Kinv_Kx(x)).T

        return pred_var, data_term, kmean_at_x

    def _acquisition_gradient(self, location, levels=None):
        """
        The gradient of the acquisition function
        :param location: location where to evalute shape = (number of evaluations, input_dim-1)
        :param levels: list of fidelity levels, defaults to all
        :return: the gradient of the acquisition function, shape = (input_dim, location.shape[0], len(levels))
        """
        if levels is None:
            levels = [i for i in range(self.model.kern.parts[1].output_dim)]

        levels = np.asarray(levels)
        assert location.shape[1] == self.model.input_dim - 1

        # make new array to with levels at the end
        x = np.concatenate((np.tile(location, (levels.size, 1)), np.repeat(levels, location.shape[0])[:, None]), axis=1)

        c  = self.cost_functions.evaluate(location=location, levels=levels)[None, ...]
        dc = self.cost_functions.evaluate_gradient(location=location, levels=levels)

        q = self._integral_variance_reduction(x).reshape(1, location.shape[0], len(levels))
        dq = self._integral_variance_reduction_gradient(x).reshape(self.model.input_dim-1, location.shape[0], len(levels))

        return ((c*dq - q*dc) / c**2)

    def _integral_variance_reduction_gradient(self, x):
        """
        Computes the gradient of the acquisition function

        :param x: location at which to evaluate the gradient
        :return: the gradient at x
        """
        dpredvar_dx, ddat_dx, dqK_dx = self._integral_variance_reduction_gradient_terms(x)
        pred_var, data_term, kmean_at_x = self._integral_variance_reduction_terms(x)

        return (2./pred_var * (data_term - kmean_at_x) * (ddat_dx - dqK_dx) \
               - 1./pred_var**2 * dpredvar_dx * (data_term - kmean_at_x)**2)

    def _integral_variance_reduction_gradient_terms(self, x):
        """
        Compute the terms needed for the gradient of the acquisition function
        (see _integral_variance_reduction_terms)

        :param x: location at which to evaluate the gradient, contains fidelity levels
        :return: the gradient of (pred_var, data_term, kmean_at_x) at x
        """
        dpredvar_dx = -2. * (self.model.kern.dK_dx(x, self.model.X) *
                             (self._Kinv_Kx(x).T)[None,...]).sum(axis=2, keepdims=True)

        ddat_dx = np.dot(self.model.kern.dK_dx(x, self.model.X), self._Kinv_Kq())

        dqK_dx = self.model.kern.dqK_dx(x, levels=[0])

        return dpredvar_dx, ddat_dx, dqK_dx


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