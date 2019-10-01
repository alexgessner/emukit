
import numpy as np
import scipy.linalg as slinalg
import constrained_gaussian_integrals as cgi



def joint_min(mu: np.ndarray, var: np.ndarray, with_derivatives: bool=False) -> np.ndarray:
    """
    Computes the probability of every given point (of N representer points) to be the minimum
    based on the HDR[1] algorithm.
    [1] A. Gessner, O. Kanjilal, P. Hennig
    Efficient Black-Box Computation of Integrals over Linearly Constrained Gaussians
    (In preparation)

    :param mu: Mean value of each of the N points, dims (N,).
    :param var: Covariance matrix for all points, dims (N, N).
    :param with_derivatives: If True than also the gradients are computed.
    :returns: pmin distribution, dims (N, 1).
    """

    pmin = ProbMinLoop(mu, var, with_derivatives)
    pmin.run()

    if not with_derivatives:
        return pmin.log_pmin

    return pmin.log_pmin, pmin.dlogPdMu, pmin.dlogPdSigma, pmin.dlogPdMudMu


class ProbMinLoop():
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, with_derivatives: bool, N_subset: int = 16,
                 N_hdr: int = 1024):
        """
        Computes the approximate probability of every of the N representer points to be the minimum.
        This requires the solution to a linearly constrained Gaussian integral, which is solved using HDR from the
        constrained_gaussian_integrals package.
        :param mu: mean of representer points (from GP), dims (N,).
        :param Sigma: covariance of representer points (from GP), dims (N, N).
        :param N_subset: Number of samples used to construct the subset sequence, defaults to 8
        :param N_hdr: Number of samples used for HDR, defaults to 1024
        """
        self.N = mu.shape[0]
        self.mu = mu
        self.Sigma = Sigma
        self.L = slinalg.cholesky(Sigma + 1.e-10*np.eye(self.N), lower=True)
        self.with_derivatives = with_derivatives
        self.N_subset = N_subset
        self.N_hdr = N_hdr

        # initialize crucial values
        self.log_pmin = np.zeros((self.N, 1))

        if self.with_derivatives:
            self.deriv = None

    def run(self):
        """
        Compute the logarithm of the approximate integral for p_min using HDR
        :return: log p_min at all representer points
        """

        # handle extra arguments if available, otherwise set default values
        # n_skip = kwargs['n_skip'] if 'n_skip' in kwargs.keys() else 3
        # verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else False

        for i in range(self.N):
            # compute the p_min for the current representer point
            pmini = ProbMinSingle(i, self.mu, self.L, self.with_derivatives, self.N_subset, self.N_hdr)
            self.log_pmin[i,0] = pmini.log_pmin()
            print('Done with element ', i)

            # TODO: Add derivatives

        return self.log_pmin

    def run_idx(self, idx):
        """
        Compute the logarithm of the approximate integral for p_min using HDR for a list of indices only
        :param idx: list of indices
        :return: log p_min for given representer points
        """
        for i in idx:
            pmini = ProbMinSingle(i, self.mu, self.L, self.with_derivatives, self.N_subset, self.N_hdr)
            self.log_pmin[i, 0] = pmini.log_pmin()
            print('Done with element ', i)
        return self.log_pmin[idx, :]



class ProbMinSingle():
    def __init__(self, i, mu: np.ndarray, cholSigma: np.ndarray, with_derivatives: bool, N_subset: int, N_hdr: int):
        """
        Computes the approximate probability of _ONE_ of the N representer points to be the minimum.
        This requires the solution to a linearly constrained Gaussian integral, which is solved using HDR from the
        constrained_gaussian_integrals package.
        :param i: index of currently considered representer point
        :param mu: mean of representer points (from GP), dims (N,).
        :param Sigma: covariance of representer points (from GP), dims (N, N).
        :param with_derivatives: If True than also the gradients are computed (relevant for storage only)
        :param N_subset: Number of samples used to construct the subset sequence
        :param N_hdr: Number of samples used for HDR
        """
        self.i = i
        self.N = mu.shape[0]
        self.mu = mu
        self.L = cholSigma
        self.with_derivatives = with_derivatives
        self.N_subset = N_subset
        self.N_hdr = N_hdr

        # linear constraints
        M = np.split(np.eye(self.N-1), np.array([self.i]), axis=1)
        M = np.hstack((M[0], -np.ones((self.N - 1, 1)), M[1]))
        self.lincon = cgi.LinearConstraints(np.dot(M, self.L), np.dot(M, self.mu))

        # subset simulation
        self.subsetsim = cgi.subset_simulation.SubsetSimulation(self.lincon, self.N_subset, 0.5, n_skip=9)
        self.subsetsim.run_loop(verbose=False)

        # set up HDR
        self.hdr = cgi.hdr.HDR(self.lincon, self.subsetsim.tracker.shifts(), self.N_hdr,
                               self.subsetsim.tracker.x_inits(), n_skip=2) # TODO: surface n_skip

        if self.with_derivatives:
            self.first_moment = None
            self.second_moment = None

        # store samples when computing the integral, and the integral
        self.X_samples = None

    def log_pmin(self):
        """
        Compute the logarithm of the approximate integral for p_min using HDR
        :return: np.float log of pmin
        """
        self.X_samples = self.hdr.run(verbose=False)
        return self.hdr.tracker.log_integral()

    def samples(self, N_samples):
        """
        Draw samples from the argument of the integral
        :param N_samples: number of samples drawn
        :return: samples (np.ndarray)
        """
        # ESS from domain
        if self.X_samples is None:
            # get initial samples from subset simulation
            x_init = self.subsetsim.tracker.sequence_of_nestings[-1].X_inside()
        else:
            # reuse samples from the integral computation
            x_init = self.X_samples
        ess = cgi.loop.EllipticalSliceOuterLoop(N_samples - x_init.shape[-1],
                                                self.lincon, n_skip=2, x_init=x_init)
        ess.run_loop()

        # add samples to the records
        self.X_samples = ess.loop_state.X

        # Now these samples are drawn in the whitened space, so they need to be back-transformed
        return np.dot(self.L, ess.loop_state.X) + self.mu

    def get_moments(self, f_samples):
        """
        Computes the first and second moment of f w.r.t. the integrand of pmin
        :param f_samples: samples from the integrand
        :return: first moment of f (np.ndarray)
        """
        return self.get_first_moment(f_samples), self.get_second_moment(f_samples)

    def get_first_moment(self, f_samples):
        """
        Computes the first moment of f w.r.t. the integrand of pmin
        :param f_samples: samples from the integrand
        :return: first moment of f (np.ndarray)
        """
        return np.mean(f_samples, axis=1)

    def get_second_moment(self, f_samples):
        """
        Computes the second moment of f w.r.t. the integrand of pmin
        :param f_samples: samples from the integrand
        :return: second moment of f (np.ndarray)
        """
        return np.dot(f_samples, f_samples.T)/(f_samples.shape[1] - 1)

    def dPdMu(self, f_samples):
        """
        Gradient of P for every representer point w.r.t. mu
        :param f_samples: samples from the integrand
        :return: with gradient, np.ndarray, dim (N)
        """
        f_mu_mean = self.get_first_moment(f_samples - self.mu)
        return slinalg.solve_triangular(self.L.T, slinalg.solve_triangular(self.L, f_mu_mean, lower=True), lower=False)

    def dPdSigma(self, f_samples):
        """
        Gradient of P for every representer point w.r.t. Sigma.
        Since Sigma is symmetric, only N(N+1)/2 values need to be stored per representer point
        :param f_samples: samples from the integrand
        :return: (N(N+1)/2,) np.ndarray with gradient
        """
        f_mu = f_samples - self.mu
        A = self.get_second_moment(f_mu) - np.dot(self.L, self.L.T)
        A = slinalg.solve_triangular(self.L.T, slinalg.solve_triangular(self.L, A, lower=True), lower=False)
        A = slinalg.solve_triangular(self.L.T, slinalg.solve_triangular(self.L, A.T, lower=True), lower=False)
        return 0.5 * A[np.tril_indices(f_samples.shape[0])]

    def dPdMudMu(self, f_samples):
        """
        Hessian w.r.t. mean mu
        :param f_samples: samples from the integrand
        :return: Hessian, dim (N,N)
        """
        M1 = self._restore_symmetric_matrix_from_vector(2.*self.dPdSigma(f_samples), f_samples.shape[0])
        dlpdmu = self.dPdMu(f_samples)
        M2 = np.outer(dlpdmu, dlpdmu)
        return M1 - M2

    def dlogPdMu(self, f_samples):
        """
        Gradient of logP for every representer point w.r.t. mu
        :param f_samples: samples from the integrand
        :return: with gradient, np.ndarray, dim (N)
        """
        return self.dPdMu(f_samples) / self.hdr.tracker.integral()

    def dlogPdSigma(self, f_samples):
        """
        Gradient of logP for every representer point w.r.t. Sigma.
        Since Sigma is symmetric, only N(N+1)/2 values need to be stored per representer point
        :param f_samples: samples from the integrand
        :return: (N(N+1)/2,) np.ndarray with gradient
        """
        return self.dPdSigma(f_samples) / self.hdr.tracker.integral()

    def dlogPdMudMu(self, f_samples):
        """
        Hessian w.r.t. mean mu
        :param f_samples: samples from the integrand
        :return: Hessian, dim (N,N)
        """
        M1 = self._restore_symmetric_matrix_from_vector(2.*self.dlogPdSigma(f_samples), f_samples.shape[0])
        dlpdmu = self.dlogPdMu(f_samples)
        M2 = np.outer(dlpdmu, dlpdmu)
        return M1 - M2

    def _restore_symmetric_matrix_from_vector(self, v, size):
        """
        For a vector v that is the triangular part of a symmetric matrix, this routine restores the matrix
        :param v: triangular part of symmetric matrix stored as vector
        :param size: size of array
        :return: symmetric array (size, size)
        """
        M = np.zeros((size, size))
        M[np.tril_indices(size)] = v
        return M + M.T - np.diag(M.diagonal())