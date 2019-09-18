
import numpy as np
import numpy.linalg as slinalg
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

    pmin = ProbMinHDR(mu, var)

    if not with_derivatives:
        return pmin.logP

    return pmin.logP, pmin.dlogPdMu, pmin.dlogPdSigma, pmin.dlogPdMudMu


class ProbMinLoop():
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, with_derivatives: bool):
        """
        Computes the approximate probability of every of the N representer points to be the minimum.
        This requires the solution to a linearly constrained Gaussian integral, which is solved using HDR from the
        constrained_gaussian_integrals package.
        :param mu: mean of representer points (from GP), dims (N,).
        :param Sigma: covariance of representer points (from GP), dims (N, N).
        """
        self.N = mu.shape[0]
        self.mu = mu
        self.Sigma = Sigma
        self.L = slinalg.cholesky(Sigma, lower=True)

        # initialize crucial values
        self.log_pmin = np.zeros((self.N, 1))

        if self.with_derivatives:
            self.deriv = None

    def run(self):
        """
        Compute the logarithm of the approximate integral for p_min using HDR

        :return: None
        """
        for i in range(self.N):
            # compute the p_min for the current representer point
            pmin_i = ProbMinSingle(self.mu, self.L, self.with_derivates)
            self.log_pmin[i,0] = pmin_i.log_pmin()
        return


class ProbMinSingle():
    def __init__(self, i, mu: np.ndarray, cholSigma: np.ndarray, with_derivates: bool):
        """
        Computes the approximate probability of _ONE_ of the N representer points to be the minimum.
        This requires the solution to a linearly constrained Gaussian integral, which is solved using HDR from the
        constrained_gaussian_integrals package.
        :param i: index of currently considered representer point
        :param mu: mean of representer points (from GP), dims (N,).
        :param Sigma: covariance of representer points (from GP), dims (N, N).
        :param with_derivatives: If True than also the gradients are computed (relevant for storage only)
        """
        self.i = i
        self.N = mu.shape[0]
        self.mu = mu
        self.L = cholSigma
        self.with_derivatives = with_derivates

        # linear constraints
        M = np.split(np.eye(self.N-1), np.array([self.i]), axis=1)
        M = np.hstack((A[0], -np.ones((N, 1)), A[1]))
        self.lincon = cgi.LinearConstraints(np.dot(M, self.L), np.dot(M, self.mu))

        if self.with_derivatives:
            self.first_moment = None
            self.second_moment = None

    def log_pmin(self):
        """
        Compute the logarithm of the approximate integral for p_min using HDR
        :return: np.float log of pmin
        """
        pass

    def samples(self, N_samples):
        """
        Draw samples from the argument of the integral
        :param N_samples: number of samples drawn
        :return: samples (np.ndarray)
        """

    def get_moments(self, f_samples):
        """
        Computes the first and second moment of f w.r.t. the integrand of pmin
        :param f_samples: samples from the integrand
        :return: first moment of f (np.ndarray)
        """
        pass

    def dlogPdMu(self):
        """
        Gradient of logP for every representer point w.r.t. mu
        :return: with gradient, np.ndarray, dim (N)
        """
        pass

    def dlogPdSigma(self):
        """
        Gradient of logP for every representer point w.r.t. Sigma.
        Since Sigma is symmetric, only N(N+1)/2 values need to be stored per representer point
        :return: (N(N+1)/2,) np.ndarray with gradient
        """
        pass

    def dlogPdMudMu(self):
        """
        Hessian w.r.t. mean mu
        :return: Hessian, dim (N,N)
        """
        pass
