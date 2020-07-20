
import numpy as np


def forrester_high(x):
    """
    High fidelity version of Forrester function

    Reference:
    Engineering design via surrogate modelling: a practical guide.
    Forrester, A., Sobester, A., & Keane, A. (2008).
    """
    x = np.atleast_2d(x).reshape(-1, 1)

    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def forrester_low(x):
    """
    Low fidelity version of Forrester function

    Reference:
    Engineering design via surrogate modelling: a practical guide.
    Forrester, A., Sobester, A., & Keane, A. (2008).
    """
    x = np.atleast_2d(x).reshape(-1, 1)

    return 0.5 * forrester_high(x) + 10 * (x - 0.5) + 5