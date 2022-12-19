import numpy as np
from scipy.integrate import quad
from scipy.special import expn, i0, k0


def mils(Fo, Pe, phi):
    # MOVING INFINITE LINE SOURCE MODEL
    return np.exp(Pe / 2 * np.cos(phi)) * _Wh(1 / (4 * Fo), Pe * Pe / 16)


def milsmean(Fo, Pe):
    # MOVING INFINITE LINE SOURCE MODEL ANGLE MEAN
    return i0(Pe / 2) * _Wh(1 / (4 * Fo), Pe * Pe / 16)


def _Wh(x, b, isquad=False):
    if isquad:
        v_gamma = np.vectorize(_gamma_quad)
        W = v_gamma(0, x, b)
    else:
        Nl = 10
        W = np.zeros_like(x)
        idx = x >= np.sqrt(b)
        W[~idx] = 2 * k0(2 * np.sqrt(b))
        for n in range(Nl):
            W[idx] += (
                (-b / x[idx]) ** n / float(np.math.factorial(n)) * expn(n + 1, x[idx])
            )
            W[~idx] += -(
                (-x[~idx]) ** n / float(np.math.factorial(n)) * expn(n + 1, b / x[~idx])
            )
    return W


def _gamma_quad(a, x, b):
    return quad(_integrand, x, np.inf, args=(a, b))[0]


def _integrand(t, a, b):
    f_int = t ** (a - 1) * np.exp(-t - b / t)
    return f_int
