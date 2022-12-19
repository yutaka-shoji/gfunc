import numpy as np
from scipy.integrate import quad

from . import libcmils

ep = 1.0e-3
V = 6.0e2
N = 6
narr = np.arange(0, N + 1, 2)
tmpI2 = np.zeros((len(narr)))


def _cmils1uquad(n, Fo, RA, RB, RD, alpha, k):
    tmpI2 = quad(
        libcmils.integrand,
        ep,
        V,
        epsabs=1.0e-6,
        epsrel=0,
        limit=512,
        args=(n, Fo, RA, RB, RD, alpha, k),
    )[0]

    return tmpI2


_ucmils1uquad = np.frompyfunc(_cmils1uquad, 7, 1)


def cmils1u(Fo, RA, RB, RD, alpha, k):
    # COMPOSITE-MEDUIM INFINITE LINE SOURCE MODEL
    I1 = ep * ep * Fo / (2.0 * k)

    nmesh, Fomesh = np.meshgrid(narr, Fo)
    tmpI2 = _ucmils1uquad(nmesh, Fomesh, RA, RB, RD, alpha, k)

    I2 = tmpI2[:, 0] + 2 * np.sum(tmpI2[:, 1:], axis=1)

    f_cmils = I1 + I2

    return f_cmils
