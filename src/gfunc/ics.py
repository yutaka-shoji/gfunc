import numpy as np
from scipy.integrate import quad

from . import libics


# INFINITE CYLINDER SOURCE MODEL
def _icsquad(Fo, R):
    return quad(libics.integrand, 0, np.inf, args=(Fo, R))[0]


_uicsquad = np.frompyfunc(_icsquad, 2, 1)


def ics(Fo, R):
    return _uicsquad(Fo, R)
