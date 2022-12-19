from scipy.special import exp1


def ils(Fo):
    # INFINITE LINE SOURCE MODEL
    return exp1(1 / (4 * Fo))
