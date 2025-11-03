import numpy as np

def Q(theta):
    """
    Computes Q(theta) = 1/theta + exp(theta)
    for theta in [0.1, 3].
    Works with scalar or 1D numpy array (for use with scipy.optimize.minimize).
    """
    theta = np.asarray(theta)  # gør input til array
    return np.sum(1/theta + np.exp(theta))  # returnér som skalar
