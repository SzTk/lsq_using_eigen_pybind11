import numpy as np
from _svd_lsq import _lsq


def lsq(x, y):
    """
    Compute reast square regression.

    Parameters
    ----------
    x : np.ndarray (m,1)
    y : np,ndarray (1,)

    Returns
    -------
    np.ndarray
        Coefficients of OLS model.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    coef = _lsq(x, y)
    return coef
