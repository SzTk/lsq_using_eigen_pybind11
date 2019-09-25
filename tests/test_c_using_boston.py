import _svd_lsq
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_ridge():
    boston = datasets.load_boston()

    lsqq = _svd_lsq._lsq(boston.data, boston.target)
    target_pred = np.dot(lsqq, boston.data.T)
    ridge_coef = _svd_lsq._ridge(boston.data, boston.target, 1e+6)
    target_pred_ridge = np.dot(ridge_coef, boston.data.T)

    norm_ridge = np.linalg.norm(target_pred_ridge - boston.target)
    norm_lsq = np.linalg.norm(target_pred - boston.target)
    np.testing.assert_almost_equal(norm_ridge, 199.21913385463492)
    np.testing.assert_almost_equal(norm_lsq, 110.58049674804326)