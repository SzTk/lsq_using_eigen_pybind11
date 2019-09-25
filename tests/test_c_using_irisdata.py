import _svd_lsq
import numpy as np
import pandas as pd

def test_using_iris():
    irisdata = pd.read_csv('./tests/iris.data', header=None)
    input = irisdata.loc[:, [0, 1, 2, 3]]
    target = irisdata.loc[:, [4]] == 'Iris-setosa'

    input = np.ascontiguousarray(input.values.astype(np.float64))
    target = np.ascontiguousarray(target.values.astype(np.float64))
    target = target.reshape(150)

    lsq_coef = _svd_lsq._lsq(input, target)
    pred = np.dot(lsq_coef, input.T)
    rmse = np.sqrt(np.linalg.norm(pred - target))
    np.testing.assert_almost_equal(rmse, 1.327294809305265)
