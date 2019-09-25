import numpy as np
import svd_lsq

def test_py_simple_lsq():
    x = np.array([[1,2,3],[6,4,5],[8,9,7],[10,11,12]])
    y = np.array([1.,2.,4.,5.])
    coef = svd_lsq.lsq(x, y)
    expedted_coef = np.array([0.0436137, 0.3613707, 0.0529595])
    np.testing.assert_almost_equal(coef, expedted_coef)