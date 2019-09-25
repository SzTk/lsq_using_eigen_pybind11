import _svd_lsq
import numpy as np

def test_simple_svd():
    input = np.array([[1,2,3],[6,4,5],[8,9,7],[10,11,12]])
    input = input.astype(np.float64)
    Umat = _svd_lsq._svd(input)
    assert Umat is not None

def test_simple_lsq():
    input = np.array([[1,2,3],[6,4,5],[8,9,7],[10,11,12]])
    input = input.astype(np.float64)
    lh = np.array([1.,2.,4.,5.])
    lsqq = _svd_lsq._lsq(input, lh)
    assert lsqq is not None