import _svd_lsq
import numpy as np

input = np.array([[1,2,3],[6,4,5],[8,9,7],[10,11,12]])
input = input.astype(np.float64)
Umat = _svd_lsq.svd(input)
print(Umat)

# get least square solution.
lh = np.array([1.,2.,4.,5.])
lsqq = _svd_lsq.lsq(input, lh)
print(lsqq)