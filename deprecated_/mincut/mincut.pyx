cimport cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt
from cython.operator import postincrement as inc
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "float.h":
    double DBL_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def min_cut(np.ndarray ssm, Py_ssize_t K):
    cdef Py_ssize_t N = ssm.shape[0]

    # cdef double[:,::1] C = np.ones((N, K), dtype=np.float32) * DBL_MAX
    cdef double[:,::1] C = np.ones((N, K), dtype=np.float32, order="C") * DBL_MAX
    cdef int[:,::1] B = np.ones((N, K), dtype=np.int32)

    C[0,0] = 0.

    cdef list temp, obj
    cdef Py_ssize_t i, j, k, ind
    for i in range(1,N):
        temp = [(ssm[j:i, j:i].sum() / 2., ssm[j:i, :j].sum() + ssm[j:i, i:].sum()) for j in range(i)]
        for k in range(1,K):
            obj = [C[j, k-1] + item[1]/(item[0]+item[1]) for j, item in enumerate(temp)]
            ind = np.argmin(obj)
            B[i,k] = ind
            C[i,k] = obj[ind]
    
    # backtrack
    cdef list boundary = []
    cdef Py_ssize_t prev_b = N - 1
    cdef list loop = list(range(K))[::-1][:-1]
    boundary.append(prev_b)
    for k in loop:
        prev_b = B[prev_b,k]
        boundary.append(prev_b)
    boundary = boundary[::-1] # reverse
    # boundary = boundary[1:-1] # chop start and end
    # boundary = [item - 0.5 for item in boundary] # adjust
    # boundary[0], boundary[-1] = boundary[0] + 0.5, boundary[-1] + 0.5
    return boundary

