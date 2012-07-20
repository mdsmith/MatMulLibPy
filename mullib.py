

import time

mm_functions = {}
from naiveFunctions import naive_mul as naivemm
from naiveFunctions import numpy_mul as numpymm
mm_functions['naive'] = naivemm
mm_functions['numpy'] = numpymm
mm = naivemm
try:
    from oclFunctions import matrix_multiply as oclmm
    mm_functions['ocl'] = oclmm
    #mm = oclmm
except ImportError:
    pass

import numpy

class ShapeException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def matrix_multiply(A, B):
    check_shape(A,B)
    # Really should check and clean input as well
    return mm(A,B)

def check_shape(A,B):
    if B.shape[0] != A.shape[1]:
        raise ShapeException("Arrays have incompatible dimensions!")

def matrix_multiply_test(A,B):
    check_shape(A,B)
    goldenC = A.dot(B)
    for name, mmf in mm_functions.items():
        print("Method: ", name)
        tstart = time.clock()
        C = mmf(A,B)
        print("Time: ", time.clock() - tstart)
        if (C == goldenC).all():
            result = "PASSED"
        else:
            result = "FAILED"
            #print(C)
            #print(goldenC)
        print("Result: ", result)
    return matrix_multiply(A,B)
