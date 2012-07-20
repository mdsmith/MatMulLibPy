#! /usr/local/bin/python3

from mullib import matrix_multiply
from mullib import matrix_multiply_test
import numpy

A = numpy.random.random_sample((2000,64))
B = numpy.random.random_sample((64,64))

def golden(A,B):
    return A.dot(B)

if __name__ == '__main__':
    C = matrix_multiply_test(A, B)
    '''
    gC = golden(A,B)
    if (C==gC).all():
        result = "PASSED"
    else:
        result = "FAILED"
    print("Algorithm ", result, " the test!")
    '''
