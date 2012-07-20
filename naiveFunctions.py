# Author: Martin Smith
# Created on: 7/18/12
# Updated on: 7/18/12

import numpy

def naive_mul(A,B):
    C = numpy.zeros((A.shape[0], B.shape[1]))
    for rowI in range(C.shape[0]):
        for colI in range(C.shape[1]):
            sum = 0.0
            for i in range(A.shape[1]):
                sum += A[rowI, i] * B[i,colI]
            C[rowI, colI] = sum
    return C


def numpy_mul(A,B):
    return A.dot(B)


def matrix_multiply(A,B):
    return naive_mul(A,B)
