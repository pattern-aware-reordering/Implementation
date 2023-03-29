# BW, LA, PR come from: Matrix Reordering Methods for Table and Network Visualization
# DIS comes from Optimizing the Ordering of Tables With Evolutionary Computation
import math
import numpy as np
import networkx as nx
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import scale
from Utils.IO import read_graph, write_matrix
from functools import lru_cache


def PHI(matrix, scaled=False):
    """the sum of the similarities of adjacent leaves in the ordering
    """
    n = matrix.shape[0]
    X = matrix.copy()[1:]
    Y = matrix.copy()[:-1]
    phi = np.sum(np.sqrt(np.sum((X-Y)**2, axis=1)))
    return phi


@lru_cache(maxsize=32)
def factorial(n):
    return math.factorial(n)


def TSSA(matrix, scaled=False):
    """Defined by: An Effective Two-Stage Simulated Annealing Algorithm for the Minimum Linear Arrangement Problem
    """
    nonzerors = np.nonzero(matrix)
    if nonzerors[0].shape[0] == 0:
        return 0
    D = np.abs(nonzerors[0] - nonzerors[1], dtype=np.int32) * \
        np.asarray(matrix[nonzerors], dtype=np.int32)
    k, dk = np.unique(D, return_counts=True)
    n = k.shape[0]
    fac_n = factorial(n)
    fac_n_k = np.array(list(map(factorial, k + n)))
    stress = np.sum(k * dk) + np.sum(fac_n / fac_n_k * dk)
    if scaled:
        stress = 1 - (np.sum(k * dk) +
                      np.sum(fac_n / fac_n_k * dk)) / (n * m + 2)
    return stress


def MI(matrix, scaled=False, isUndirected=True):
    """compute Moran's I measure
    defined by: https://github.com/nvbeusekom/reorder.js
    """
    n = matrix.shape[0]
    if isUndirected:
        matrix = matrix + np.eye(n)  # the diagonal of matrices in reorder.js are all 1
    m = np.sum(matrix)
    N = n * n
    W = ((n - 2) ** 2) * 4 + (n - 2) * 3 * 2 + (n - 2) * 3 * 2 + 8

    num = 0
    denom = 0
    for j in range(n):
        for i in range(n):
            denom += (matrix[j, i] - m / N)**2
            innersum = 0
            y = max(0, j - 1)
            while y < min(n, j + 2):
                x = max(0, i - 1)
                while x < min(n, i + 2):
                    if y != j or x != i:
                        if i - x >= -1 and i - x <= 1 and j == y:
                            innersum += (matrix[j, i] * N - m) * \
                                (matrix[y, x] * N - m)
                        if i == x and j - y >= -1 and j - y <= 1:
                            innersum += (matrix[j, i] * N - m) * \
                                (matrix[y, x] * N - m)
                    x += 1
                y += 1
            num += innersum
    if num == 0 and denom == 0:
        return 1
    res = (N / W) * (num / denom) / (N * N)
    if scaled:
        res = (res + 1) / 2
    return res


def BW(matrix, scaled=False):
    """
    Bandwidth metric for matrix ordering
    :param matrix: n * n numpy matrix
    :return: the stress
    """
    nonzerors = np.nonzero(matrix)
    if nonzerors[0].shape[0] == 0:
        return 0
    stress = np.max(
        np.abs(nonzerors[0] - nonzerors[1]) * np.asarray(matrix[nonzerors]))
    if scaled:
        n = matrix.shape[0]
        stress = 1 - stress / n
    return stress


def LA(matrix, scaled=False):
    """
    linear arrangement metric for matrix ordering
    :param matrix: n * n numpy matrix
    :return: the stress
    """
    nonzerors = np.nonzero(matrix)
    if nonzerors[0].shape[0] == 0:
        return 0
    stress = np.sum(
        np.abs(nonzerors[0] - nonzerors[1]) * np.asarray(matrix[nonzerors]))
    if scaled:
        n = matrix.shape[0]
        m = nonzerors[0].shape[0]
        stress = 1 - stress / (n * m)
    return stress


def PR(matrix, scaled=False):
    """
    profile metric for matrix ordering
    :param matrix: n * n numpy matrix
    :return:
    """
    matrix = np.asarray(matrix).copy()
    n = matrix.shape[0]
    stress = 0
    for i in range(n):
        min_j = i
        for j in range(i):
            if matrix[i, j] > 0:
                min_j = j
                break
                    
        stress += (i - min_j)  # * matrix[i, min_j]

    if scaled:
        n = matrix.shape[0]
        stress = 1 - stress / (n * (n - 1) / 2)

    return stress


def DIS(matrix):
    """
    dissimilarity
    :param matrix: n * n numpy matrix
    :return:
    """
    matrices = [matrix.copy() for i in range(8)]

    matrices[0][1:, 1:] = matrix[:-1, :-1]
    matrices[1][1:, :] = matrix[:-1, :]
    matrices[2][1:, :-1] = matrix[:-1, 1:]
    matrices[3][:, 1:] = matrix[:, :-1]
    matrices[4][:, :-1] = matrix[:, 1:]
    matrices[5][:-1, 1:] = matrix[1:, :-1]
    matrices[6][:-1, :] = matrix[1:, :]
    matrices[7][:-1, :-1] = matrix[1:, 1:]

    stress = np.sum([np.sum(np.abs(matrices[i] - matrix))
                     for i in range(len(matrices))])
    return stress


if __name__ == '__main__':
    matrix = np.ones((2, 2)) - np.eye(2)
    print(TSSA(matrix), LA(matrix))
    matrix = np.ones((3, 3)) - np.eye(3)
    print(TSSA(matrix), LA(matrix))
    matrix = np.ones((4, 4)) - np.eye(4)
    print(TSSA(matrix), LA(matrix))
    matrix = np.ones((100, 100)) - np.eye(100)
    print(TSSA(matrix), LA(matrix))
