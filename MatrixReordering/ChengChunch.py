import numpy as np

from MatrixReordering.Matrix import order_to_permutation, order_by


def cal_residue(matrix):
    """
    :param matrix:
    :return: residue, H
    """
    n = matrix.shape[0]
    row_mean = np.tile(np.mean(matrix, axis=0), (n, 1))  # a_iJ
    column_mean = row_mean.transpose()
    mean = np.mean(matrix)
    residue = np.asarray(matrix - row_mean - column_mean + mean)
    H = np.mean(residue ** 2)
    return residue, H


def biclustering(matrix, delta=0):
    matrix_saved = matrix.copy()
    n = matrix.shape[0]
    residue, H = cal_residue(matrix)
    N_ = np.arange(n)
    N = np.arange(n)
    order = []
    while N.shape[0] > 0:
        d = np.mean(residue**2, axis=0)
        # find the node with max residue
        idx = np.argmin(d)
        order.append(N_[idx])
        matrix = matrix[np.ix_(N != idx, N != idx)]
        N = np.arange(matrix.shape[0])
        N_ = N_[N_ != N_[idx]]
        residue, H = cal_residue(matrix)

    matrix = order_by(matrix_saved, order)

    return matrix, order

    # # TODO multiple node deletion
    #
    # I[I_] = -1
    # J[J_] = -1
    # I_ = I.copy()
    # J_ = J.copy()
    # I = I[I > 0]
    # J = J[I > 0]
    # row_mean = row_mean[I]
    # column_mean = column_mean[J]
    # mean = np.mean(matrix[I, J])
    # residue = matrix[I, J] - row_mean - column_mean + mean
    # H = np.mean(residue ** 2)
    # while I.shape[0] > 0 or J.shape[0] > 0:
    #     d = np.mean(residue, axis=0)
    #     e = np.mean(residue, axis=1)
    #     I_[I[d <= H]] = -1
    #     J_[J[e <= H]] = -1
    #     I = I[d > H]
    #     J = J[d > H]
    #     row_mean = row_mean[I]
    #     column_mean = column_mean[J]
    #     mean = np.mean(matrix[I, J])
    #     residue = matrix[I, J] - row_mean - column_mean + mean
    #     H = np.mean(residue ** 2)
    #
    # I = I < 0
    # J = J < 0
