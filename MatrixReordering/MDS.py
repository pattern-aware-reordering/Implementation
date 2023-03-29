import networkx as nx
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from MatrixReordering.Matrix import order_by, cal_pairwise_dist, order_to_permutation


def MDS(matrix, dist_func=cal_pairwise_dist):
    """
    :param matrix: n * n matrix
    :param dist_func: function(matrix) => distance matrix
    :return: (n_samples, n_dims)
    """
    data = np.asarray(matrix)
    n, d = data.shape

    dist = dist_func(data)
    # dist = squareform(pdist(data))
    dist[dist < 0] = 0
    T1 = np.ones((n, n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis=1, keepdims=True)/n
    T3 = np.sum(dist, axis=0, keepdims=True)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)

    n_dims = 1
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    X = picked_eig_vector*picked_eig_val**(0.5)

    order = np.argsort(X.reshape((n, ))).flatten()


    matrix = order_by(matrix, order)
    return matrix, order
