# Fast optimal leaf ordering for hierarchical clustering
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform


from MatrixReordering.Matrix import order_by, cal_pairwise_dist


def optimal_leaf_ordering(matrix, dist=cal_pairwise_dist, Z=None):
    pair_dist_mat = dist(matrix)
    pair_dist_mat -= np.min(pair_dist_mat)
    pair_dist_mat *= np.ones(pair_dist_mat.shape) - np.eye(pair_dist_mat.shape[0])
    y = squareform(pair_dist_mat)
    if Z is None:
        Z = hierarchy.ward(y) # [cluster1, cluster2, distance, # of elements]
    # hierarchy.leaves_list(Z)
    order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, np.asmatrix(matrix)))

    order = order.tolist()
    # permutation = order_to_permutation(order)
    # matrix = order_by(matrix, permutation)
    # return matrix, permutation

    matrix = order_by(matrix, order)
    return matrix, order
