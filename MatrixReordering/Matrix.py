import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.distance import pdist, squareform


def order_by(matrix, order):
    """
    order a matrix by an order,
    :param matrix: np matrix
    :param order: list
    :return: the reordered matrix
    """
    matrix = matrix.copy()
    idx = np.asarray(order)
    matrix[:] = matrix[:, idx]
    matrix[:] = matrix[idx, :]
    return matrix


def cal_pairwise_dist(matrix):
    """
    (a-b)^2 = a^2 + b^2 - 2*a*b
    """
    unit_matrix = matrix.copy()
    pair_dist_mat = squareform(pdist(unit_matrix))
    return pair_dist_mat


def cal_pairwise_moran_dist(matrix):
    """
    return C_B * B + C_W * W - 1
    """

    n = matrix.shape[0]
    # the diagonal of matrices in reorder.js are all 1
    matrix = matrix + np.eye(n)
    m = np.sum(matrix)

    pair_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum((matrix[i] * (n**2) - m) * (matrix[j] * (n**2) - m))
            pair_dist_mat[i, j] = pair_dist_mat[j, i] = -d

    return pair_dist_mat


if __name__ == "__main__":
    matrix = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    cal_pairwise_moran_dist(matrix)


def order_to_permutation(order):
    """
    transform order to permutation
    :param order: the order of nodes, e.g., [0, 2, 3, 1]
    :return: the permutation of nodes: [0, 3, 1, 2] (indeces of nodes in the order)
    """
    n = len(order)
    permutation = [None] * n
    i = 0
    for node in order:
        permutation[node] = i
        i += 1

    return permutation


def permutation_to_order(permutation):
    """
    transform order to permutation
    :param permutation: the permutation of nodes: [0, 3, 1, 2] (indeces of nodes in the order)
    :return: order: the order of nodes, e.g., [0, 2, 3, 1]
    """
    if isinstance(permutation, np.ndarray):
        permutation = permutation.tolist()

    n = len(permutation)
    order = [None] * n
    i = 0
    for index in permutation:
        order[index] = i
        i += 1

    return order


def geo_dis_mat(matrix):
    """
    Floyd Warshall
    :param matrix: adjacency matrix
    :return: 
    """
    G = nx.from_numpy_matrix(matrix)
    n = matrix.shape[0]
    distance = nx.floyd_warshall_numpy(G)

    return distance

def BFS(matrix, start):
    # Visited vector to so that a
    # vertex is not visited more than
    # once Initializing the vector to
    # false as no vertex is visited at
    # the beginning
    n = matrix.shape[0]
    visited = [False] * n
    q = [start]

    # Set source as visited
    visited[start] = True
    order = []
    while len(q):
        vis = q[0]
        # Print current node
        q.pop(0)
        # For every adjacent vertex to
        # the current vertex
        for i in range(n):
            if matrix[vis, i] == 1 and (not visited[i]):
                # Push the adjacent node
                # in the queue
                q.append(i)
                order.append((i, vis))
                # set
                visited[i] = True
    return order


def connected_components(matrix):
    graph = csr_matrix(matrix)
    n_components, labels = csgraph.connected_components(
        csgraph=graph, directed=False, return_labels=True)
    components = [[] for i in range(n_components)]
    for k in range(len(labels)):
        label = labels[k]
        components[label].append(k)
    sub_matrices = []
    for component in components:
        component = list(component)
        sub_matrix = matrix[np.ix_(component, component)]
        sub_matrices.append(sub_matrix)
    return sub_matrices
