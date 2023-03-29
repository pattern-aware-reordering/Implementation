# Reducing the bandwidth of sparse symmetric matrices
import networkx as nx
import numpy as np

from MatrixReordering.Matrix import order_by, order_to_permutation


def CuthillMckee(matrix):
    G = nx.from_numpy_matrix(matrix)
    components = list(nx.connected_components(G))
    n = len(G.nodes)
    new_matrix = np.zeros((n, n))
    order = []
    if len(components) > 1:
        count = 0
        for component in components:
            component = np.asarray(list(component))
            length = component.shape[0]
            sub_matrix = matrix[np.ix_(component, component)]
            if length > 1:
                sub_matrix, sub_order = CuthillMckee(sub_matrix)
                sub_order = component[sub_order].tolist()
            else:
                sub_order = component.tolist()
            new_matrix[count:count + length, count:count + length] = sub_matrix
            order += sub_order
            count += length

        return new_matrix, order

    degree = np.array(sorted(G.degree, key=lambda x: x[0]))[:, 1]
    v = list(G.nodes)
    n = len(v)
    v_start = np.random.choice(np.arange(n)[degree == np.min(degree)])
    traversed_v = set()
    order = [v_start]
    traversed_v.add(v_start)

    i = 0
    while i < n:
        neighbors = np.array(list(nx.neighbors(G, order[i])))
        neighbors = neighbors[(degree[neighbors] / matrix[order[i], neighbors]).argsort()]
        for v_n in neighbors:
            if v_n not in traversed_v:
                traversed_v.add(v_n)
                order.append(v_n)
                if len(order) == n:
                    break
        if len(order) == n:
            break
        i += 1

    # permutation = order_to_permutation(order)
    matrix = order_by(matrix, order)

    return matrix, order
