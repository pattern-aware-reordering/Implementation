import math
import scipy.stats as st
import numpy as np
import random
import numpy.random as rn
from MatrixReordering.Matrix import order_by, permutation_to_order
from MatrixReordering.Metrics import TSSA

# An effective two-stage simulated annealing algorithm for the minimum linear arrangement problem


def generate_independent_solutions(matrix, order, cost_fun, runtimes=1000):
    tssa = []
    position = order.copy()
    for i in range(runtimes):
        random.shuffle(position)
        ordered_matrix = order_by(matrix, position)
        tssa.append(cost_fun(ordered_matrix))
    mean = np.mean(tssa)
    std = np.std(tssa)
    return mean, std


def fim(matrix):
    """
    3.4 Initial solution,
    select the node with minimum links to unselected nodes (U, Tr),
    while with maximum links to selected nodes (P, Tl);
    :param matrix:
    :return:
    """
    n = matrix.shape[0]
    P = np.zeros(n, dtype=np.bool)  # selected nodes
    U = np.ones(n, dtype=np.bool)  # unselected nodes
    Dg = np.sum(matrix, axis=1) - np.diag(matrix)  # degree
    node = 0
    order = []
    while len(order) < n:
        order.append(node)
        U[node] = False
        P[node] = True
        NP = np.tile(P, (n, 1))
        NTl = np.sum(NP * matrix, axis=1)
        NTr = Dg - NTl
        NSf = NTr - NTl
        NSf[P] = 1e6
        node = int(np.argmin(NSf))

    return np.asarray(order)
    # permutation = order_to_permutation(order)
    # return permutation


def median(matrix, order, u):
    permutation = permutation_to_order(order)
    permutation = np.asarray(permutation)
    zero_diag_matrix = matrix.copy() - np.diag(np.diag(matrix))
    neighbor_idx = np.where(zero_diag_matrix[u, :] > 0)[0]
    if neighbor_idx.shape[0] == 0:
        return u
    neighbor_idx = neighbor_idx[permutation[neighbor_idx].argsort()]
    median_u = np.floor(np.median(permutation[neighbor_idx]))
    return int(median_u)


def swap_nodes(order, node_1, node_2):
    pos_1 = np.where(order == node_1)[0]
    pos_2 = np.where(order == node_2)[0]
    new_order = order.copy()
    new_order[pos_2] = node_1
    new_order[pos_1] = node_2
    return new_order


def find_best_order(matrix, order, u, v_range, cost_fun):
    if v_range.shape[0] == 0:
        return order
    MU = []
    MU_tssa = []
    orders = []
    for v in v_range:
        MU.append(v)
        new_order = swap_nodes(order, u, v)
        orders.append(new_order)
        new_matrix = order_by(matrix, new_order)
        MU_tssa.append(cost_fun(new_matrix))
    MU_tssa = np.asarray(MU_tssa)
    min_tssa = np.min(MU_tssa)
    indices = np.where(MU_tssa == min_tssa)[0]
    index = np.random.choice(indices)
    return orders[index]


def move_node_n1(matrix, order, u, cost_fun):
    n = matrix.shape[0]
    median_u = median(matrix, order, u)

    # calculate possible changes and its tssa
    left = max(median_u - 2, 0)
    right = min(median_u + 2 + 1, n)
    v_range = order[np.arange(left, right)]
    order = find_best_order(matrix, order, u, v_range, cost_fun)
    return order


def move_node_n2(matrix, order, u, cost_fun):
    n = matrix.shape[0]
    neighbors_idx = np.arange(n)[matrix[u, :] > 0]
    neighbors_idx = neighbors_idx[neighbors_idx != u]
    if neighbors_idx.shape[0] == 0:
        return order
    # calculate possible changes and its tssa
    order = find_best_order(matrix, order, u, neighbors_idx, cost_fun)
    return order


def move_node_n3(matrix, order, node, p, cost_fun):
    if p > random.random():
        result = move_node_n1(matrix, order, node, cost_fun)
        return result, 'n1'
    else:
        result = move_node_n2(matrix, order, node, cost_fun)
        return result, 'n2'


def temperature(C_inf, Sig_inf, tssa):
    r = 50000
    p = 1 - abs(r) ** (-1)
    p = 0.5 + p / 2
    gamma_inf = st.norm.ppf(p, loc=C_inf, scale=Sig_inf)
    gamma_inf = (gamma_inf - C_inf) / Sig_inf
    return Sig_inf**2 / (C_inf - tssa - Sig_inf * gamma_inf)


def accept_probability(cost, new_cost, T):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(-(new_cost - cost) / T)
        return p


def next_temperature(T, sigma, delta):
    return T * (1 + (math.log(1 + delta) * T) / (3 * sigma))**(-1)


def annealing(matrix, order, T0, cost_fun, maxsteps=1000, delta=.1, p=.9):
    state = order
    ordered_matrix = order_by(matrix, state)
    cost = cost_fun(ordered_matrix)
    states, costs = [state], [cost]
    T = T0
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        node = random.randint(0, matrix.shape[0]-1)
        new_state, _ = move_node_n3(matrix, state, node, p, cost_fun)
        new_matrix = order_by(matrix, new_state)
        new_cost = cost_fun(new_matrix)
        if accept_probability(cost, new_cost, T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
        # sigma_temp = 1
        sigma_temp = np.std(costs)
        T = next_temperature(T, sigma_temp, delta)
    new_matrix = order_by(matrix, new_state)
    return state, cost_fun(new_matrix), states, costs


def MinLA(A, cost_fun=TSSA):
    # initial placement
    initial_order = fim(A)

    # calculate initial evaluation
    ordered_A = order_by(A, initial_order)
    initial_tssa = TSSA(ordered_A)

    # initial temperature determination
    C_inf, Sig_inf = generate_independent_solutions(
        A, initial_order, cost_fun, 10 ** 3)
    T0 = temperature(C_inf, Sig_inf, initial_tssa)
    output = annealing(A, initial_order, T0, cost_fun, maxsteps=1000)

    order = np.asarray(output[0], dtype=np.int).tolist()
    matrix = order_by(A, order)
    return matrix, order
