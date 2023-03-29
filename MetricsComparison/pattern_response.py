import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from MatrixReordering.Matrix import order_by
from MatrixReordering.Metrics import BW, PR, LA, MI
from PatternPrecisionComparison.synthetic_generator import synthetic_generator
from DEFINITIONS import S, C, BC, CH, E
from Utils.IO import write_matrix


def matrixize(G):
    return np.asarray(nx.adjacency_matrix(G).todense())


def generate_block_diagonal():
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    min_maxs = [[0, 3], [4, 6], [7, 8], [9, 9]]
    for (min, max) in min_maxs:
        for i in range(min, max + 1):
            for j in range(i, max + 1):
                G.add_edge(i, j)
    return matrixize(G)


def generate_star():
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    for i in range(10):
        G.add_edge(3, i)
    return matrixize(G)


def generate_off_diagonal_block():
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    for i in range(0, 4):
        for j in range(6, 10):
            G.add_edge(i, j)
    return matrixize(G)


def generate_bands():
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    for i in range(2, 10):
        G.add_edge(i - 2, i)
    return matrixize(G)


def generate_anti():
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    edges = [[0, 3], [0, 5], [0, 6], [1, 9], [
        2, 4], [2, 6], [3, 7], [5, 9], [7, 9], [1, 1], [4, 4], [6, 6], [7, 7]]
    G.add_edges_from(edges)
    return matrixize(G)


def generate_bandwidth_anti():
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    edges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6],
             [3, 7], [4, 7], [5, 8], [6, 8], [7, 9], [8, 9], [0, 0], [9, 9]]
    G.add_edges_from(edges)
    return matrixize(G)


def generate_random():
    G = nx.gnp_random_graph(10, 0.4)
    G.remove_edges_from(nx.selfloop_edges(G))
    return matrixize(G)


if __name__ == "__main__":
    generation_funcs = {
        "block": generate_block_diagonal,
        "star": generate_star,
        "off_diagonal_block": generate_off_diagonal_block,
        "bands": generate_bands,
        "anti": generate_anti,
        "bandwidth_anti": generate_bandwidth_anti
    }
    metric_funcs = {
        "MI": lambda m: MI(m, scaled=True, isUndirected=False),
        "LA": lambda m: LA(m, scaled=True),
        "PR": lambda m: PR(m, scaled=True),
        "BW": lambda m: BW(m, scaled=True)
    }
    patterns = list(generation_funcs.keys())
    metrics = list(metric_funcs.keys())
    # for i in range(10):
    #     patterns.append("random")
    for pattern in patterns:
        matrix = generation_funcs[pattern]()
        name = "origin"
        log = {
            "nodes": list(range(matrix.shape[0])),
            "name": name
        }
        print(pattern, end=": ")
        for metric in metrics:
            cost = metric_funcs[metric](matrix)
            log[metric] = cost
            print(metric, "("+str(cost)+")", end=", ")
        write_matrix('./pattern/' + pattern + '-' + name +
                     '.matrix.json', matrix, log)
        print()
