import csv
import time
import networkx as nx
import numpy as np
from MDL import compute_cost
from MatrixReordering.ChengChunch import biclustering
from MatrixReordering.MDS import MDS
from MatrixReordering.Matrix import cal_pairwise_moran_dist
from MatrixReordering.OLO import optimal_leaf_ordering
from MatrixReordering.RankTwo import rank_two
from MatrixReordering.MinLA import MinLA
from MatrixReordering.evolutionary import evolutionary_reorder
from MatrixReordering.PatternOrdering import greedy_ordering, randomized_ordering
from MetricsComparison.comparison import run_comparison
from MetricsComparison.definitions import DATASETS, METRIC_FUNCS
from Utils.IO import read_graph, write_matrix


def compute_metrics(A, default=None):
    metric_values = {}
    for metric in METRIC_FUNCS:
        if type(default) is dict and metric in default:
            metric_values[metric] = default[metric]
        else:
            metric_values[metric] = METRIC_FUNCS[metric](A)
    return metric_values


def log_metrics(A, name, nodes, times, dataset, metrics=None, clustering=None):
    log = {
        "name": name,
        "nodes": nodes,
        "times": times,
    }

    metric_values = compute_metrics(A, default=metrics)
    for metric in metric_values:
        log[metric] = metric_values[metric]
        print(metric+': ' + ("{:.4f}".format(log[metric])), end='\t')
    print(name)

    if clustering:
        log['clustering'] = clustering
    write_matrix('./matrix/' + dataset + '-' + name +
                 '.matrix.json', A, log)
    return log


def compute_reordering(A, nodes, reordering, **args):
    A, order = reordering(A.copy(), **args)
    nodes_prime = [nodes[i] for i in order]
    return A, nodes_prime


def print_cost(G, summarization):
    origin_cost = len(G.edges)
    super_node_cost, super_link_cost, correction_cost = compute_cost(
        summarization)
    cost = super_node_cost + super_link_cost + correction_cost
    print()
    print("COST:\t", str(super_node_cost) + '+' + str(super_link_cost) +
          '+' + str(correction_cost) + '=' + str(cost))
    print("SAVE:\t", str(origin_cost) + '-' +
          str(cost) + '=' + str(origin_cost - cost))


def write_log(log, keys, path='../logs/log'):
    log_file = open(path, 'a', newline='')
    writer = csv.writer(log_file, delimiter='\t')
    items = log.items()
    items = filter(lambda item: item[0] in keys, items)
    row = [item[1]
           for item in sorted(items, key=lambda item: keys.index(item[0]))]
    writer.writerow(row)
    log_file.close()


if __name__ == '__main__':
    datasets = DATASETS
    for dataset in datasets:
        filetype = 'edgelist'

        print(dataset)

        G = read_graph('../data/' + dataset + '.' + filetype)
        A = np.asarray(nx.adjacency_matrix(G).todense())
        H = G.copy()
        nodes = list(G.nodes)

        import datetime

        log_file = open('../logs/log', 'a', newline='')
        log_file.write('\n')
        log_file.write('time:\t' + str(datetime.datetime.now()) + '\n')
        log_file.write('data:\t' + dataset + '\n')
        keys = ["MI", "LA", "PR", "BW", "name", "nodes"]
        writer = csv.writer(log_file, delimiter='\t')
        writer.writerow(keys)
        log_file.close()

        logs = []

        # mdl optimal leaf ordering
        A_prime, nodes_prime, summarization, times = greedy_ordering(
            A.copy())
        nodes_prime = [nodes[i] for i in nodes_prime]
        name = greedy_ordering.__name__
        log = log_metrics(A_prime, name, nodes_prime, times, dataset)
        write_log(log, keys)

        # randomized mdl optimal leaf ordering
        A_prime, nodes_prime, _summarization, times = randomized_ordering(
            A.copy())
        nodes_prime = [nodes[i] for i in nodes_prime]
        name = randomized_ordering.__name__
        log = log_metrics(A_prime, name, nodes_prime, times,
                          dataset)
        write_log(log, keys)

        reordering_functions = [
            {"func": MinLA, "args": {}, "postfix": ""},  # MLA
            {"func": MDS, "args": {}, "postfix": ""},
            {"func": optimal_leaf_ordering, "args": {}, "postfix": ""},  # OLO
            {"func": optimal_leaf_ordering, "args": {
                "dist": cal_pairwise_moran_dist}, "postfix": "-delta"},  # OLO_delta
            {"func": evolutionary_reorder, "args": {}, "postfix": ""},  # EVO
            {"func": rank_two, "args": {}, "postfix": ""},  # RT
            {"func": biclustering, "args": {}, "postfix": ""},  # BC
        ]
        
        # origin techniques
        for config in reordering_functions:
            start_time = time.time()
            A_prime, nodes_prime = compute_reordering(
                A, nodes=nodes, reordering=config["func"], **config["args"])
            run_time = time.time() - start_time
            times = {
                "ordering": run_time,
                "total": run_time
            }
            name = config["func"].__name__ + config["postfix"]
            log = log_metrics(A_prime, name, nodes_prime,
                              times, dataset)
            write_log(log, keys)

    run_comparison()
