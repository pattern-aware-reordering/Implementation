import csv
import numpy as np
import networkx as nx

from DEFINITIONS import S, C, BC, CH, E
from MatrixReordering.PatternOrdering import greedy_ordering, randomized_ordering
from MatrixReordering.Matrix import cal_pairwise_moran_dist
from MatrixReordering.MinLA import MinLA
from MatrixReordering.OLO import optimal_leaf_ordering
from PatternPrecisionComparison.synthetic_generator import get_path, synthetic_generator
from Utils.IO import write_matrix, write_graph


if __name__ == '__main__':
    # add structures
    # connect them
    # add noise
    dataset_name = "synthetic"
    graph_path = "./graph"
    matrix_path = "./matrix"

    #### generate ####
    num = 1
    node_range = (8, 9)
    struct_num = {S: num, C: num, BC: num, CH: num, E: num}

    probabilities = []
    params = ["p_ni", "p_nb", "p_c"]
    
    # improving within pattern probability
    settings = {
        "p_ni": (0, 10),
        "p_nb": (80, 100),
        "p_c": (0, 10)
    }
    for i in range(5):
        probabilities.append([np.random.randint(settings[name][0], settings[name][1]) for name in params])

    file = open('./probabilities.csv', 'w', newline="")
    writer = csv.writer(file)
    writer.writerow(params)
    writer.writerows(probabilities)

    for settings in probabilities:
        config = {
            'struct_num': struct_num,
            'node_range': node_range,
            'dataset_name': dataset_name,
            'params': params,
        }
        for param in params:
            config[param] = settings[params.index(param)]

        G = synthetic_generator(config)
        json_path = get_path({
            **config,
            'path': graph_path
        }) + ".json"
        write_graph(G, json_path)
        nodes = list(G.nodes)
        A = np.asarray(nx.adjacency_matrix(G).todense())
        algorithms = {
            "origin": lambda A: (A.copy(), nodes),
            "MinLA": lambda A: MinLA(A.copy()),
            "LeafOrder-δI": lambda A: optimal_leaf_ordering(A.copy(), dist=cal_pairwise_moran_dist),
            "Ours (GRD)": lambda A: greedy_ordering(A.copy())[0:2],
            "Ours (RDM)": lambda A: randomized_ordering(A.copy())[0:2]
        }
        for algorithm in algorithms:
            print(algorithm)
            matrix, order = algorithms[algorithm](A)
            path = get_path({
                **config,
                'path': matrix_path
            }) + '-' + algorithm + '.matrix.json'
            write_matrix(path, matrix, {"nodes": order})
