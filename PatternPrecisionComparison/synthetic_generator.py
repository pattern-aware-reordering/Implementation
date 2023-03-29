import csv
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

from DEFINITIONS import S, C, BC, CH, E
from Utils.IO import write_graph


def star(n):
    return nx.star_graph(n - 1)


def bipartite_core(n):
    G = bipartite.random_graph(int(n / 2), n - int(n / 2), 1)
    return G


def empty_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G


def relabel(G, begin):
    mapping = {}
    i = 0
    for node in G.nodes:
        mapping[node] = i + begin
        i += 1
    G = nx.relabel_nodes(G, mapping)
    return G


def synthetic_generator(configs):
    struct_num = configs["struct_num"]
    node_range = configs["node_range"]
    p_ni = configs["p_ni"]
    p_nb = configs["p_nb"]
    p_c = configs["p_c"]
    generators = {S: star, C: nx.complete_graph,
                  BC: bipartite_core, E: empty_graph, CH: nx.path_graph}
    i = 0
    subgraphs = []
    for struct in struct_num:
        num = struct_num[struct]
        for j in range(num):
            n = np.random.randint(
                node_range[0], node_range[1])
            subgraph = generators[struct](n)
            edges_to_be_removed = []
            edges_to_be_added = []
            for x in range(n):
                for y in range(x):
                    if np.random.randint(100) < p_ni:
                        if subgraph.has_edge(x, y):
                            edges_to_be_removed.append((x, y))
                        else:
                            edges_to_be_added.append((x, y))

            subgraph.remove_edges_from(edges_to_be_removed)
            subgraph.add_edges_from(edges_to_be_added)
            subgraph = relabel(subgraph, i)
            for sub_node in subgraph.nodes:
                subgraph.nodes[sub_node]['encoding'] = struct
                subgraph.nodes[sub_node]['label'] = len(subgraphs)
            subgraphs.append(subgraph)
            i += len(subgraph.nodes)

    G = nx.Graph()
    i = 0
    num = len(subgraphs)
    for subgraph in subgraphs:
        G = nx.compose(G, subgraph)

    while i < num:
        j = i + 1
        while j < num:
            if np.random.randint(100) < p_c:  # connection 
                for u in subgraphs[i].nodes:
                    for v in subgraphs[j].nodes:
                        if np.random.randint(100) >= p_nb:
                            G.add_edge(u, v)
            j += 1
        i += 1

    return G


def get_path(config):
    path = config['path'] + '/' + config['dataset_name'] + '-' + ('-'.join([str(config[param]) for param in config["params"]]))
    return path


def write_synthetic_graph(config):
    struct_num = config['struct_num']
    G = synthetic_generator(config)
    mapping = {node: str(node) for node in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    path = get_path(config=config)
    json_path = path + ".json"
    edgelist_path = path + '.edgelist'
    vog_out_path = path + '.out'  # for vog input
    write_graph(G, json_path)
    nx.write_edgelist(G, edgelist_path)
    nx.set_edge_attributes(G, 1, "weight")
    mapping = {node: str(int(node) + 1) for node in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    nx.write_edgelist(G, vog_out_path, data=["weight"])

    return G


if __name__ == '__main__':
    # add structures
    # connect them
    # add noise
    dataset_name = "synthetic"
    path = "./data/"

    #### generate ####
    num = 5
    node_range = (5, 10)
    struct_num = {S: num, C: num, BC: num, CH: num, E: num}

    probabilities = []
    params = ["p_ni", "p_nb", "p_c"]
    
    # improving within pattern probability
    settings = {
        "p_ni": 0,
        "p_nb": 25,
        "p_c": 25
    }
    while settings["p_ni"] <= 50:
        probabilities.append([settings[name] for name in params])
        settings["p_ni"] += 5
    
    # improving between pattern probability
    settings = {
        "p_ni": 25,
        "p_nb": 0,
        "p_c": 25
    }
    while settings["p_nb"] <= 50:
        probabilities.append([settings[name] for name in params])
        settings["p_nb"] += 5

    # improving numbers of nodes in patterns
    settings = {
        "p_ni": 25,
        "p_nb": 25,
        "p_c": 0
    }
    while settings["p_c"] <= 50:
        probabilities.append([settings[name] for name in params])
        settings["p_c"] += 5

    file = open('./probabilities.csv', 'w', newline="")
    writer = csv.writer(file)
    writer.writerow(params)
    writer.writerows(probabilities)

    for settings in probabilities:
        config = {
            'struct_num': struct_num,
            'node_range': node_range,
            'path': path,
            'dataset_name': dataset_name,
            'params': params
        }
        for param in params:
            config[param] = settings[params.index(param)]
        
        write_synthetic_graph(config)
