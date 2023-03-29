import json
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


def write_graph(G, path):
    node_link_graph = nx.node_link_data(G)
    file = open(path, 'w')
    json.dump(node_link_graph, file)
    file.close()


def read_graph(path):
    def read_json_file(filename):
        with open(filename) as f:
            js_graph = json.load(f)
        return json_graph.node_link_graph(js_graph)

    filetype = path.split('.')[-1]

    if filetype == 'json':
        G = read_json_file(path)
    # elif filetype == 'edgelist':
    else:
        G = nx.read_edgelist(path, data=False)

    # remove self loops
    selfloops = list(nx.selfloop_edges(G))
    G.remove_edges_from(selfloops)
    print(str(len(list(selfloops))) + ' selfloops removed.')

    mapping = {node: str(node) for node in G.nodes}
    # node_list = list(G.nodes)
    # mapping = {node_list[i]: i for i in range(len(node_list))}
    G = nx.relabel_nodes(G, mapping)

    return G


def write_matrix(path, matrix, log):
    file = open(path, 'w')
    obj = {
        "matrix": matrix.tolist(),
        "information": log
    }
    json.dump(obj, file)
    file.close()


def read_matrix(path):
    file = open(path, 'r')
    obj = json.load(file)
    matrix = np.array(obj["matrix"])
    log = obj["information"]
    file.close()
    return matrix, log
