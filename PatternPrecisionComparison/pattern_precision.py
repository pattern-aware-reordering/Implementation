import csv
import json

import networkx as nx

from DEFINITIONS import E, S, C, CH, BC
from MDL import greedy, filter_super_nodes, randomized, origin_greedy
from Utils.IO import write_graph, read_graph
from PatternPrecisionComparison.synthetic_generator import get_path


def ensemble(G, nodes):
    label_count = {}
    for node in nodes:
        label = G.nodes[node]['label']
        label_count[label] = label_count[label] if label in label_count else 0
        label_count[label] = label_count[label] + 1

    max_label = G.nodes[node[0]]['label']
    max_count = 0
    for label, count in label_count.items():
        if count > max_count:
            max_count = count
            max_label = label

    return max_label


def test_mdl(G, alg=greedy, use_node_cost=True):
    summarization = alg(G, use_node_cost=use_node_cost)
    supernodes = filter_super_nodes(summarization)

    print()

    node_to_supernode = {}

    for supernode in supernodes:
        nodes = supernode['nodes']
        label = ensemble(G, nodes)
        supernode['label'] = label
        for node in nodes:
            node_to_supernode[node] = supernode

    label_TP = 0
    encoding_TP = 0
    TP = 0
    for node in G.nodes:
        truth_label = G.nodes[node]['label']
        truth_encoding = G.nodes[node]['encoding']
        encoding = node_to_supernode[node]['encoding'] if node in node_to_supernode else E
        label = node_to_supernode[node]['label'] if node in node_to_supernode else -1
        if label == truth_label:  # correctly grouped
            label_TP += 1
        if encoding == truth_encoding:  # correctly encoded as some structure
            encoding_TP += 1
        if label == truth_label and encoding == truth_encoding:
            TP += 1

    encoding_precision = encoding_TP / len(G.nodes)
    label_precision = label_TP / len(G.nodes)
    precision = TP / len(G.nodes)
    return encoding_precision, label_precision, precision


def test_vog(G, path):
    file = open(path, 'r')
    encoding_map = {'st': S, 'nb': BC, 'bc': BC, 'fc': C, 'nc': C, 'ch': CH}

    while True:
        line = file.readline()
        if not line and len(line) <= 0:
            break
        encoding = encoding_map[line[0:2]]
        nodes = line[3:].replace(',', '').replace('\n', '').split(' ')
        nodes = [str(int(node) - 1) for node in nodes]
        label = ensemble(G, nodes)
        for node in nodes:
            truth_label = G.nodes[node]['label']
            truth_encoding = G.nodes[node]['encoding']
            G.nodes[node]['right_labeled'] = G.nodes[node]['right_labeled'] if 'right_labeled' in G.nodes[node] else False
            G.nodes[node]['right_labeled'] |= label == truth_label
            G.nodes[node]['right_encoded'] = G.nodes[node]['right_encoded'] if 'right_encoded' in G.nodes[node] else False
            G.nodes[node]['right_encoded'] |= encoding == truth_encoding

    label_TP = 0
    encoding_TP = 0
    TP = 0
    for node in G.nodes:
        right_labeled = G.nodes[node]['right_labeled'] if 'right_labeled' in G.nodes[node] else False
        right_encoded = G.nodes[node]['right_encoded'] if 'right_encoded' in G.nodes[node] else G.nodes[node]['encoding'] == E
        if right_labeled:
            label_TP += 1
        if right_encoded:
            encoding_TP += 1
        if right_labeled and right_encoded:
            TP += 1

    encoding_precision = encoding_TP / len(G.nodes)
    label_precision = label_TP / len(G.nodes)
    precision = TP / len(G.nodes)
    return encoding_precision, label_precision, precision


if __name__ == '__main__':
    dataset = 'synthetic'
    i = 0
    log_path = 'precision.csv'
    log_file = open(log_path, 'w', newline='')
    writer = csv.writer(log_file, delimiter=',')
    header = ["ALG"]
    file = open('./probabilities.csv', 'r', newline="")
    reader = csv.DictReader(file)
    params = reader.fieldnames
    header = header + params + ["encoding_precision", "label_precision", "precision"]
    writer.writerow(header)
    log_file.close()

    i = 0
    log_file = open(log_path, 'a', newline='')
    writer = csv.writer(log_file, delimiter=',')
    for line in reader:
        dic = dict(line)
        settings = {param: dic[param] for param in params}
        print('--------')
        print(i)
        print('--------')
        i += 1
        path = get_path({
            "path": "./data",
            "dataset_name": dataset,
            "params": params,
            **settings,
        }) + '.json'
        G = read_graph(path)

        encoding_precision, label_precision, precision = test_mdl(
            G.copy(), alg=greedy, use_node_cost=True)
        writer.writerow(['Ours (GRD)'] + [dic[param] for param in params] + [encoding_precision, label_precision, precision])
        
        encoding_precision, label_precision, precision = test_mdl(
            G.copy(), alg=randomized, use_node_cost=True)
        writer.writerow(['Ours (RDM)'] + [dic[param] for param in params] + [encoding_precision, label_precision, precision])

        encoding_precision, label_precision, precision = test_vog(G.copy(), path=get_path({
            "path": "./VoG/DATA",
            "dataset_name": dataset,
            "params": params,
            **settings,
        }) + '_orderedALL.model')
        writer.writerow(['VoG'] + [dic[param] for param in params] + [encoding_precision, label_precision, precision])

    log_file.close()
    file.close()
