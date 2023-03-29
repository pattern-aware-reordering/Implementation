#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import csv

import numpy as np
import pandas as pd
import scipy.stats as st
import scikit_posthocs as sp

# read results
names = ["1-1661934866373", "3-1661999533002", "5-1662002195049", "9-1662010258606", "0-1661933688574",
         "11-1662013922332", "2-1661935567051", "6-1662003410816", "10-1662013153023", "4-1662000612817",
         "8-1662009390411", "7-1662008650057", "12-1662090291751", "13-1662121289542", "14-1662122943293"]

file = open('./probabilities.csv', 'r', newline="")
reader = csv.DictReader(file)
probability_names = reader.fieldnames
file.close()

datasets = None
algorithms = None
statistics = None

patterns = ["clique", "bipartite core", "star", "chain", "none"]

for name in names:
    result_path = "./results/" + name + ".json"
    file = open(result_path, 'r')
    result = json.load(file)
    file.close()

    if statistics is None:
        datasets = list({task["dataset"] for task in result["tasks"]})
        algorithms = list({task["algorithm"].replace('未', 'δ') for task in result["tasks"]})
        statistics = {algorithm: {dataset: {} for dataset in datasets}
                      for algorithm in algorithms}

    i = 0
    for task in result["tasks"]:
        dataset = task["dataset"]
        algorithm = task["algorithm"].replace('未', 'δ')
        probabilities = dataset.split("-")[-3:]

        ##### read orders #####
        matrix_path = "./matrix/" + dataset + '-' + algorithm + ".matrix.json"
        matrix_file = open(matrix_path, "r")
        matrix_data = json.load(matrix_file)
        order = matrix_data["information"]["nodes"]
        ##### read orders #####

        ##### read graph data #####
        graph_path = "./graph/" + dataset + ".json"
        graph_file = open(graph_path, "r")
        graph_data = json.load(graph_file)
        # truth label
        nodes = graph_data["nodes"]
        id2node = {node["id"]: node for node in nodes}
        edges = graph_data["links"]
        edge_truth_label = {}
        for edge in edges:
            source = id2node[edge["source"]]
            target = id2node[edge["target"]]
            source_id = source["id"]
            target_id = target["id"]
            key1 = str(source_id) + '-' + str(target_id)
            key2 = str(target_id) + '-' + str(source_id)
            if source["label"] == target["label"]:
                # same pattern
                pattern = source["encoding"].replace("_", " ")
            else:
                pattern = "none"
            edge_truth_label[key1] = edge_truth_label[key2] = pattern

        ##### read graph data #####

        ##### read user label #####
        edge_user_label = {}
        for item in task["label"]:
            cells = item["cells"]
            pattern = item["label"]
            for cell in cells:
                source_id = order[cell[0]]
                target_id = order[cell[1]]
                key1 = str(source_id) + '-' + str(target_id)
                key2 = str(target_id) + '-' + str(source_id)
                edge_user_label[key1] = edge_user_label[key2] = pattern
        ##### read user label #####

        label_amount = {pattern: 0 for pattern in patterns}
        for key in edge_user_label:
            label_amount[edge_user_label[key]] += 1
        label_proportion = {pattern: label_amount[pattern]/len(edge_user_label) for pattern in label_amount}

        time = result["times"][i]
        time_per_label = time / len(task["label"])

        # https://zhuanlan.zhihu.com/p/147663370
        tp = {}
        fn = {}
        fp = {}
        for pattern in patterns:
            tp[pattern] = 0
            fn[pattern] = 0
            fp[pattern] = 0

        for key in edge_truth_label:
            user_label = "none"
            if key in edge_user_label:
                user_label = edge_user_label[key]
            truth_label = edge_truth_label[key]
            # tp
            if user_label == truth_label:
                tp[user_label] += 1
            # fn and fp
            if user_label != truth_label:
                fn[user_label] += 1
                fp[truth_label] += 1

        tp_sum = np.sum([tp[pattern] for pattern in patterns])
        fp_sum = np.sum([fp[pattern] for pattern in patterns])
        fn_sum = np.sum([fn[pattern] for pattern in patterns])
        accuracy = tp_sum / (tp_sum + fp_sum)
        recall = tp_sum / (tp_sum + fn_sum)

        statistic = {
            # "label-amount": len(edge_user_label) / 2,
            # "precision": precision,
            # "recall": recall,
            # "F1": F1_score,
            "accuracy": accuracy,
            "time": time,
            "time-per-pattern": time_per_label,
            # "tp": tp,
            # "fn": fn
        }

        for pattern in patterns:
            if tp[pattern] == 0:
                precision = recall = 0
            else:
                precision = tp[pattern] / (tp[pattern] + fp[pattern])
                recall = tp[pattern] / (tp[pattern] + fn[pattern])
            statistic[pattern + '-precision'] = precision
            statistic[pattern + '-recall'] = recall

        statistics[algorithm][dataset][name] = statistic

        i += 1

stat_names = statistics[algorithms[0]][datasets[0]][names[0]].keys()

aggregated_data = {
    "summary": {
        algorithm: {
            "data": {stat_name: [] for stat_name in stat_names},
            "ci": {stat_name: [] for stat_name in stat_names}
        } for algorithm in algorithms
    }
}

for algorithm in algorithms:
    for stat_name in stat_names:
        values = []
        for dataset in datasets:
            for name in names:
                value = statistics[algorithm][dataset][name][stat_name]
                [stat_name].append(value)
                values.append(value)
    
        ci = st.t.interval(alpha=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))
        aggregated_data["summary"][algorithm]["data"][stat_name] = values
        aggregated_data["summary"][algorithm]["ci"][stat_name] = ci

# compare different algorithms
data = {
    "name": [],
    "dataset": [],
    "algorithm": []
}

for stat_name in stat_names:
    data[stat_name] = []

for algorithm in algorithms:
    for dataset in datasets:
        for name in names:
            data["name"].append(name)
            data["dataset"].append(dataset)
            data["algorithm"].append(algorithm)
            for stat_name in stat_names:
                value = statistics[algorithm][dataset][name][stat_name]
                data[stat_name].append(value)
df = pd.DataFrame(data)
df.to_csv("./results.csv")

aggregated_data["significance"] = {
    stat_name: {
        algorithm: {
            algorithm: None for algorithm in algorithms
        } for algorithm in algorithms
    } for stat_name in stat_names
}

for stat_name in stat_names:
    print(stat_name)
    table = sp.posthoc_conover(df, val_col=stat_name,
                               group_col='algorithm', p_adjust='holm')
    values = {algorithm: df[df.algorithm == algorithm][stat_name].tolist() for algorithm in algorithms}
    means = {algorithm: np.mean(values[algorithm]) for algorithm in algorithms}
    stds = {algorithm: np.std(values[algorithm]) for algorithm in algorithms}

    n = len(algorithms)
    for i in range(n):
        alg1 = algorithms[i]
        for j in range(n):
            alg2 = algorithms[j]
            p = table[alg1][alg2]
            if means[alg1] < means[alg2]:
                p = -p
            aggregated_data["significance"][stat_name][alg1][alg2] = p
            print('Conover: ' + alg1 + "(" + "{:.2f}".format(means[alg1]) + ")" + '-' + alg2 + "(" + "{:.2f}".format(means[alg2]) + ")", p)

filepath = "./results.json"
file = open(filepath, 'w')
json.dump(aggregated_data, file)
file.close()