import csv
import json
import pandas as pd
import numpy as np
import scipy.stats as st
import scikit_posthocs as sp

from MetricsComparison.definitions import DATASETS, METRIC_FUNCS


def run_comparison(recompute=None):
    datasets = DATASETS
    baselines = ["MinLA", "biclustering", "evolutionary_reorder", "optimal_leaf_ordering", "MDS", "rank_two",
                 "optimal_leaf_ordering-delta"]
    ours = ['greedy_ordering',
            "randomized_ordering"]
    algorithms = baselines + ours
    metrics = ["MI", "LA", "PR", "BW"]

    metric_summary = {}
    times = {}
    for algorithm in algorithms:
        print(algorithm)
        metric_summary[algorithm] = {}
        times[algorithm] = {}
        data = {}
        for metric in metrics:
            data[metric] = []
        for dataset in datasets:
            print(dataset, end=", ")
            filepath = "./matrix/" + dataset
            filepath += "-" + algorithm + ".matrix.json"
            file = open(filepath)
            info = json.load(file)
            for metric in metrics:
                if metric in info["information"] and (not recompute or metric not in recompute):
                    value = info["information"][metric]
                else:
                    value = METRIC_FUNCS[metric](np.asarray(info["matrix"]))
                data[metric].append(float(value))
            file.close()
            matrix = np.asarray(info["matrix"])
            times[algorithm][dataset] = {
                "nodes": matrix.shape[0],
                "links": np.sum(matrix) / 2,
                "times": info["information"]["times"]
            }
        metric_summary[algorithm]['data'] = data

        cis = {}
        for metric in metrics:
            cis[metric] = st.t.interval(alpha=0.95, df=len(
                data[metric])-1, loc=np.mean(data[metric]), scale=st.sem(data[metric]))
        metric_summary[algorithm]['ci'] = cis
        print()

    results = {
        "summary": metric_summary,
        "times": times,
        "significance": {metric: {algorithm: {algorithm: {} for algorithm in algorithms} for algorithm in algorithms} for metric in metrics}
    }

    for metric in metrics:
        for baseline in algorithms:
            baseline_data = metric_summary[baseline]["data"][metric]
            for our in ours:
                our_data = metric_summary[our]["data"][metric]
                if baseline == our:
                    p = 1.0
                else:
                    data = {"value": [], "method": []}
                    for datum in baseline_data:
                        data['value'].append(datum)
                        data['method'].append('baseline')
                    for datum in our_data:
                        data['value'].append(datum)
                        data['method'].append('our')
                    df = pd.DataFrame(data)
                    table = sp.posthoc_conover(df, val_col='value',
                                               group_col='method', p_adjust='holm')
                    p = table['baseline']['our']
                    origin_mean = np.mean(baseline_data)
                    two_level_mean = np.mean(our_data)
                    if two_level_mean < origin_mean:
                        p *= -1
                results["significance"][metric][our][baseline] = p
                results["significance"][metric][baseline][our] = -p

    filepath = "./results.json"
    file = open(filepath, 'w')
    json.dump(results, file)
    file.close()


if __name__ == "__main__":
    run_comparison()
