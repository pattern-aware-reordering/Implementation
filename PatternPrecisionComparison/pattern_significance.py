import csv
import pandas as pd
import numpy as np
import scikit_posthocs as sp

if __name__ == '__main__':

    def get_key(probabilities):
        return "-".join([str(p) for p in probabilities])

    def split_key(key):
        return key.split('-')

    def get_item(results, alg, key):
        results[alg] = results[alg] if alg in results else {}
        results[alg][key] = results[alg][key] if key in results[alg] else {
            'encoding': [], 'label': []}
        return results[alg][key]

    file = open('./probabilities.csv', 'r', newline="")
    reader = csv.DictReader(file)
    params = reader.fieldnames
    file.close()


    log_file = open('./precision.csv', 'r')
    reader = csv.DictReader(log_file, delimiter=',')

    results = {}
    for line in reader:
        dic = dict(line)
        alg = dic['ALG']
        probabilities = [dic[param] for param in params]
        encoding_precision = float(dic['encoding_precision'])
        label_precision = float(dic['label_precision'])

        key = get_key(probabilities)
        item = get_item(results, alg, key)
        item['encoding'].append(encoding_precision)
        item['label'].append(label_precision)

    for alg in results:
        for key in results[alg]:
            results[alg][key] = {
                'encoding': np.mean(results[alg][key]['encoding']),
                'label': np.mean(results[alg][key]['label'])
            }

    file = open('./precision-avg.csv', 'w')
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(["ALG", "subgraph_noise", "connecting_p",
                     "edge_noise", "precision"])
    lists = {}
    csv_lists = []
    for alg in results:
        lists[alg] = []
        for key in results[alg]:
            precision = results[alg][key]['label'] * \
                results[alg][key]['encoding']
            ps = split_key(key)
            row = [alg] + ps + [precision]
            writer.writerow(row)
            lists[alg].append(precision)

    file.close()

    data = {"value": [], "method": []}
    for key in results:
        for value in lists[key]:
            data['value'].append(value)
            data['method'].append(key)
    df = pd.DataFrame(data)
    table = sp.posthoc_conover(df, val_col='value',
                               group_col='method', p_adjust='holm')
    keys = list(results.keys())
    ours_keys = list(filter(lambda k: k[0:3] == 'Our', keys))
    VoG_key = list(filter(lambda k: k[0:3] == 'VoG', keys))[0]
    for ours_key in ours_keys:
        p = table[VoG_key][ours_key]

        print('Conover', p)
        print('mean: ', ours_key + '(', np.mean(
            lists[ours_key]), ')', VoG_key+'(', np.mean(lists[VoG_key]), ')')
        print('std: ', ours_key + '(', np.std(
            lists[ours_key]), ')', VoG_key + '(', np.std(lists[VoG_key]), ')')
