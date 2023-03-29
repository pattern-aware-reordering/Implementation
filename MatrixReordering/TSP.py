# travelling saleperson
import mlrose
import numpy as np
from MatrixReordering.Matrix import order_by, cal_pairwise_dist, order_to_permutation

eps = 1e-5


def travelling_saleperson(matrix, dist_func=cal_pairwise_dist):
    dist = dist_func(matrix) + eps
    n = matrix.shape[0]
    dist_list = filter(lambda tuple: tuple[0] != tuple[1], [(i, j, dist[i, j]) for i in range(n) for j in range(n)])
    fitness_dists = mlrose.TravellingSales(distances=dist_list)

    problem_fit = mlrose.TSPOpt(length=n, fitness_fn=fitness_dists, maximize=False)
    # best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob=0.2, max_attempts=100, random_state=2)
    order, best_fitness = mlrose.mimic(problem_fit, max_attempts=20)
    # more problem solver: https://mlrose.readthedocs.io/en/stable/source/algorithms.html
    # permutation = order_to_permutation(order)

    matrix = order_by(matrix, order)
    return matrix, order


