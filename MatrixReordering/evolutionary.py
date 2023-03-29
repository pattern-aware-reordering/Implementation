import numpy as np

from MatrixReordering.Metrics import LA, PR, BW, DIS
from MatrixReordering.Matrix import order_by, permutation_to_order

MUTATION_PROBABILITY = 0.5
INITIAL_POPULATION = 20
GENERATIONS = 200
STOP_THRESHOLD = 30


def mutation(permutations, prob=MUTATION_PROBABILITY):
    """
    :param permutations: [[1,2,3,..], [3,2,1,...], ...]
    :param prob: mutation probability
    :return:
    """
    population = permutations.shape[0]  # number of permutations
    n = permutations.shape[1]  # matrix length
    idx_of_mutate = np.nonzero(np.random.binomial(1, prob, population))[0]
    for i in idx_of_mutate:
        b1, b2 = np.sort(np.random.randint(n, size=2))
        permutations[i][b1:b2] = np.flip(permutations[i][b1:b2])
    return permutations


def compute_fitnesses(matrix, permutations, fitness):
    """
    :param matrix:
    :param permutations:
    :param fitness: function
    :return:
    """
    population = permutations.shape[0]
    fitnesses = np.array([[fitness(order_by(matrix, permutation_to_order(
        permutations[i]))), i] for i in range(population)])
    return fitnesses


def find_best_permutation(fitnesses):
    """
    :param fitnesses:
    :return:
    """
    best_fitness = fitnesses[np.argmin(fitnesses[:, 0], axis=0)][0]
    best_indeces = fitnesses[fitnesses[:, 0] == best_fitness][:, 1]
    best_index = int(np.random.choice(best_indeces))
    return best_fitness, best_index


def evolutionary_reorder(matrix, population=INITIAL_POPULATION, generations=GENERATIONS, stop_thresh=STOP_THRESHOLD, fitness=DIS):
    matrix_ = """
    ordering matrix by Optimizing the Ordering of Tables With Evolutionary Computation
    :param matrix: square numpy matrix
    :param population: int
    :param generations: int
    :param stop_thresh: int, stop the evolution if best fitness does not change more than STOP_THRESHOLD
    :param fitness: function the fitness function
    :return: reordered matrix
    """
    n = matrix.shape[0]
    permutations = [np.arange(n)]
    for i in range(population):
        permutation = np.arange(n)
        np.random.shuffle(permutation)
        permutations.append(permutation)
    permutations = np.array(permutations)

    fitnesses = compute_fitnesses(matrix, permutations, fitness)
    best_fitness, best_index = find_best_permutation(fitnesses)
    best_permutation = permutations[best_index]

    no_change_generation = 0
    last_best_fitness = best_fitness
    for i in range(generations):
        population = len(permutations)
        new_permutations = [best_permutation]

        permutations = mutation(permutations.copy())

        for _ in range(population):
            idxs_for_cmpr = np.random.randint(population, size=2)  # indeces for comparison
            better_fitness, better_index = find_best_permutation(fitnesses[idxs_for_cmpr])
            better_permutation = permutations[better_index]
            new_permutations.append(better_permutation)

        permutations = np.array(new_permutations)

        fitnesses = compute_fitnesses(matrix, permutations, fitness)
        best_fitness, best_index = find_best_permutation(fitnesses)
        best_permutation = permutations[best_index]
        new_permutations.append(best_permutation)

        if best_fitness == last_best_fitness:
            no_change_generation += 1
            if no_change_generation > stop_thresh:
                break
        elif best_fitness < last_best_fitness:
            # print(best_fitness, no_change_generation, i)
            last_best_fitness = best_fitness
            no_change_generation = 0

    best_order = permutation_to_order(best_permutation)
    # best_order = best_permutation
    matrix = order_by(matrix, best_order)
    return matrix, best_order
