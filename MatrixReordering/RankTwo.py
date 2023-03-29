import numpy as np
from ellipse import LsqEllipse
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
from MatrixReordering.Matrix import order_by, cal_pairwise_dist
from circle_fit import least_squares_circle


def pearson_correlation(matrix):
    M = pd.DataFrame(matrix)
    R = M.corr()
    R = R.to_numpy()
    return R


def fit_circle(x, y):
    # print("Input three coordinate of the circle:")
    x1, x2, x3, y1, y2, y3 = x.tolist() + y.tolist()
    c = (x1 - x2) ** 2 + (y1 - y2) ** 2
    a = (x2 - x3) ** 2 + (y2 - y3) ** 2
    b = (x3 - x1) ** 2 + (y3 - y1) ** 2
    s = 2 * (a * b + b * c + c * a) - (a * a + b * b + c * c)
    px = (a * (b + c - a) * x1 + b * (c + a - b) * x2 + c * (a + b - c) * x3) / s
    py = (a * (b + c - a) * y1 + b * (c + a - b) * y2 + c * (a + b - c) * y3) / s
    ar = a ** 0.5
    br = b ** 0.5
    cr = c ** 0.5
    r = ar * br * cr / ((ar + br + cr) * (-ar + br + cr) * (ar - br + cr) * (ar + br - cr)) ** 0.5
    # print("Radius of the said circle:")
    # print("{:>.3f}".format(r))
    # print("Central coordinate (x, y) of the circle:")
    # print("{:>.3f}".format(px), "{:>.3f}".format(py))
    return px, py, r


def rank_two(matrix):
    if np.sum(matrix) < 1:
        return matrix, np.arange(matrix.shape[0])
    D = cal_pairwise_dist(matrix)
    rank = matrix.shape[0]
    R = D
    while True:
        last_rank = rank
        last_R = R.copy()
        R = pearson_correlation(R)
        try:
            rank = np.linalg.matrix_rank(R)
        except:
            breakpoint()
        if rank <= 2:
            rank = last_rank
            R = last_R
            break

    pca = decomposition.PCA(n_components=2)
    pca.fit(R)
    X = pca.transform(R)
    x = X[:, 0]
    y = X[:, 1]

    if x.shape[0] > 4 and not np.all(np.polyfit(x, y, 1) < 1e-5):
        reg = LsqEllipse().fit(X)
        try:
            center, width, height, phi = reg.as_parameters()
        except IndexError:
            breakpoint()
        x0, y0 = center
    else:
        circle = least_squares_circle(X)
        x0, y0, r, residual = circle

    V = X - [x0, y0]
    theta = np.arcsin(V[:, 0] / np.linalg.norm(V, axis=1))
    theta[V[:, 0] * V[:, 1] < 0] += np.pi / 2
    order = np.argsort(theta)

    matrix = order_by(matrix, order)
    return matrix, order
