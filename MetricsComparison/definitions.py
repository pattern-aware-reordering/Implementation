from MatrixReordering.Metrics import MI, LA, PR, BW

METRIC_FUNCS = {
    "MI": lambda matrix: MI(matrix, scaled=True),
    "LA": lambda matrix: LA(matrix, scaled=True),
    "PR": lambda matrix: PR(matrix, scaled=True),
    "BW": lambda matrix: BW(matrix, scaled=True)
}

ALGORITHMS = ["biclustering", "evolutionary_reorder", "greedy_ordering", "MDS", "MinLA", "optimal_leaf_ordering-delta", "optimal_leaf_ordering", "randomized_ordering", "rank_two"]

DATASETS = ["wiki_talk_br", "petster-hamster", "chesapeake", "bn-mouse", "wiki_edit_eu", "bn-fly", "everglades", "lesmis", "jazz", "bio-grid-plant", "socfb-Caltech36", "radoslaw", "asoiaf", "visbrazil", "econ-wm2", "bio-grid-mouse", "sch", "netscience", "price_1000", "dwt_419"]

# DATASETS = ["chesapeake", "everglades", "lesmis", "radoslaw", "jazz"]  # small graphs
