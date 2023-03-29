import json
import random
import time
import heapq

import networkx as nx
import numpy as np

from DEFINITIONS import *
from MDL import SuperGraph, connect_nodes, output, filter_level_1_nodes, get_clustering, cost_computations
from MatrixReordering.Matrix import order_by, cal_pairwise_moran_dist
from MatrixReordering.MinLA import MinLA
from MatrixReordering.OLO import optimal_leaf_ordering
from MatrixReordering.TwoLevelReordering import merge_matrix_with_sets


class HCluster:
    def __init__(self, leaves, dist_func):
        self.n_leaves = len(leaves)
        self.nodes = [{
            "name": leaves[i],
            "index": i,
            "leaves": [leaves[i]],
            "left": None,
            "right": None,
            "parent": None
        } for i in range(self.n_leaves)]
        self.leaves = self.nodes.copy()  # only copy one level
        self.name2node = {leaves[i]: self.nodes[i]
                          for i in range(self.n_leaves)}
        self.name2root = self.name2node.copy()
        self.dist2ancestor = {leaf: {leaf: 0} for leaf in leaves}
        self.lowest_common_ancestor = {leaf: {} for leaf in leaves}
        self.node_depth = {}
        self.Z = []
        self.dist_func = dist_func
        self.root_distances = None
        self.leaf_distances = None

    def merge(self, u, v, w, distance):
        if distance < 0:
            breakpoint()
        self.name2node[w] = {
            "index": len(self.nodes),
            "leaves": self.name2node[u]["leaves"] + self.name2node[v]["leaves"],
            "name": w,
            "left": self.name2node[u],
            "right": self.name2node[v],
            "distance": distance,
            "parent": None
        }
        self.name2root[w] = self.name2node[w]
        self.nodes.append(self.name2node[w])
        for x in (u, v):
            self.name2node[x]["parent"] = self.name2node[w]
            del self.name2root[x]

        self.Z.append([self.name2node[u]["index"], self.name2node[v]
                       ["index"], distance, len(self.name2node[w]["leaves"])])

    @staticmethod
    def single_root_preorder_traversal(root, callback):
        stack = []
        node = root
        # pre-order traversal (https://zh.m.wikipedia.org/zh-hans/%E6%A0%91%E7%9A%84%E9%81%8D%E5%8E%86)
        while node or stack:
            while node:
                # update distance to root (dist2ancestor) and node depth
                callback(node)
                stack.append(node)
                node = node["left"]

            node = stack.pop()
            node = node["right"]

    def preorder_traversal(self, callback):
        roots = list(self.name2root.values())
        for root in roots:
            self.single_root_preorder_traversal(root, callback)

    def compute_leaf_distances(self, SG, VS, is_merge_roots=False):
        def traversal_callback(node):
            name = node["name"]
            self.dist2ancestor[name] = self.dist2ancestor[name] if name in self.dist2ancestor else {
            }
            self.dist2ancestor[name][name] = 0
            if node["parent"] is not None:  # not a root
                parent = node["parent"]
                parent_name = parent["name"]
                self.node_depth[name] = self.node_depth[parent_name] + 1
                ancestor = parent
                while ancestor:
                    ancestor_name = ancestor["name"]
                    self.dist2ancestor[name][ancestor_name] = self.dist2ancestor[parent_name][ancestor_name] + parent["distance"]
                    ancestor = ancestor["parent"]
            self.dist2ancestor[name][name] = 0
            left_tree = node["left"]
            right_tree = node["right"]
            if left_tree and right_tree:
                for l_leaf in left_tree["leaves"]:
                    for r_leaf in right_tree["leaves"]:
                        self.lowest_common_ancestor[l_leaf][r_leaf] = self.lowest_common_ancestor[r_leaf][l_leaf] = node

        if is_merge_roots:
            self.merge_roots(SG, VS)

        self.node_depth = {root_name: 1 for root_name in self.name2root}
        # compute distance to ancestor and node depth
        self.preorder_traversal(callback=traversal_callback)

        leaf_to_root = {}
        for root_name in self.name2root:
            root = self.name2root[root_name]
            for leaf_name in root["leaves"]:
                leaf_to_root[leaf_name] = root

        roots = list(self.name2root.values())
        root2index = {}
        for i in range(len(roots)):
            name = roots[i]["name"]
            root2index[name] = i
        distances = np.zeros((self.n_leaves, self.n_leaves))
        for i in range(self.n_leaves):
            leaf_i = self.leaves[i]
            leaf_i_name = leaf_i["name"]
            for j in range(i + 1, self.n_leaves):
                leaf_j = self.leaves[j]
                leaf_j_name = leaf_j["name"]
                if leaf_j_name in self.lowest_common_ancestor[leaf_i_name]:
                    # lowest common ancestor
                    lca = self.lowest_common_ancestor[leaf_i_name][leaf_j_name]
                    lca_name = lca["name"]
                    distance = self.dist2ancestor[leaf_i_name][lca_name] + \
                        self.dist2ancestor[leaf_j_name][lca_name]
                elif is_merge_roots:
                    raise Exception('Has multiple roots!')
                else:
                    distance = np.Inf
                distances[j, i] = distances[i, j] = distance
        self.leaf_distances = distances
        return distances

    def merge_roots(self, SG, VS):
        SG = SG.copy()
        # calculate distance between each pair of roots
        roots = list(self.name2root.values())
        n_roots = len(roots)
        root_distances = np.zeros((n_roots, n_roots))

        # FIRST PART: initialization phase
        H = []
        VS = VS.copy()
        traversed = set()
        for node_1 in VS:
            if node_1 not in SG.node_to_supernode:
                for node_2 in VS:
                    if node_2 not in SG.node_to_supernode:
                        if node_1 is not node_2 and (node_1, node_2) not in traversed and (node_2, node_1) not in traversed:
                            traversed.add((node_1, node_2))
                            s = SG.s(node_1, node_2)
                            # use -1.0 because heapq is a min-heap
                            heapq.heappush(H, (-1.0 * s, (node_1, node_2)))

        # SECOND PART: iterative merging phase
        k = 0
        while len(H) > 0:
            k += 1
            # largest_pair = H[0][1]
            s, largest_pair = heapq.heappop(H)
            u, v = largest_pair

            # merge supernodes u and v and remove them from VS
            # try:
            w, _ = SG.merge(u, v)
            VS.remove(u)
            VS.remove(v)
            VS.add(w)
            self.merge(u, v, w, distance=1-s)
            # except KeyError:
            #     breakpoint()

            # remove all pairs (u, x) and (v, x) in H that x is within 2 hops of u or v
            i = 0
            while i < len(H):
                pair = H[i][1]
                if (pair[0] in {u, v} and pair[1] in VS) or (pair[1] in {u, v} and pair[0] in VS):
                    # remove pair from H
                    H[i] = H[-1]
                    H.pop()
                else:
                    i += 1
            # heapq.heapify(H)
            for x in VS:
                if w is not x:
                    s = SG.s(w, x)
                    H.append((-1.0 * s, (w, x)))

            # recompute costs of w's neighbors
            def is_node_not_in_supernodes(node):
                return node not in SG.node_to_supernode

            neighbors_of_w = set(
                filter(lambda n: (n is not w and is_node_not_in_supernodes(n)), SG.G[w]))
            i = 0
            while i < len(H):
                u, v = H[i][1]
                if u in neighbors_of_w or v in neighbors_of_w:
                    s = SG.s(u, v)
                    H[i] = (-1.0 * s, (u, v))
                i += 1
            heapq.heapify(H)

    def compute_root_distances(self):
        # calculate distance between each pair of roots
        roots = list(self.name2root.values())
        n_roots = len(roots)
        root_distances = np.zeros((n_roots, n_roots))
        for i in range(n_roots):
            root_i = roots[i]["name"]
            for j in range(i + 1, n_roots):
                root_j = roots[j]["name"]
                d = self.dist_func(root_i, root_j)
                root_distances[i][j] = root_distances[j][i] = d
        self.root_distances = root_distances
        return root_distances

    def linkage(self):
        return np.array(self.Z)


def order_by_minLA_OLO(G, distances, summarization, hCluster):
    # Two-Level Reordering
    node_list = list(G.nodes)
    index_dict = dict([(node_list[i], i) for i in range(len(node_list))])
    A = np.asarray(nx.adjacency_matrix(G).todense())

    # the first level
    level_1_nodes = filter_level_1_nodes(summarization)
    merge_sets = get_clustering(summarization)
    for i in range(len(merge_sets)):
        merge_set = merge_sets[i]
        merge_sets[i] = [index_dict[id] for id in merge_set]
    level_1_A = merge_matrix_with_sets(A, merge_sets)
    level_1_A, level_1_order = MinLA(level_1_A)

    # the second level
    order_id = []
    order_id_split = []
    k = 0
    for i in level_1_order:
        level_1_node = level_1_nodes[i]
        if level_1_node['type'] == SUPERNODE:
            super_node = level_1_node
            level_2_nodes_id = super_node['nodes']
            level_2_nodes_idx = [index_dict[id]
                                 for id in super_node['nodes']]
            level_2_A = A[np.ix_(level_2_nodes_idx, level_2_nodes_idx)]
            if level_2_A.shape[0] > 2:
                level_2_A, level_2_order = MinLA(level_2_A)
            else:
                level_2_order = range(level_2_A.shape[0])

            level_2_nodes = []
            for j in level_2_order:
                index = level_2_nodes_idx[j]
                level_2_node = node_list[index]
                level_2_nodes.append(level_2_node)
                order_id.append(level_2_node)

            order_id_split.append(level_2_nodes)

        else:
            order_id.append(level_1_node['id'])
            order_id_split.append([level_1_node['id']])

        k += 1

    order = [index_dict[id] for id in order_id]
    matrix = order_by(A, order)

    return matrix, order


def greedy_ordering(matrix, cost_computations=cost_computations, use_node_cost=True, use_edge_cost=True):
    times = {
        "summarization": 0,
        "ordering": 0,
        "total": 0
    }
    start_time = time.time()
    G = nx.from_numpy_matrix(matrix)
    mapping = {node: str(node) for node in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes)
    n = len(nodes)
    m = len(G.edges)
    SG = SuperGraph(G, cost_computations, use_node_cost, use_edge_cost)

    def is_node_not_in_supernodes(n):
        return n not in SG.node_to_supernode
    # FIRST PART: initialization phase
    H = []
    VS = set(list(filter(is_node_not_in_supernodes, SG.G.nodes)))
    traversed = nx.Graph()
    for node in VS:
        for hop_2_node in VS:
            if node is not hop_2_node and not traversed.has_edge(node, hop_2_node):
                traversed.add_edge(node, hop_2_node)
                s = SG.s(node, hop_2_node)
                if s > 0:
                    # use -1.0 because heapq is a min-heap
                    heapq.heappush(H, (-1.0 * s, (node, hop_2_node)))

    # SECOND PART: iterative merging phase
    hCluster = HCluster(nodes, dist_func=lambda a,
                        b: 1 - SG.s(a, b))

    k = 0
    sequence = []
    while len(H) > 0:
        k += 1
        # largest_pair = H[0][1]
        s, largest_pair = heapq.heappop(H)
        u, v = largest_pair
        sequence.append(largest_pair)

        # merge supernodes u and v and remove them from VS
        w, _ = SG.merge(u, v)
        VS.remove(u)
        VS.remove(v)
        VS.add(w)

        hCluster.merge(u, v, w, distance=1-s)

        # remove all pairs (u, x) and (v, x) in H that x is within 2 hops of u or v
        i = 0
        while i < len(H):
            pair = H[i][1]
            if (pair[0] in {u, v} and pair[1] in VS) or (pair[1] in {u, v} and pair[0] in VS):
                # remove pair from H
                H[i] = H[-1]
                H.pop()
            else:
                i += 1
        # heapq.heapify(H)
        for x in VS:
            if w is not x:
                s = SG.s(w, x)
                if s > 0:
                    H.append((-1.0 * s, (w, x)))

        # recompute costs of w's neighbors
        def is_node_not_in_supernodes(node):
            return node not in SG.node_to_supernode

        neighbors_of_w = set(
            filter(lambda n: (n is not w and is_node_not_in_supernodes(n)), SG.G[w]))
        i = 0
        while i < len(H):
            u, v = H[i][1]
            if u in neighbors_of_w or v in neighbors_of_w:
                s = SG.s(u, v)
                if s > 0:
                    H[i] = (-1.0 * s, (u, v))
                else:
                    H[i] = H[-1]
                    H.pop()
                    continue
            i += 1
        heapq.heapify(H)

    summarization = output(SG, G, VS)

    ##### first level: minLA, second level: OLO #####
    hCluster.compute_leaf_distances(SG, VS, is_merge_roots=False)
    matrix, order = order_by_minLA_OLO(
        G, hCluster.leaf_distances, summarization, hCluster)
    ##### first level: minLA, second level: OLO #####

    times["total"] = time.time() - start_time

    return matrix, order, summarization, times


def randomized_ordering(matrix, cost_computations=cost_computations, use_node_cost=True, use_edge_cost=True):
    times = {
        "summarization": 0,
        "ordering": 0,
        "total": 0
    }
    start_time = time.time()
    G = nx.from_numpy_matrix(matrix)
    mapping = {node: str(node) for node in G.nodes}
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes)
    n = len(nodes)
    m = len(G.edges)
    SG = SuperGraph(G, cost_computations, use_node_cost, use_edge_cost)

    def is_node_not_in_supernodes(n):
        return n not in SG.node_to_supernode

    F = []
    U = set(list(filter(is_node_not_in_supernodes, SG.G.nodes)))
    VS = set(list(filter(is_node_not_in_supernodes, SG.G.nodes)))

    hCluster = HCluster(nodes, dist_func=lambda a,
                        b: 1 - SG.s(a, b))

    while len(U) > 0:
        u = random.sample(U, 1)
        i = 0
        u = list(U)[i]
        max_s = -1
        max_v = None

        v_set = VS.copy()

        for v in v_set:
            if v is not u and v in U:
                s = SG.s(u, v)
                if s > max_s:
                    max_s = s
                    max_v = v

        if max_s > 0:
            v = max_v

            # clear cache of s
            for x in [u, v]:
                for node in SG.s_cache[x]:
                    SG.s_cache[node] = {}
                SG.s_cache[x] = {}

            w, _ = SG.merge(u, v)
            U.remove(u)
            U.remove(v)
            U.add(w)
            VS.remove(u)
            VS.remove(v)
            VS.add(w)

            hCluster.merge(u, v, w, distance=1-s)

            # update hop_2_nodes
        else:
            U.remove(u)
            F.append(u)


    summarization = output(SG, G, VS)

    ##### first level: minLA, second level: OLO #####
    hCluster.compute_leaf_distances(SG, VS, is_merge_roots=False)
    matrix, order = order_by_minLA_OLO(
        G, hCluster.leaf_distances, summarization, hCluster)
    ##### first level: minLA, second level: OLO #####

    times["total"] = time.time() - start_time

    return matrix, order, summarization, times
