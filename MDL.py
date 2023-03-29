import random
import heapq
import networkx as nx
import numpy as np
from DEFINITIONS import *

EPS = 1
WEIGHT_OF_POSITIVE_CORRECTION = 1
WEIGHT_OF_PATTERN = 1


def connect_nodes(u, v):
    return str(u) + '+' + str(v)


def filter_level_1_nodes(summarization):
    return list(filter(lambda x: x['type'] in [SUPERNODE, INDEPENDENT_NODE], summarization['nodes']))


def filter_super_nodes(summarization):
    return list(filter(lambda x: x['type'] == SUPERNODE, summarization['nodes']))


def get_clustering(summarization):
    level_1_nodes = filter_level_1_nodes(summarization)
    merge_sets = [level_1_node['nodes'] if 'nodes' in level_1_node else [level_1_node['id']] for level_1_node in
                  level_1_nodes]
    return merge_sets


def count_edges_or_not_between(G, left_nodes, right_nodes):
    existing_edges = []  # A_uv
    not_existing_edges = []  # Pi_uv - A_uv
    traversed = set()
    for left_node in left_nodes:
        for right_node in right_nodes:
            if left_node is not right_node and (left_node, right_node) not in traversed and (right_node, left_node) not in traversed:
                if G.has_edge(left_node, right_node):
                    existing_edges.append((left_node, right_node))
                else:
                    not_existing_edges.append((left_node, right_node))
            traversed.add((left_node, right_node))
    existing_edges_count = len(existing_edges)
    not_existing_edges_count = len(not_existing_edges)

    return existing_edges_count, not_existing_edges_count


def compute_cost_of_clique(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a clique
    :param v: the supernode
    :return: the correnction edges, -1/1 mean minus/add it to correct the supernode to make it a clique
    """
    n = len(nodes)
    m = len(edges)
    return n * (n - 1) / 2 - m


def correction_of_clique(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a clique
    :param v: the supernode
    :return: the correnction edges, -1/1 mean minus/add it to correct the supernode to make it a clique
    """
    correction = []
    # if regard as a clique
    nodes_of_v = list(nodes)
    i = 0
    while i < len(nodes_of_v):
        j = i + 1
        while j < len(nodes_of_v):
            if (nodes_of_v[i], nodes_of_v[j]) not in edges and (nodes_of_v[j], nodes_of_v[i]) not in edges:
                correction.append((1, (nodes_of_v[i], nodes_of_v[j])))
            j += 1
        i += 1

    return correction


def compute_cost_of_star(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a star
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode to make it a star
    """
    n = len(nodes)
    m = len(edges)
    hub, max_degree = max(degrees, key=lambda x: x[1])
    cost = (n - 1 - max_degree) + (m - max_degree)
    return cost


def correction_of_star(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a star
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode to make it a star
    """
    correction = []
    # if regard as a star
    hub = max(degrees, key=lambda x: x[1])[0]
    for edge in edges:
        if edge[0] != hub and edge[1] != hub:
            correction.append((-1, edge))
    for node in nodes:
        if (node, hub) not in edges and (hub, node) not in edges:
            correction.append((1, (node, hub)))

    return correction


def compute_cost_of_bipartite_core(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a bipartite core
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode to make it a bipartite core
    """
    nodes = [node for (node, degree) in sorted(
        degrees, key=lambda x: x[1])]
    center_of_left = nodes[-1]  # max degree node
    neighbors_of_left = list(subgraph.neighbors(center_of_left))
    cost = 0
    n = len(nodes)

    if len(neighbors_of_left) == 0:  # no edges
        return int(np.floor(n / 2)) * (int(np.floor(n / 2)) + 1)

    i = 0
    degree_of_neighbors = []
    while i < len(neighbors_of_left):
        x = neighbors_of_left[i]
        d = subgraph.degree(x)
        j = 0
        while j < len(neighbors_of_left):
            y = neighbors_of_left[j]
            if i != j:
                if (x, y) in edges or (y, x) in edges:
                    d -= 1
            j += 1
        # d = degree - number of self-connected links
        degree_of_neighbors.append((x, d))
        i += 1
    center_of_right = max(degree_of_neighbors, key=lambda x: x[1])[0]

    left = [center_of_left]
    right = [center_of_right]
    for node in nodes:
        if node != center_of_left and node != center_of_right:
            left_edges_count, left_no_edges_count = count_edges_or_not_between(
                subgraph, left, [node])
            right_edges_count, right_no_edges_count = count_edges_or_not_between(
                subgraph, right, [node])
            cost_as_left = left_edges_count + right_no_edges_count
            cost_as_right = right_edges_count + left_no_edges_count
            cost += min(cost_as_right, cost_as_left)
            if cost_as_left < cost_as_right:
                left.append(node)
            else:
                right.append(node)

    return cost


def correction_of_bipartite_core(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a bipartite core
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode to make it a bipartite core
    """
    nodes = [node for (node, degree) in sorted(
        degrees, key=lambda x: x[1])]
    center_of_left = nodes[-1]  # max degree node
    neighbors_of_left = list(subgraph.neighbors(center_of_left))
    correction = []

    if len(neighbors_of_left) == 0:  # no edges
        left_nodes = []
        right_nodes = []
        for i in range(len(nodes)):
            j = len(nodes) - i - 1
            if i < j:
                left_nodes.append(nodes[i])
                right_nodes.append(nodes[j])
            elif i == j:
                left_nodes.append(nodes[i])
            else:
                break
        for left_node in left_nodes:
            for right_node in right_nodes:
                correction.append((1, (left_node, right_node)))
        return correction

    i = 0
    degree_of_neighbors = []
    while i < len(neighbors_of_left):
        x = neighbors_of_left[i]
        d = subgraph.degree(x)
        j = 0
        while j < len(neighbors_of_left):
            y = neighbors_of_left[j]
            if i != j:
                if (x, y) in edges or (y, x) in edges:
                    d -= 1
            j += 1
        degree_of_neighbors.append((x, d))
        i += 1
    center_of_right = max(degree_of_neighbors, key=lambda x: x[1])[0]

    # center_of_right = max(subgraph.degree(neighbors_of_left), key=lambda x: x[1])[0]
    left = center_of_left
    right = center_of_right
    super_subgraph = SuperGraph(subgraph)
    for node in nodes:
        if node != center_of_left and node != center_of_right:
            left_edges, left_no_edges = super_subgraph.existing_or_not_edges(
                left, node)
            right_edges, right_no_edges = super_subgraph.existing_or_not_edges(
                right, node)
            if (len(left_edges) + len(right_no_edges)) < (len(right_edges) + len(left_no_edges)):  # left node
                left, _ = super_subgraph.merge(left, node)
                for edge in left_edges:
                    correction.append((-1, edge))
                for edge in right_no_edges:
                    correction.append((1, edge))
            else:
                right, _ = super_subgraph.merge(right, node)
                for edge in right_edges:
                    correction.append((-1, edge))
                for edge in left_no_edges:
                    correction.append((1, edge))

    return correction


def compute_cost_of_chain(subgraph, nodes, edges, degrees):
    correction = correction_of_chain(subgraph, nodes, edges, degrees)
    return len(correction)


def correction_of_chain(subgraph, nodes, edges, degrees):
    components = sorted(nx.connected_components(
        subgraph), key=len, reverse=True)
    components = [subgraph.subgraph(c) for c in components]
    chains = []
    chain_edges = set()
    while len(components) > 0:
        component = components.pop()
        if len(component.nodes) == 1:
            chains.append(list(component.nodes))
            continue
        subgraph = component.copy()
        degrees = subgraph.degree
        min_deg_node = min(degrees, key=lambda x: x[1])[0]
        furthest_node = list(nx.bfs_edges(
            subgraph, source=min_deg_node))[-1][-1]
        init_node = furthest_node
        chain = []
        while True:
            predecessors = list(nx.bfs_predecessors(
                subgraph, source=init_node))
            if len(predecessors) < 1:
                chain.append(init_node)
                subgraph.remove_node(init_node)
                break
            subsequent_furthest_node = predecessors[-1][0]
            predecessors = dict(predecessors)
            sub_chain = []
            node = subsequent_furthest_node
            while node != init_node:
                next_node = predecessors[node]
                chain_edges.add((node, next_node))
                sub_chain.append(next_node)
                node = next_node
            subgraph.remove_nodes_from(sub_chain)
            chain = chain + list(reversed(sub_chain))
            init_node = subsequent_furthest_node

        components += [subgraph.subgraph(c) for c in sorted(
            nx.connected_components(subgraph), key=len, reverse=True)]
        chains.append(chain)

    correction = []
    i = 0
    while i < len(chains) - 1:
        correction.append((1, (chains[i][-1], chains[i + 1][0])))
        i += 1

    for edge in edges:
        if edge not in chain_edges and (edge[1], edge[0]) not in chain_edges:
            correction.append((-1, edge))

    return correction


def compute_cost_of_none(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as no-structure, namely the number of edges contained in v
    :param v: the supernode
    :return: the correction edges to make it no-structure, namely the number of edges contained in v
    """
    return len(edges)


def correction_of_none(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as no-structure, namely the number of edges contained in v
    :param v: the supernode
    :return: the correction edges to make it no-structure, namely the number of edges contained in v
    """
    correction = [(-1, edge) for edge in edges]

    return correction


def compute_cost_of_wheel(subgraph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a wheel
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode to make it a star
    """
    n = len(nodes)
    m = len(edges)
    hub, max_degree = max(degrees, key=lambda x: x[1])
    cost = (n - 1 - max_degree) + (m - max_degree)
    return cost


def compute_cost_of_tree(subgraph: nx.Graph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a tree
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode
    """
    n = len(nodes)
    m = len(edges)
    hub, max_degree = max(degrees, key=lambda x: x[1])
    bfs_edges = nx.bfs_tree(subgraph, source=hub).edges()
    cost = m - len(list(bfs_edges))
    return cost


def compute_cost_of_prism(subgraph: nx.Graph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as a Prism Graph (https://mathworld.wolfram.com/PrismGraph.html)
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode
    """
    n = len(nodes)
    graph = subgraph.copy()
    components = sorted(nx.connected_components(
        subgraph), key=len, reverse=True)
    components = [subgraph.subgraph(c) for c in components]
    chains = []
    chain_edges = set()
    while len(components) > 0:
        component = components.pop()
        if len(component.nodes) == 1:
            chains.append(list(component.nodes))
            continue
        subgraph = component.copy()
        degrees = subgraph.degree
        min_deg_node = min(degrees, key=lambda x: x[1])[0]
        furthest_node = list(nx.bfs_edges(
            subgraph, source=min_deg_node))[-1][-1]
        init_node = furthest_node
        chain = []
        while True:
            predecessors = list(nx.bfs_predecessors(
                subgraph, source=init_node))
            if len(predecessors) < 1:
                chain.append(init_node)
                subgraph.remove_node(init_node)
                break
            subsequent_furthest_node = predecessors[-1][0]
            predecessors = dict(predecessors)
            sub_chain = []
            node = subsequent_furthest_node
            while node != init_node:
                next_node = predecessors[node]
                chain_edges.add((node, next_node))
                sub_chain.append(next_node)
                node = next_node
            subgraph.remove_nodes_from(sub_chain)
            chain = chain + list(reversed(sub_chain))
            init_node = subsequent_furthest_node

        components += [subgraph.subgraph(c) for c in sorted(
            nx.connected_components(subgraph), key=len, reverse=True)]
        chains.append(chain)

    cost = -1
    chain = []
    i = 0
    while i < len(chains):
        chain = chain + chains[i]
        cost += 1
        # correction.append((1, (chains[i][-1], chains[i + 1][0])))
        i += 1

    if len(chain) % 2:
        chain.pop()
        cost += 1

    n = len(chain)

    sub_costs = []
    visited_edges_list = []
    chain_edges_list = []
    chains = [chain, chain.copy()]
    chains[1].reverse()
    for chain in chains:
        for i in range(len(chain) // 2):
            chain_copy = chain.copy()
            chain_copy = chain_copy[i:] + chain_copy[0:i]
            chain_edges = set()
            for i in range(len(chain_copy) - 1):
                chain_edges.add((chain_copy[i], chain_copy[i + 1]))
            chain_edges_list.append(chain_edges)
            visited_edges = set()
            sub_cost = 0
            for j in range(n // 2 - 1):
                u = chain_copy[j]
                v = chain_copy[n - 1 - j]
                if not graph.has_edge(u, v):
                    sub_cost += 1
                else:
                    visited_edges.add((u, v))

            for edge_index in [(0, n // 2 - 1), (-1, -(n//2))]:
                edge = (chain_copy[edge_index[0]], chain_copy[edge_index[1]])
                if not graph.has_edge(edge[0], edge[1]):
                    sub_cost += 1
                else:
                    visited_edges.add(edge)
            sub_costs.append(sub_cost)
            visited_edges_list.append(visited_edges)

    min_i = 0
    min_sub_cost = sub_costs[min_i]
    for i in range(len(sub_costs)):
        sub_cost = sub_costs[i]
        if sub_cost < min_sub_cost:
            min_sub_cost = sub_cost
            min_i = i

    visited_edges = visited_edges_list[min_i]
    chain_edges = chain_edges_list[min_i]

    for edge in edges:
        if edge not in chain_edges and edge not in visited_edges and (edge[1], edge[0]) not in chain_edges and (edge[1], edge[0]) not in visited_edges:
            cost += 1

    return cost


def compute_cost_of_mesh(subgraph: nx.Graph, nodes, edges, degrees):
    """
    compute the correction of regarding supernode v as meshes
    :param v: the supernode
    :return: the correction edges, -1/1 mean minus/add it to correct the supernode
    """
    # neighbors of one node should connect to another neighbor of it
    n = len(nodes)
    m = len(edges)
    subgraph = subgraph.copy()
    isolated_nodes = []
    isolated_edges = set()
    cost = 0
    for i in range(n):
        node = nodes[i]
        neighbors = list(subgraph.neighbors(node))
        if len(neighbors) == 0:
            isolated_nodes.append(node)
        elif len(neighbors) == 1:
            neighbor = neighbors[0]
            neighbors_of_neighbor = list(subgraph.neighbors(neighbor))
            if len(neighbors_of_neighbor) == 1:
                # isolated edges
                if (neighbor, node) not in isolated_edges:
                    isolated_edges.add((node, neighbor))
            else:
                for nn in neighbors_of_neighbor:
                    if nn is not node:
                        subgraph.add_edge(node, nn)
                        cost += 1
                        break
        else:
            not_connected_neighbors = []
            for j in range(len(neighbors)):
                is_connected = False
                for k in range(len(neighbors)):
                    if subgraph.has_edge(neighbors[j], neighbors[k]):
                        is_connected = True
                        break
                if not is_connected:
                    not_connected_neighbors.append(neighbors[j])
            connected_neighbors = set()
            for j in range(len(not_connected_neighbors)):
                node1 = not_connected_neighbors[j]
                if node1 in connected_neighbors:
                    continue
                is_found = False
                for k in range(len(not_connected_neighbors)):
                    node2 = not_connected_neighbors[k]
                    if node2 in connected_neighbors:
                        continue
                    if node1 is not node2:
                        subgraph.add_edge(node1, node2)
                        connected_neighbors.add(node1)
                        connected_neighbors.add(node2)
                        cost += 1
                        is_found = True
                        break
                if not is_found:
                    for neighbor in subgraph.neighbors(node):
                        if neighbor is not node1:
                            subgraph.add_edge(node1, neighbor)
                            cost += 1
                            break

    cost += int(len(isolated_edges) / 2) * 3 + (len(isolated_edges) % 2) * 2
    cost += int(len(isolated_nodes) / 3) * 3 + (len(isolated_nodes) % 3) * 1.5

    return cost


cost_computations = {
    C: compute_cost_of_clique,
    CH: compute_cost_of_chain,
    BC: compute_cost_of_bipartite_core,
    S: compute_cost_of_star,
    E: compute_cost_of_none,
}

correction_computations = {
    C: correction_of_clique,
    CH: correction_of_chain,
    BC: correction_of_bipartite_core,
    S: correction_of_star,
    E: correction_of_none,
}


class SuperGraph:
    def __init__(self, G, cost_computations=cost_computations, use_node_cost=True, use_edge_cost=True):
        self.G = G.copy()
        self.use_node_cost = use_node_cost
        self.use_edge_cost = use_edge_cost
        # self.A = nx.adjacency_matrix(self.G).todense()
        self.supernode_to_node = {}
        self.node_to_supernode = {}
        self.same_role_nodes_map = {}
        self.NODE_COUNT = len(self.G.nodes)
        # origin_nodes = list(self.G.nodes)
        # self.origin_node_id2index = dict([(origin_nodes[i], i) for i in range(len(origin_nodes))])

        self.exone_cache = {}  # cache of EXisiting Or Not Edges
        self.node_cost_cache = {}  # correction cache
        self.s_cache = {}
        self.cost_computations = cost_computations

    def copy(self):
        SG = SuperGraph(self.G, self.use_node_cost, self.use_edge_cost)
        # SG.A = np.deepcopy(self.A)
        SG.supernode_to_node = {supernode_id: self.supernode_to_node[supernode_id].copy(
        ) for supernode_id in self.supernode_to_node}
        SG.node_to_supernode = self.node_to_supernode.copy()
        SG.same_role_nodes_map = {
            id: self.same_role_nodes_map[id].copy() for id in self.same_role_nodes_map}
        return SG

    def merge_same_role_nodes(self):
        # Find Nodes with Same Neighbors
        all_nodes = list(self.G.nodes)
        n = len(all_nodes)
        i = 0
        while i < n:
            u = all_nodes[i]
            neighbors_of_u = list(self.G.neighbors(u))
            j = i + 1
            while j < n:
                v = all_nodes[j]
                if u == v:
                    j += 1
                    continue
                neighbors_of_v = set(self.G.neighbors(v))
                if len(neighbors_of_u) == len(neighbors_of_v):
                    is_all_same = True
                    for neighbor_of_u in neighbors_of_u:
                        if neighbor_of_u not in neighbors_of_v:
                            is_all_same = False
                            continue
                    if is_all_same:
                        same_nodes = set()
                        if u in self.same_role_nodes_map:
                            same_nodes = self.same_role_nodes_map[u]
                        elif v in self.same_role_nodes_map:
                            same_nodes = self.same_role_nodes_map[v]
                        same_nodes.add(u)
                        same_nodes.add(v)
                        self.same_role_nodes_map[u] = same_nodes
                        self.same_role_nodes_map[v] = same_nodes
                j += 1
            i += 1

        for node in self.same_role_nodes_map:
            if node not in self.node_to_supernode:
                nodes = list(self.same_role_nodes_map[node])
                merged_nodes = []
                while len(nodes) > 1:
                    i = 0
                    while i < len(nodes) / 2:
                        j = len(nodes) - 1 - i
                        if i < j:
                            merged_node, _ = self.merge(
                                nodes[i], nodes[len(nodes) - 1 - i])
                            merged_nodes.append(merged_node)
                        else:
                            merged_nodes.append(nodes[i])
                        i += 1
                    nodes = merged_nodes
                    merged_nodes = []

    def count_existing_or_not_edges(self, u, v):
        """
        compute exsisting edges and not exsisting edges between two supernode u and v
        :param u: node/supernode
        :param v: node/supernode
        :return: list of existing edges, list of not existing edges
        """
        # read from cache
        if u in self.exone_cache:
            if v in self.exone_cache[u]:
                return self.exone_cache[u][v]

        nodes_of_u = [u]
        nodes_of_v = [v]
        if u in self.supernode_to_node:
            nodes_of_u = list(self.supernode_to_node[u])
        if v in self.supernode_to_node:
            nodes_of_v = list(self.supernode_to_node[v])

        existing_edges_count, not_existing_edges_count = count_edges_or_not_between(
            self.G, nodes_of_u, nodes_of_v)

        self.exone_cache[u] = self.exone_cache[u] if u in self.exone_cache else {}
        self.exone_cache[u][v] = (
            existing_edges_count, not_existing_edges_count)
        self.exone_cache[v] = self.exone_cache[v] if v in self.exone_cache else {}
        self.exone_cache[v][u] = (
            existing_edges_count, not_existing_edges_count)

        return existing_edges_count, not_existing_edges_count

    def existing_or_not_edges(self, u, v):
        """
        compute exsisting edges and not exsisting edges between two supernode u and v
        :param u: node/supernode
        :param v: node/supernode
        :return: list of existing edges, list of not existing edges
        """
        existing_edges = []  # A_uv
        not_existing_edges = []  # Pi_uv - A_uv
        nodes_of_u = [u]
        nodes_of_v = [v]
        if u in self.supernode_to_node:
            nodes_of_u = self.supernode_to_node[u]
        if v in self.supernode_to_node:
            nodes_of_v = self.supernode_to_node[v]

        traversed = set()
        for node_of_u in nodes_of_u:
            for node_of_v in nodes_of_v:
                if node_of_u is not node_of_v and (node_of_u, node_of_v) not in traversed and (node_of_v, node_of_u) not in traversed:
                    if self.G.has_edge(node_of_u, node_of_v):
                        existing_edges.append((node_of_u, node_of_v))
                    else:
                        not_existing_edges.append((node_of_u, node_of_v))
                traversed.add((node_of_v, node_of_u))

        return existing_edges, not_existing_edges

    def cost_of_edge(self, u, v):
        """
        computing the cost of encoding a superedge uv
        :param u: the source supernode
        :param v: the target supernode
        :return: the cost
        """
        # existing_edges, not_existing_edges = self.count_existing_or_not_edges(u, v)
        # A_uv = len(existing_edges)
        # Pi_uv = len(existing_edges) + len(not_existing_edges)
        existing_edges_count, not_existing_edges_count = self.count_existing_or_not_edges(
            u, v)
        A_uv = existing_edges_count
        Pi_uv = existing_edges_count + not_existing_edges_count
        cost = min(Pi_uv - A_uv + 1, A_uv * WEIGHT_OF_POSITIVE_CORRECTION)

        return cost

    def cost_of_edge_if_merging(self, u, v, neighbor):
        """
        computing the cost of uv to neighbor
        :param u: the node to be merge
        :param v: the node to be merge
        :param neighbor: their neighbor
        :return: the cost
        """
        # existing_edges, not_existing_edges = self.count_existing_or_not_edges(u, v)
        # A_uv = len(existing_edges)
        # Pi_uv = len(existing_edges) + len(not_existing_edges)
        u_existing_edges_count, u_not_existing_edges_count = self.count_existing_or_not_edges(
            u, neighbor)
        v_existing_edges_count, v_not_existing_edges_count = self.count_existing_or_not_edges(
            v, neighbor)

        A_uv = u_existing_edges_count + v_existing_edges_count
        Pi_uv = u_not_existing_edges_count + v_not_existing_edges_count + A_uv
        cost = min(Pi_uv - A_uv + 1, A_uv * WEIGHT_OF_POSITIVE_CORRECTION)

        return cost

    def compute_min_cost_of_nodes(self, nodes):
        # sub_A = self.A[np.ix_(nodes, nodes)]
        subgraph = self.G.subgraph(nodes).copy()
        edges = set([(edge[0], edge[1]) for edge in subgraph.edges])
        degrees = list(subgraph.degree)
        # n = len(nodes)
        # m = len(edges)
        # n = sub_A.shape[0]
        # m = np.nonzero(sub_A)[0].shape[0] / 2

        # for 'none' and 'clique', min_costs_in_theory is actually their costs
        # thus if their costs are less than min_costs_in_theory of star and bipartite_core
        # we do not need to calculate the others
        costs = {}
        for key in self.cost_computations:
            costs[key] = self.cost_computations[key](
                subgraph, nodes, edges, degrees)

        min_struct = None
        min_cost = float('inf')
        min_structs = []
        for struct in costs:
            cost = costs[struct]
            if cost < min_cost:
                min_structs = [struct]
                min_cost = cost
            elif cost == min_cost:
                min_structs.append(struct)

        priority = {S: 30, C: 10, BC: 40, CH: 20,
                    E: 50, WH: 35, MS: 15, TR: 35, PR: 20}
        min_struct = sorted(min_structs, key=lambda s: priority[s])[0]

        return costs, min_cost, min_struct

    def compute_node_min_cost(self, v):
        """
        computing the min cost of encoding a supernode v as clique/star/...
        :param v: a supernode
        :return: the correction, the structure
        """
        if v in self.node_cost_cache:
            return self.node_cost_cache[v]

        if v not in self.supernode_to_node:
            self.node_cost_cache[v] = {
                "corrections": {key: 0 for key in self.cost_computations},
                "min_cost": (0, E)
            }
            return self.node_cost_cache[v]

        nodes = list(self.supernode_to_node[v])
        costs, min_cost, min_struct = self.compute_min_cost_of_nodes(nodes)

        self.node_cost_cache[v] = {
            "costs": costs,
            "min_cost": (min_cost, min_struct)
        }
        return self.node_cost_cache[v]

    def cost_of_node(self, v):
        """
        computing the cost of encoding a supernode v
        :param v: a supernode
        :return: the cost
        """
        # compute which structure has the min encoding
        node_cost = 1

        if self.use_node_cost:
            corrections = self.compute_node_min_cost(v)
            correction_cost, struct = corrections['min_cost']
            node_cost += correction_cost

        edge_cost = 0
        if self.use_edge_cost:
            for neighbor_of_v in self.G[v]:
                if neighbor_of_v not in self.node_to_supernode:
                    edge_cost += self.cost_of_edge(v, neighbor_of_v)

        # # add node mapping cost
        # if v in self.supernode_to_node:
        #     cost += len(self.supernode_to_node[v])
        # else:
        #     cost += 1

        return node_cost, edge_cost

    def cost_of_merging_nodes(self, u, v):
        """
        computing the cost of encoding a supernode v
        :param v: a supernode
        :return: the cost
        """
        cost = 0

        # compute which structure has the min encoding
        nodes_of_u = self.supernode_to_node[u] if u in self.supernode_to_node else [
            u]
        nodes_of_v = self.supernode_to_node[v] if v in self.supernode_to_node else [
            v]
        nodes = list(nodes_of_u) + list(nodes_of_v)
        _, min_cost, min_struct = self.compute_min_cost_of_nodes(nodes)
        cost += min_cost

        nodes = set(nodes)
        neighbors = set(list(filter(lambda node: node not in nodes and node is not u and node is not v, list(
            self.G[u]) + list(self.G[v]))))
        for neighbor in neighbors:
            cost += self.cost_of_edge_if_merging(u, v, neighbor)

        cost += 1

        return cost

    def merge(self, u, v):
        """
        merge two supernode u and v into one supernode w
        1. add a new node w
        2. transfer nodes of u and v to w
        3. transfer links of u and v to w
        4. remove u and v if they are supernode
        :param u: a supernode
        :param v: a supernode
        :return: w, the new supernode
        """
        w = connect_nodes(u, v)
        # w = self.NODE_COUNT + 1
        # self.NODE_COUNT += 1
        self.G.add_node(w)
        self.supernode_to_node[w] = set()
        old_supernode_to_node = dict()
        old_supernode_to_node[u] = None
        old_supernode_to_node[v] = None
        old_neighbors = dict()
        old_neighbors[u] = set()
        old_neighbors[v] = set()

        # merge supernodes in supernode_to_node
        for x in [u, v]:
            neighbors_of_x = list(self.G[x])
            for neighbor_of_x in neighbors_of_x:
                old_neighbors[x].add(neighbor_of_x)
                if neighbor_of_x != u and neighbor_of_x != v:
                    self.G.add_edge(w, neighbor_of_x)
                # if neighbor_of_x == u or neighbor_of_x == v:
                #     self.G.add_edge(w, w)
                # else:
                #     self.G.add_edge(w, neighbor_of_x)
            if x in self.supernode_to_node:
                for node_of_x in self.supernode_to_node[x]:
                    self.supernode_to_node[w].add(node_of_x)
                    self.node_to_supernode[node_of_x] = w
                old_supernode_to_node[x] = self.supernode_to_node[x]
                del self.supernode_to_node[x]
                self.G.remove_node(x)
            else:
                self.supernode_to_node[w].add(x)
                self.node_to_supernode[x] = w

        def reverse_merge():
            """
            reverse the merge of u and v
            1. add back u and v
            2. restore nodes of u and v
            3. transfer links of u and v to w
            4. remove w
            :return:
            """
            # step 1: add back u and v
            self.G.add_nodes_from([u, v])
            for x in [u, v]:
                # step 2: restore nodes of u and v
                if old_supernode_to_node[x]:
                    self.supernode_to_node[x] = old_supernode_to_node[x]
                    for node_of_x in self.supernode_to_node[x]:
                        self.node_to_supernode[node_of_x] = x
                else:
                    if x in self.node_to_supernode:
                        del self.node_to_supernode[x]
                # step 3: transfer links of w to u and v
                for neighbor_of_x in old_neighbors[x]:
                    if not self.G.has_edge(x, neighbor_of_x):
                        self.G.add_edge(x, neighbor_of_x)

            # step 4: remove w
            self.G.remove_node(w)
            del self.supernode_to_node[w]

        return w, reverse_merge


    def s(self, u, v):
        """
        the ratio of the reduction in cost as a result of merging u and v (into a new supernode w),
        :param u: the source supernode
        :param v: the target supernode
        :return: the saved cost
        """
        if u in self.s_cache and v in self.s_cache[u]:
            return self.s_cache[u][v]
        node_cost_u, edge_cost_u = self.cost_of_node(u)
        node_cost_v, edge_cost_v = self.cost_of_node(v)
        # cost_w = self.cost_of_merging_nodes(u, v)
        w, reverse_merge = self.merge(u, v)
        node_cost_w, edge_cost_w = self.cost_of_node(w)
        reverse_merge()
        node_cost_u_and_v = node_cost_u + node_cost_v
        edge_cost_u_and_v = edge_cost_u + edge_cost_v - self.cost_of_edge(u, v)
        node_r = node_cost_u_and_v - node_cost_w
        edge_r = edge_cost_u_and_v - edge_cost_w
        cost_u_and_v = node_cost_u_and_v * WEIGHT_OF_PATTERN + edge_cost_u_and_v
        if cost_u_and_v == 0:
            return -1
        r = node_r * WEIGHT_OF_PATTERN + edge_r
        s = r / cost_u_and_v  # ratio
        self.s_cache[u] = self.s_cache[u] if u in self.s_cache else {}
        self.s_cache[v] = self.s_cache[v] if v in self.s_cache else {}
        self.s_cache[u][v] = self.s_cache[v][u] = s
        return s


def randomized(G, use_node_cost=True, use_edge_cost=True):
    """
    The Greedy Algorithm in Graph Summarization with Bounded Error
    :param SG: a super graph
    :param no_print: boolean
    :return:
    """
    SG = SuperGraph(G, use_node_cost=use_node_cost,
                    use_edge_cost=use_edge_cost)
    # SG.merge_same_role_nodes()

    def is_node_not_in_supernodes(n):
        return n not in SG.node_to_supernode

    F = []
    U = set(list(filter(is_node_not_in_supernodes, SG.G.nodes)))
    VS = set(list(filter(is_node_not_in_supernodes, SG.G.nodes)))

    while len(U) > 0:
        u = random.sample(U, 1)
        i = 0
        u = list(U)[i]
        max_s = -1
        max_v = None

        v_set = VS.copy()

        # # find all hop 2 nodes
        # v_set = set()
        # for hop_1_node in SG.G[u]:
        #     v_set.add(hop_1_node)
        #     for hop_2_node in SG.G[hop_1_node]:
        #         v_set.add(hop_2_node)

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

            # clear cache of s
            for x in [u, v]:
                for node in SG.s_cache[x]:
                    SG.s_cache[node] = {}
                SG.s_cache[x] = {}

            # update hop_2_nodes

        else:
            # U[i] = U[-1]
            # U.pop()
            U.remove(u)
            F.append(u)

    summarization = output(SG, G, VS)

    cost = 0
    for node in VS:
        node_cost, edge_cost = SG.cost_of_node(node)
        cost += node_cost + edge_cost / 2

    return summarization


def greedy(G, use_node_cost=True, use_edge_cost=True):
    """
    The Greedy Algorithm in Graph Summarization with Bounded Error
    :param G: networkx Graph
    :return:
    """
    SG = SuperGraph(G, use_node_cost=use_node_cost,
                    use_edge_cost=use_edge_cost)

    def is_node_not_in_supernodes(n):
        return n not in SG.node_to_supernode

    # SG.merge_same_role_nodes()

    # FIRST PART: initialization phase
    H = []
    VS = set(filter(is_node_not_in_supernodes, SG.G.nodes))
    traversed = nx.Graph()
    for node in VS:
        for hop_2_node in VS:
            if node is not hop_2_node and not traversed.has_edge(node, hop_2_node):
                traversed.add_edge(node, hop_2_node)
                s = SG.s(node, hop_2_node)
                if s > 0:
                    # use -1.0 because heapq is a min-heap
                    heapq.heappush(H, (-1.0 * s, (node, hop_2_node)))

    k = 0
    while len(H) > 0:
        k += 1
        # largest_pair = H[0][1]
        s, largest_pair = heapq.heappop(H)
        u, v = largest_pair

        # merge supernodes u and v and remove them from VS
        w, _ = SG.merge(u, v)
        VS.remove(u)
        VS.remove(v)
        VS.add(w)

        # clear cache of s
        for x in [u, v]:
            for node in SG.s_cache[x]:
                SG.s_cache[node] = {}
            SG.s_cache[x] = {}

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

    return summarization


def compute_cost(summarization):
    """
    compute the cost after compression
    :param SG: SuperGraph
    :param G: nx.Graph
    :param VS: set of supernodes
    :return: node-link-graph
    """
    origin_nodes = filter(lambda x: x['type']
                          != SUPERNODE, summarization['nodes'])
    origin_links = filter(lambda x: 'type' not in x or x['type']
                          != SUPERLINK, summarization['links'])
    G = nx.node_link_graph({"nodes": origin_nodes, "links": origin_links})

    supernodes = filter_super_nodes(summarization)
    super_node_cost = 0
    correction_cost = 0
    for supernode in supernodes:
        nodes = supernode['nodes']
        struct = supernode['encoding']
        subgraph = G.subgraph(nodes)
        edges = set(subgraph.edges)
        degrees = list(subgraph.degree)
        correction = correction_computations[struct](
            subgraph, nodes, edges, degrees)
        correction_cost += len(correction)
        super_node_cost += 1

    level_1_nodes = filter_level_1_nodes(summarization)
    super_link_cost = 0
    i = 0
    while i < len(level_1_nodes):
        j = i + 1
        while j < len(level_1_nodes):
            u, v = level_1_nodes[i], level_1_nodes[j]
            nodes_of_u = u['nodes'] if 'nodes' in u else [u['id']]
            nodes_of_v = v['nodes'] if 'nodes' in v else [v['id']]
            existing_edges_count, not_existing_edges_count = count_edges_or_not_between(
                G, nodes_of_u, nodes_of_v)
            A_uv = existing_edges_count
            Pi_uv = existing_edges_count + not_existing_edges_count
            if Pi_uv - A_uv + 1 > A_uv:
                correction_cost += A_uv
            else:
                correction_cost += Pi_uv - A_uv
                super_link_cost += 1

            j += 1
        i += 1

    return super_node_cost, super_link_cost, correction_cost


def output(SG, G, VS):
    """
    transform the summarized graph into a node-link-graph form
    :param SG: SuperGraph
    :param G: nx.Graph
    :param VS: set of supernodes
    :return: node-link-graph
    """
    nodes = []
    links = []
    traversed_nodes = set()
    traversed_links = set()
    node_dict = {}
    VS = list(VS)
    i = 0
    SG.node_cost_cache = {}
    while i < len(VS):
        u = VS[i]
        j = i + 1
        while j < len(VS):
            v = VS[j]
            existing_edges_count, not_existing_edges_count = SG.count_existing_or_not_edges(
                u, v)
            A_uv = existing_edges_count
            Pi_uv = existing_edges_count + not_existing_edges_count
            if Pi_uv > 0:
                if A_uv > (Pi_uv + 1) / 2:
                    superlink = {
                        "source": u,
                        "target": v,
                        "type": SUPERLINK,
                    }
                    links.append(superlink)
            j += 1

        if (u in SG.supernode_to_node) and (len(SG.supernode_to_node[u]) > 1):
            # u is a supernode
            node = {
                'id': u,
                'type': SUPERNODE,
                'nodes': list(SG.supernode_to_node[u]),
            }
            for v in node['nodes']:
                child = {
                    'id': v,
                    'type': NODE_IN_SUPERNODE
                }
                nodes.append(child)
                traversed_nodes.add(v)
                node_dict[v] = child

            corrections = SG.compute_node_min_cost(u)
            correction_cost, struct = corrections['min_cost']
            node['encoding'] = struct
            nodes.append(node)
            node_dict[node['id']] = node

        i += 1

    for v in G.nodes:
        if v not in traversed_nodes:
            node = {
                'id': v,
                'type': INDEPENDENT_NODE
            }
            nodes.append(node)
            node_dict[node['id']] = node
            traversed_nodes.add(v)

    for e in G.edges:
        source = e[0]
        target = e[1]
        if (source, target) not in traversed_links and (target, source) not in traversed_links:
            link = {
                'source': source,
                'target': target
            }
            traversed_links.add((source, target))
            links.append(link)

    return {
        'nodes': nodes,
        'links': links
    }
