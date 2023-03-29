
# NODE TYPES
SUPERNODE = 'SUPERNODE'  # supernodes
NODE_IN_SUPERNODE = 'NODE_IN_SUPERNODE'  # nodes in supernodes
INDEPENDENT_NODE = 'INDEPENDENT_NODE'  # nodes not in supernodes

# LINK TYPES
SUPERLINK = 'SUPERLINK'  # superlinks
LINK_IN_SUPERLINK = 'LINK_IN_SUPERLINK'  # links in superlinks
LINK_IN_SUPERNODE = 'LINK_IN_SUPERNODE'  # links in supernodes
# negative correction links in superlinks
NEG_LINK_IN_SUPERLINK = 'NEG_LINK_IN_SUPERLINK'
# positive correction links in superlinks
POS_LINK_IN_SUPERLINK = 'POS_LINK_IN_SUPERLINK'
# negative correction links in supernodes
NEG_LINK_IN_SUPERNODE = 'NEG_LINK_IN_SUPERNODE'
# positive correction links in supernodes
POS_LINK_IN_SUPERNODE = 'POS_LINK_IN_SUPERNODE'
INDEPENDENT_LINK = 'INDEPENDENT_LINK'  # links not in superlinks & supernodes

ORIGIN_NODE_TYPES = set([NODE_IN_SUPERNODE, INDEPENDENT_NODE])
ORIGIN_LINK_TYPES = set([LINK_IN_SUPERLINK, INDEPENDENT_LINK,
                         POS_LINK_IN_SUPERLINK, POS_LINK_IN_SUPERNODE])

S = 'star'
C = 'clique'
BC = 'bipartite_core'
CH = 'chain'
E = 'none'
WH = "wheel"
MS = "mesh"
TR = "tree"
PR = "Prism"
