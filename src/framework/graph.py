"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""

"""Graph utilities."""
import numpy as np
import logging
from io import open
from time import time
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from itertools import product, permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import cpu_count

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
logger = logging.getLogger("netwalk")

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(G.keys())]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def generate_walk(G, num_walks, walk_len, alpha=0, rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(walk_len, rand=rand, alpha=alpha, start=node))
    walk_mat = np.array(walks, dtype=int) - 1
    rows = walk_mat.flatten()
    cols = np.array(range(len(nodes) * walk_len * num_walks))
    data = np.array([1] * (len(nodes) * walk_len * num_walks))
    coo = coo_matrix((data, (rows, cols)))
    onehot_walks = csr_matrix(coo)
    return (walks, onehot_walks)

def clique(size):
    return from_adjlist(permutations(range(1, size + 1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):
    if unchecked:
        parse_func = parse_adjacencylist_unchecked
        convert_func = from_adjlist_unchecked
    else:
        parse_func = parse_adjacencylist
        convert_func = from_adjlist

    adjlist = []

    t0 = time()

    with open(file_) as f:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            total = 0
            for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
                adjlist.extend(adj_chunk)
                total += len(adj_chunk)

    t1 = time()

    logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1 - t0))

    t0 = time()
    G = convert_func(adjlist)
    t1 = time()

    logger.info('Converted edges to graph in {}s'.format(t1 - t0))

    if undirected:
        t0 = time()
        G = G.make_undirected()
        t1 = time()
        logger.info('Made graph undirected in {}s'.format(t1 - t0))

    return G


def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]

    return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G
