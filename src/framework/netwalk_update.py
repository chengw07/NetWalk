"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wenchao Yu <yuwenchao@ucla.edu>
    Affiliation: NEC Labs America
"""
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix


class Reservior:
    """
    class maintains a sketch of the dynamic graph
    """

    def __init__(self, edges, vertices, dim=10, seed=24):
        self.reservior = {}
        self.degree = {}
        self.init_edges = edges
        self.vertices = vertices
        self.reservior_dim = dim
        self.seed = seed
        self.__build()

    def __build(self):
        """
        construct initial reservior using the inital graph (edge list)
        :return:
        """
        # TODO: check matlab code
        # TODO: currently implementation is for undirected graph
        g = nx.Graph()
        g.add_edges_from(self.init_edges)
        for v in self.vertices:
            if v in g:
                nbrs = list(g.neighbors(v))
                np.random.seed(self.seed)
                indices = np.random.randint(len(nbrs), size=self.reservior_dim)
                self.reservior[v] = np.array([nbrs[idx] for idx in indices])
                self.degree[v] = len(nbrs)
            else:
                self.reservior[v] = np.array([None] * self.reservior_dim)
                self.degree[v] = 0

    def update(self, edges):
        """
        update the reservior based on the incoming edge(s)
        :param edges: new incoming edges
        :return:
        """
        assert len(edges)
        for edge in tqdm(edges):
            u, v = edge

            # update u's reservior, edges can not be duplicated in the training and updating
            assert v not in self.reservior[u]

            self.degree[u] += 1
            indices = np.random.randint(self.degree[u], size=self.reservior_dim)
            replace_idx = np.where(indices == self.degree[u] - 1)
            self.reservior[u][replace_idx] = v

            # update v's reservior, edges can not be duplicated in the training and updating
            assert u not in self.reservior[v]

            self.degree[v] += 1
            indices = np.random.randint(self.degree[v], size=self.reservior_dim)
            replace_idx = np.where(indices == self.degree[v] - 1)
            self.reservior[v][replace_idx] = u


class WalkUpdate:
    """WalkUpdate update the Reservior and generate new batch of walks.

        """
    def __init__(self, init_edges, vertices, walk_len=3, walk_per_node=5, prev_percent=1, seed=24):
        self.init_edges = init_edges
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        self.prev_percent = prev_percent
        self.seed = seed

        self.reservior = Reservior(edges=self.init_edges, vertices=vertices)

        # previous walks
        self.prev_walks = self.__init_walks()

        # new generated walks
        self.new_walks = None
        # training walks = new walks + "percent" * old walks
        self.training_walks = None

    def __init_walks(self):
        """
        Initial walks generated from random walking
        :return:
        """
        g = nx.Graph()
        g.add_edges_from(self.init_edges)
        rand = random.Random(self.seed)
        walks = []
        nodes = list(g.nodes())
        for cnt in range(self.walk_per_node):
            rand.shuffle(nodes)
            for node in nodes:
                walks.append(self.__random_walk(g, node, rand=rand))
        return walks

    def __random_walk(self, g, start, alpha=0, rand=random.Random(0)):
        """
        return a truncated random walk
        :param alpha: probability of restars
        :param start: the start node of the random walk
        :return:
        """
        walk = [start]

        while len(walk) < self.walk_len:
            cur = walk[-1]
            if len(list(g.neighbors(cur))) > 0:
                if rand.random() >= alpha:
                    walk.append(rand.choice(list(g.neighbors(cur))))
                else:
                    walk.append(walk[0])
            else:
                break
        return walk

    def __generate(self, new_edges, update_type="randomwalk"):
        walk_set = []
        rand = random.Random(self.seed)

        ## using random walk in the reservior for updating new set of walks for training
        # it's slower but very accurate, it is probabilistically equal to do random walk
        # in the whole graph with all edges so far arrived
        if update_type == "randomwalk":
            start_node = []
            for edge in new_edges:
                u, v = edge
                start_node.append(u)
                start_node.append(v)
            for n in set(start_node):
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[n])
                    y = rand.choice(self.reservior.reservior[x])
                    walk_set.append([n, x, y])

            self.new_walks = walk_set
            old_samples = [w for w in self.prev_walks if w[0] not in start_node]
            self.training_walks = self.new_walks + old_samples
            print("length of training walks: %d" % len(self.training_walks))
            self.prev_walks = walk_set + old_samples

            return

        # current implementation works for len = 3 and 4
        assert self.walk_len < 5

        # decide the number of walk per node???
        for edge in new_edges:
            u, v = edge
            if self.walk_len == 3:

                ################
                walk_set = [x for x in walk_set if x[0] != v]
                walk_set = [x for x in walk_set if x[0] != u]


                # walk u-v-x
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[v])
                    walk_set.append([u, v, x])

                # walk v-u-x
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[u])
                    walk_set.append([v, u, x])

                ################
                self.prev_walks = [x for x in self.prev_walks if x[0] != v]
                self.prev_walks = [x for x in self.prev_walks if x[0] != u]



            elif self.walk_len == 4:
                # walk u-v-xy
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[v])
                    y = rand.choice(self.reservior.reservior[x])
                    walk_set.append([u, v, x, y])

                # walk v-u-xy
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[u])
                    y = rand.choice(self.reservior.reservior[x])
                    walk_set.append([v, u, x, y])

                # walk x-u-v-y
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[u])
                    y = rand.choice(self.reservior.reservior[v])
                    walk_set.append([x, u, v, y])

                # walk x-v-u-y
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[v])
                    y = rand.choice(self.reservior.reservior[u])
                    walk_set.append([x, v, u, y])
            else:
                pass

        self.new_walks = walk_set
        old_samples = self.prev_walks#random.sample(self.prev_walks, int(self.prev_percent * len(self.prev_walks)))
        self.training_walks = self.new_walks + old_samples
        self.prev_walks = self.training_walks   #self.prev_walks.extend(walk_set)

    def update(self, new_edges):
        """
        Updating reservior and generate new set of walks for re-training using newly come edges
        :param new_edges: newly arrived edges
        :return: new set of training walks
        """
        # update reservior
        self.reservior.update(new_edges)

        ## if reconduct randomwalk then the new set of walks are probabilistically equal to conducting
        ## randomwalk in the graph with all edges so far, it's slower than approximated method
        self.__generate(new_edges, update_type="randomwalk")

        return self.training_walks


class NetWalk_update:
    """
    Preparing both training initial graph walks and testing list of walks in each snapshot
    """
    def __init__(self, path, walk_per_node=5, walk_len=3, init_percent=0.5, snap=10):
        """
        Initialization of data preparing
        :param path: Could be either edge list file path, or the tuple including training and testing edge lists
        :param walk_per_node: for each node, how many walks start from the node to be sampled
        :param walk_len: the length of walk
        :param init_percent: initial percentage of edges sampled for training
        :param snap: number of edges in each time snapshot for updating
        """
        self.data_path = path
        self.walk_len = walk_len
        self.init_percent = init_percent
        self.snap = snap
        self.vertices = None
        self.idx = 0
        self.walk_per_node = walk_per_node

        if isinstance(path, str):
            self.data = self.__get_data()
        else:
            test = path[0][:, 0:2]
            train = path[1]
            self.data = self.__get_data_mat(test, train)


        init_edges, snapshots = self.data
        self.walk_update = WalkUpdate(init_edges, self.vertices, walk_len=self.walk_len, walk_per_node=self.walk_per_node, prev_percent=1, seed=24)

    def __get_data_mat(self, test, train):
        """
        Generate initial walk list for training and list of walk lists in each upcoming snapshot
        :param test: edge list for testing
        :param train: edge list for training
        :return: initial walk list for training and list of walk lists in each upcoming snapshot
        """
        edges = np.concatenate((train, test), axis=0)
        self.vertices = np.unique(edges)
        init_edges = train
        print("total edges: %d, initial edges: %d; total vertices: %d"
              % (len(edges), len(init_edges), len(self.vertices)))

        snapshots = []
        current = 0
        while current < len(test):
            # if index >= len(edges), equals to edges[current:]
            # length of last snapshot <= self.snap
            snapshots.append(test[current:current + self.snap])
            current += self.snap
        print("number of snapshots: %d, edges in each snapshot: %d" % (len(snapshots), self.snap))
        data = (init_edges, snapshots)
        return data

    def __get_data(self):
        """
        Generate initial walk list for training and list of walk lists in each upcoming snapshot
        :return: initial walk list for training and list of walk lists in each upcoming snapshot
        """
        edges = np.loadtxt(self.data_path, dtype=int, comments='%')
        self.vertices = np.unique(edges)
        init_idx = int(len(edges) * self.init_percent)
        init_edges = edges[:init_idx]
        print("total edges: %d, initial edges: %d; total vertices: %d"
              % (len(edges), len(init_edges), len(self.vertices)))

        snapshots = []
        current = init_idx
        while current < len(edges):
            # if index >= len(edges), equals to edges[current:]
            # length of last snapshot <= self.snap
            snapshots.append(edges[current:current + self.snap])
            current += self.snap
        print("number of snapshots: %d, edges in each snapshot: %d" % (len(snapshots), self.snap))
        data = (init_edges, snapshots)
        return data

    def run(self):
        """
        perform netwalk program with input data
        :return:
        """
        init_edges, snapshots = self.data
        walk_update = WalkUpdate(init_edges, self.vertices, self.walk_per_node)
        # call netwalk core
        # save embedding
        # save model
        for snapshot in snapshots:
            training_set = walk_update.update(snapshot)

            self.getOnehot(training_set)
            print(training_set)
            # load model
            # update embeddings

    def getNumsnapshots(self):
        """
        get number of total time snapshots
        :return: The number of total time snapshots
        """
        init_edges, snapshots = self.data
        return len(snapshots)

    def nextOnehotWalks(self):
        """
        get next list of walks for re-training in next time snapshot
        :return: next list of walks for re-training in next time snapshot
        """
        if not self.hasNext():
            return False
        _, snapshots = self.data
        snapshot = snapshots[self.idx]
        self.idx += 1
        training_set = self.walk_update.update(snapshot)

        return self.getOnehot(training_set)

    def hasNext(self):
        """
        Checking if still has next snapshot
        :return: true if has, or return false
        """
        init_edges, snapshots = self.data
        if self.idx >= len(snapshots):
            return False
        else:
            return True

    def getInitWalk(self):
        """
        Get inital walk list
        :return: list of walks for initial training
        """
        walks = self.walk_update.prev_walks
        return self.getOnehot(walks)


    def getOnehot(self, walks):
        """
        transform walks with id number to one-hot walk list
        :param walks: walk list
        :return: one-hot walk list
        """
        walk_mat = np.array(walks, dtype=int) - 1
        rows = walk_mat.flatten()
        cols = np.array(range(len(rows)))
        data = np.array([1] * (len(rows)))

        coo = coo_matrix((data, (rows, cols)), shape=(len(self.vertices), len(rows)))
        onehot_walks = csr_matrix(coo)
        return onehot_walks.toarray()



# demo with dataset karate and emails
# data without weight, undirected graph, no duplicate edges
if __name__ == "__main__":
    data_path = "../../data/karate.edges"#""../../data/karate/karate.edgelist"
    netwalk = NetWalk_update(data_path)
    netwalk.run()
