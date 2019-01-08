"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com> & Wenchao Yu
    Affiliation: NEC Labs America
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import matplotlib.pyplot as plt

from framework.imports import *
import framework.Model as MD

from datetime import datetime
import tensorflow as tf
import numpy as np
import networkx as nx

import warnings
from plots.viz_stream import viz_stream

from framework.netwalk_update import NetWalk_update

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


def static_process(args):


    # STEP 0: Parameters
    hidden_size = args.representation_size     # size of hidden codes to learn, default is 20

    activation = tf.nn.sigmoid

    rho = 0.5           # sparsity ratio
    lamb = 0.0017       # weight decay
    beta = 1            # sparsity weight
    gama = 340          # autoencoder weight
    walk_len = args.walk_length
    epoch = 400
    batch_size = 20     # number of epoch for optimizing, could be larger
    learning_rate = 0.1 # learning rate, for adam, using 0.01, for rmsprop using 0.1
    optimizer = "rmsprop"#"gd"#"rmsprop" #""lbfgs"#"rmsprop"#"adam"#"gd"#""lbfgs"#"adam"#
    corrupt_prob = [0]  # corrupt probability, for denoising AE

    # STEP 1: Preparing data: training data and testing list of edges(for online updating)
    data_path = args.input
    netwalk = NetWalk_update(data_path, walk_per_node=args.number_walks, \
                             walk_len=args.walk_length, init_percent=args.init_percent, snap=args.snap)
    n = len(netwalk.vertices) # number of total nodes

    print("{} Number of nodes: {}".format(print_time(), n))
    print("{} Number of walks: {}".format(print_time(), args.number_walks))
    print("{} Data size (walks*length): {}".format(print_time(), args.number_walks*args.walk_length))
    print("{} Generating network walks...".format(print_time()))
    print("{} Clique embedding training...".format(print_time()))

    dimension = [n, hidden_size]

    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                        epoch, batch_size, learning_rate, optimizer, corrupt_prob)


    init_edges, snapshots = netwalk.data
    data = netwalk.getInitWalk()

    fig = plt.figure(figsize=(12, 12))



    # STEP 2: Learning initial embeddings for training edges
    embedding_code(embModel, data, n, args)

    # load karate club graph
    G = nx.karate_club_graph()
    edge_list = G.edges()

    # list of initial edge list tuples
    tuples = tuple(map(tuple, init_edges - 1))

    # complementary set of edges for initial edges
    rm_list = [x for x in edge_list if x not in tuples]

    # visualize initial embedding
    viz_stream(rm_list, fig, 5, 2, 1)

    # STEP 3: over different snapshots of edges, dynamically updating embeddings of nodes and conduct
    #         online anomaly detection for edges, visualize the anomaly score of each snapshot
    snapshotNum = 0
    while(netwalk.hasNext()):
        data = netwalk.nextOnehotWalks()
        tuples = tuple(map(tuple, snapshots[snapshotNum] - 1)) + tuples
        snapshotNum += 1
        embedding_code(embModel, data, n, args)
        rm_list = [x for x in edge_list if x not in tuples]
        viz_stream(rm_list, fig, 5, 2, snapshotNum+1)

    plt.show()

    print("finished")





def embedding_code(model, data, n, args):
    """
            function embedding_code(model, data, n, args)
            #  the function feed ''data'' which is a list of walks
            #  the embedding ''model'', n: the number of total nodes
            return: the embeddings of all nodes, each row is one node, each column is one dimension of embedding
        """
    # STEP 2:  optimizing to fit parameter learning
    model.fit(data)

    # STEP 3: retrieve the embeddings
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)

    ids = np.transpose(np.array(range(n)))

    ids = np.expand_dims(ids, axis=1)

    embeddings = np.concatenate((ids, res), axis=1)

    # STEP 4: save results
    np.savetxt(args.output, embeddings, fmt="%g")
    print("{} Done! Embeddings are saved in \"{}\"".format(print_time(), args.output))
    # print embeddings


def main():
    parser = ArgumentParser("NETWALK", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--snap', default=10,
                        help='number of edges in each snapshot')

    parser.add_argument('--init_percent', default=0.5,
                        help='percentage of edges in initial graph')

    parser.add_argument('--input', nargs='?', default='../data/karate.edges',  # required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--number_walks', default=5, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', default='./tmp/embedding.txt',  # required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=2, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=24, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--walk-length', default=3, type=int,
                        help='Length of the random walk started at each node')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    static_process(args)



if __name__ == "__main__":
    main()
