"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com> & Wenchao Yu
    Affiliation: NEC Labs America
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

from framework.imports import *
import framework.Model as MD
from framework.anomaly_generation import anomaly_generation
from framework.load_email_eu import load_email_eu
from framework.anomaly_detection import anomaly_detection
from framework.anomaly_detection_stream import anomaly_detection_stream

from datetime import datetime
import tensorflow as tf
import numpy as np
import plots.DynamicUpdate as DP

import warnings
from framework.netwalk_update import NetWalk_update

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


def static_process(args):

    sample_rate = 1     #0.5

    # load emai_eu data
    data, n, m = load_email_eu(args.input, sample_rate)


    # STEP 0: Parameters
    hidden_size = args.representation_size   # size of hidden codes to learn, default is 20

    activation = tf.nn.sigmoid

    dimension = [n, hidden_size]

    rho = 0.5                           # sparsity ratio
    lamb = 0.0017                       # weight decay
    beta = 1                            # sparsity weight
    gama = 340                          # autoencoder weight
    walk_len = args.walk_length
    epoch = 30                          # number of epoch for optimizing, could be larger
    batch_size = 40                     # should be smaller or equal to args.number_walks*n
    learning_rate = 0.01                # learning rate, for adam, using 0.01, for rmsprop using 0.1
    optimizer = "adam"                  #"rmsprop"#"gd"#"rmsprop" #"""gd"#""lbfgs"
    corrupt_prob = [0]                  # corrupt probability, for denoising AE
    ini_graph_percent = args.init_percent# percent of edges in the initial graph
    anomaly_percent = 0.2               # percentage of anomaly edges in the testing edges
    alfa = 0.01                         # updating parameter for online k-means to update clustering centroids
    k = 3                               # number of clusters for kmeans to clustering edges



    # STEP 1: Preparing data: training data and testing list of edges(for online updating)
    synthetic_test, train_mat, train = anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m)
    data_zip = []
    data_zip.append(synthetic_test)
    data_zip.append(train)
    # generating initial training walks
    netwalk = NetWalk_update(data_zip, walk_per_node=args.number_walks, walk_len=args.walk_length,
                             init_percent=args.init_percent, snap=args.snap)
    ini_data = netwalk.getInitWalk()

    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                        epoch, batch_size, learning_rate, optimizer, corrupt_prob)

    # STEP 2: Learning initial embeddings for training edges
    embedding = getEmbedding(embModel, ini_data, n)

    # dynaimically plot the anomaly score over different snapshots
    d_plot = DP.DynamicUpdate()


    # conduct anomaly detection using first snapshot of testing edges
    scores, auc, n0, c0, res, ab_score = anomaly_detection(embedding, train, synthetic_test[0:args.snap, :], k)

    print('initial auc of anomaly detection:', auc)
    print('initial anomaly score:', ab_score)

    # visualize anomaly score
    d_plot.addPoint(1, ab_score)

    # STEP 3: over different snapshots of edges, dynamically updating embeddings of nodes and conduct
    #         online anomaly detection for edges, visualize the anomaly score of each snapshot
    snapshotNum = 1
    while(netwalk.hasNext()):
        snapshot_data = netwalk.nextOnehotWalks()
        embedding = getEmbedding(embModel, snapshot_data, n)
        if netwalk.hasNext():
            if len(synthetic_test) > args.snap*(snapshotNum+1):
                test_piece = synthetic_test[args.snap*snapshotNum:args.snap*(snapshotNum+1), :]
            else:
                test_piece = synthetic_test[args.snap * snapshotNum:, :]
                #return
        else:
            return

        # online anomaly detection, each execution will update the clustering center
        scores, auc, n0, c0, res, ab_score = anomaly_detection_stream(embedding, train, test_piece, k, alfa, n0, c0)


        print('auc of anomaly detection at snapshot %d: %f'  % (snapshotNum, auc))
        print('anomaly score at snapshot %d: %f' % (snapshotNum, ab_score))

        snapshotNum += 1

        # visualizing anomaly score of current snapshot
        d_plot.addPoint(snapshotNum, ab_score)


def getEmbedding(model, data, n):
    """
        function getEmbedding(model, data, n)
        #  the function feed ''data'' which is a list of walks
        #  the embedding ''model'', n: the number of total nodes
        return: the embeddings of all nodes, each row is one node, each column is one dimension of embedding
    """
    # batch optimizing to fit the model
    model.fit(data)

    # Retrieve the embeddings
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)
    return res


def main():
    parser = ArgumentParser("NETWALK", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--format', default='adjlist',
                        help='File format of input file')

    parser.add_argument('--snap', default=100,
                        help='number of edges in each snapshot')

    parser.add_argument('--init_percent', default=0.8,
                        help='percentage of edges in initial graph')

    parser.add_argument('--input', nargs='?', default='../data/email-Eu-core-sub.txt',  # required=True,
                        help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                        help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--number-walks', default=20, type=int,
                        help='Number of random walks to start at each node')

    parser.add_argument('--output', default='./tmp/embedding.txt',  # required=True,
                        help='Output representation file')

    parser.add_argument('--representation-size', default=20, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=24, type=int,
                        help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                        help='Treat graph as undirected.')

    parser.add_argument('--walk_length', default=3, type=int,
                        help='Length of the random walk started at each node')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    static_process(args)


if __name__ == "__main__":
    main()
