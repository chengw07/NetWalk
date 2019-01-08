"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering


def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train_mat: the training network with square matrix format, the training
                        network edges for initial model training;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    np.random.seed(1)
    print('[#s] generating anomalous dataset...\n', datetime.datetime.now())
    print('[#s] initial network edge percent: #.1f##, anomaly percent: #.1f##.\n', datetime.datetime.now(),
          ini_graph_percent * 100, anomaly_percent * 100)

    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    #data to adjacency_matrix
    adjacency_matrix = edgeList2Adj(data)

    # clustering nodes to clusters using spectral clustering
    kk = 3#10#42#42
    sc = SpectralClustering(kk, affinity='precomputed', n_init=100, assign_labels = 'discretize')
    labels = sc.fit_predict(adjacency_matrix)


    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    generate_edges = np.concatenate((idx_1, idx_2), axis=1)

    ####### genertate abnormal edges ####
    fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    fake_edges = processEdges(fake_edges, data)


    anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    anomalies = fake_edges[0:anomaly_num, :]

    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!

    #anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

    anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200

    idx_test[anomaly_pos] = 1
    synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)

    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)

    synthetic_test[idx_anomalies, 0:2] = anomalies
    synthetic_test[idx_normal, 0:2] = test

    train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] - 1, train[:, 1] - 1)),
                           shape=(n, n))
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n) #TODO: node addition
    train_mat = train_mat + train_mat.transpose()

    return synthetic_test, train_mat, train



def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

    tmp = fake_edges[idx_fake]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]

    fake_edges[idx_fake] = tmp

    idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)

    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []

    for i in a:
        if i not in b:
            c.append(i)
    fake_edges = np.array(c)
    return fake_edges

def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = max(max(user, item) for user, item in data)  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
        matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
    return matrix

if __name__ == "__main__":
    data_path = "../../data/karate.edges"

    edges = np.loadtxt(data_path, dtype=int, comments='%')
    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)

    synthetic_test, train_mat, train = anomaly_generation(0.5, 0.2, edges, n, m)

    print(train)
