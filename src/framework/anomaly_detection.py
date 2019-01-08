"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import datetime
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


def anomaly_detection(embedding, train, synthetic_test, k):
    """
    function anomaly_detection_stream(embedding, train, synthetic_test, k, alfa, n0, c0)
    #  the function generate codes of edges by combining embeddings of two
    #  nodes, and then using the testing codes of edges for anomaly detection
    #  Input: embedding: embeddings of each node; train: training edges; synthetic_test: testing edges with anomlies;
                k: number of clusters
    #  return scores: The anomaly severity ranking, the top-ranked are the most likely anomlies
    #   auc: AUC score
    #   n:   number of nodes in each cluster
    #   c:   cluster centroids,
    #   res: id of nodes if their distance to nearest centroid is larger than that in the training set
    #   ab_score: anomaly score for the whole snapshot, just the sum of distances to their nearest centroids
    """

    print('[#s] edge encoding...\n', datetime.datetime.now())

    src = embedding[train[:, 0] - 1, :]
    dst = embedding[train[:, 1] - 1, :]
    test_src = embedding[synthetic_test[:, 0] - 1, :]
    test_dst = embedding[synthetic_test[:, 1] - 1, :]

    # the edge encoding
    # refer Section 3.3 Edge Encoding in the KDD paper for details
    encoding_method = 'Hadamard'
    if encoding_method == 'Average':
        codes = (src + dst) / 2
        test_codes = (test_src + test_dst) / 2
    elif encoding_method == 'Hadamard':
        codes = np.multiply(src, dst)
        test_codes = np.multiply(test_src, test_dst)
    elif encoding_method == 'WeightedL1':
        codes = abs(src - dst)
        test_codes = abs(test_src - test_dst)
    elif encoding_method == 'WeightedL2':
        codes = (src - dst) ** 2
        test_codes = (test_src - test_dst) ** 2

    print('[#s] anomaly detection...\n', datetime.datetime.now())

    # conducting k-means clustering and recording centroids of different
    # clusters

    kmeans = KMeans(n_clusters=k)
    # Fitting the input data
    kmeans = kmeans.fit(codes)
    # Getting the cluster labels
    indices = kmeans.predict(codes)
    # Centroid values
    centroids = kmeans.cluster_centers_

    # [indices, centroids] = kmeans(codes, k)

    # tbl = tabulate(indices)
    # c = dict.fromkeys(indices, 0)
    #
    # for x in indices:
    #     c[x] += 1

    tbl = Counter(indices)

    n = list(tbl.values())

    c = centroids

    assert (len(n) == k)


    labels = synthetic_test[:, 2]

    # calculating distances for testing edge codes to centroids of clusters
    dist_center = cdist(test_codes, c)

    # assinging each testing edge code to nearest centroid
    min_dist = np.min(dist_center, 1)

    # sorting distances of testing edges to their nearst centroids
    scores = min_dist.argsort()
    scores = scores[::-1]


    #calculating auc score of anomly detection task, in case that all labels are 0's or all 1's
    if np.sum(labels) == 0:
        labels[0] = 1
    elif np.sum(labels) == len(labels):
        labels[0] = 0

    auc = roc_auc_score(labels, min_dist)

    # calculating distances for testing edge codes to centroids of clusters
    dist_center_tr = cdist(codes, c)
    min_dist_tr = np.min(dist_center_tr, 1)
    max_dist_tr = np.max(min_dist_tr)
    res = [1 if x > max_dist_tr else 0 for x in min_dist]
    #ab_score = np.sum(res)/(1e-10 + len(res))
    ab_score = np.sum(min_dist) / (1e-10 + len(min_dist))

    return scores, auc, n, c, res, ab_score


if __name__ == "__main__":
    embedding = np.array([[0.1, 0], [0.11, 0], [0, 0.6], [0, 0.61], [0.12, 0]])
    train = np.array([[1, 2], [3, 4],[1,5]])
    synthetic_test = np.array([[2, 5, 0], [1, 3, 1], [2, 4, 1]])
    k = 2
    scores, auc, n, c = anomaly_detection(embedding, train, synthetic_test, k)
    print(auc)
