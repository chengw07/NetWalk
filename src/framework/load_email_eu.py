"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import numpy as np
def load_email_eu(data_path, sample_rate):
    """ function load_uci_message
    #  [data, n, m] = load_uci_message(sample_rate)
    #  load data set uci_message and preprocess it
    #  parameter: sample_rate to subsample edges
    #  return data: network(each row is one edge); n: number of total nodes; m:
    #  number of edges
    #  Example:
    #      input orginal file format as following: (nodeID, nodeID)
    #        1 2
    #        3 4
    #        5 2
    #        9 14
    #      output: data, n, m
    #          data is with format (nodeID, nodeID), each row is one edge
    #          n: number of total nodes; m: number of edges
    #   Copyright 2018 NEC Labs America, Inc.
    #   $Revision: 1.0 $  $Date: 2018/10/26 17:46:36 $
    """
    edges = np.loadtxt(data_path, dtype=int, comments='%') + 1

    # change to undirected graph
    idx_reverse = np.nonzero(edges[:, 0] - edges[:, 1] > 0)
    tmp = edges[idx_reverse]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]
    edges[idx_reverse] = tmp

    # remove self-loops
    idx_remove_dups = np.nonzero(edges[:, 0] - edges[:, 1] < 0)
    edges = edges[idx_remove_dups]

    edges = edges[:, 0:2]


    # only keep unique edges
    edges, ind_ = np.unique(edges, axis=0, return_index=True)


    step = int(np.floor(1/sample_rate))
    edges = edges[0:len(edges):step, :]


    # re-assign id
    unique_id = np.unique(edges)
    n = len(unique_id)
    _, digg = ismember(edges, unique_id)

    data = digg
    m = len(digg)

    np.random.seed(101)
    np.random.shuffle(data)

    return data, n, m

def ismember(a, b_vec):
    """ MATLAB equivalent ismember function """

    shape_a = a.shape

    a_vec = a.flatten()

    bool_ind = np.isin(a_vec, b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv = np.unique(common, return_inverse=True)  # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    flag = bool_ind.reshape(shape_a)
    content = (common_ind[common_inv]).reshape(shape_a) + 1

    return flag, content
