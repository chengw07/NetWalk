"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
from scipy.sparse import csgraph
import numpy as np
from framework import DenoisingAE
import tensorflow as tf
import random
from tqdm import tqdm

class Model:
    def __init__(self, activation, dimension, walk_len, nodeNum, gama, lamb,
                 beta, rho, epoch, batch_size, learning_rate, optimizer_type, corrupt_prob):


        self.activation = activation
        self.corrupt_prob_value = corrupt_prob

        self.activation = activation
        self.optimized = False
        self.dimension = dimension
        self.walk_len = walk_len
        self.gama = gama
        self.lamb = lamb
        self.beta = beta
        self.rho = rho
        self.epoch = epoch
        self.batch_size = batch_size*self.walk_len
        self.learning_rate = learning_rate

        self.nodeNum = nodeNum

        self.optimizer_type = optimizer_type

        self.data = tf.placeholder(tf.int32, shape=[self.nodeNum, None],#self.batch_size], #
                                   name='data')
        self.corrupt_prob = tf.placeholder(tf.float32, [1])
        self.loss, self.clique_loss, self.ae_loss, self.kl_loss, self.self_weight_decay_J = self.clique_embedding_loss()


        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        elif optimizer_type == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                self.loss)
        else:
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method="L-BFGS-B",
                                                               options={'maxiter': 200, 'disp': 0})

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()

    def feedforward_autoencoder(self, data):
        if not self.optimized:
            print('Need training first!')
            return None
        current_input = data.astype(np.float32)  # ((scipy.sparse.coo_matrix)(data)).todense().astype(np.float32)

        with self.sess.as_default():
            code = tf.transpose(self.encoder_out)
            res = code.eval(feed_dict={self.data: current_input, self.corrupt_prob: self.corrupt_prob_value})
            return np.array(res)
        # for i in range(len(self.b_list_value)):
        #     W = self.W_list_value[i]
        #     b = self.b_list_value[i]
        #     output = self.sigmoid(np.transpose(np.matmul(W, current_input)) + b)
        #     current_input = output
        return current_input

    def sigmoid(self, x, derivative=False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm

    def clique_embedding_loss(self):

        result = DenoisingAE.autoencoder(self.data, self.corrupt_prob, self.dimension, self.beta, self.rho, self.activation,
                                self.lamb, self.gama)

        self.encoder_out = result['encoder_out']

        ae_cost = result['cost']

        phi = np.ones((self.walk_len, self.walk_len)) - np.eye(self.walk_len)
        L = tf.cast(csgraph.laplacian(phi, normed=False), tf.float32)


        trans_code = tf.transpose(self.encoder_out)
        trans_code = tf.reshape(trans_code, [-1, self.walk_len, self.dimension[-1]])
        t_trans_code = tf.transpose(trans_code, [0,2,1])

        left = tf.einsum('aij,jk->aik', t_trans_code, L)

        mul = tf.einsum('aij,ajk->aik', left, trans_code)
        trace_mul = tf.trace(mul)
        clique_J = tf.reduce_mean(trace_mul)


        # for i in range(num_walks):
        #     f_i = self.encoder_out[:, i * self.walk_len:(i + 1) * self.walk_len]
        #     clique_J += tf.reduce_mean(tf.linalg.diag_part(tf.matmul(tf.matmul(f_i, L), tf.transpose(f_i))))
        #     #tf.trace(tf.matmul(tf.matmul(f_i, L), tf.transpose(f_i)))

        clique_loss = clique_J #(self.walk_len / m) *
        loss = clique_loss + ae_cost
        self.optimized = True


        # self.W_list = result['W_list']
        # self.b_list = result['b_list']


        return loss, clique_loss, result['ae_loss'], result['kl_loss'], result['weight_decay_J']


    def print_loss(self, loss_evaled, cl, ae, kl, weight_loss):
        print(loss_evaled, " cl:", cl, " ae:", ae, " kl:", kl, " l2_regularizer:", weight_loss)

    def batchify(self, data, bsz, shuffle=False):
        """
        :param data: training walks sets, that is a list of node walk chain, each chain is a list of nodes
        :param bsz: batch size
        :param shuffle: indicator of if reshuffle training set then split it to batches
        :param gpu: if conduct in gpu
        :return: batches of training samples(walks)
        """
        if shuffle:
            random.shuffle(data)
        nbatch = data.shape[1] // bsz
        batches = []

        for i in range(nbatch):
            # Pad batches to maximum sequence length in batch
            batch = data[:, i * bsz:(i + 1) * bsz]
            batches.append(batch)

        return batches

    def fit(self, data_train):
        with self.sess.as_default():
            #self.sess.run(self.init)  #default: warm-start
            epochs = range(1, self.epoch+1)
            for epoch in tqdm(epochs):

                batches = self.batchify(data_train, self.batch_size)
                bt_loss = 0
                if self.optimizer_type == "lbfgs":
                    for i in range(len(batches)):
                        batch = batches[i]
                        feed_dict = {self.data: batch, self.corrupt_prob: self.corrupt_prob_value}
                        self.optimizer.minimize(self.sess, loss_callback=self.print_loss,
                                            fetches=[self.loss, self.clique_loss, self.ae_loss, self.kl_loss,
                                                     self.self_weight_decay_J],
                                            feed_dict=feed_dict)

                else:
                    for i in range(len(batches)):
                        batch = batches[i]
                        feed_dict = {self.data: batch, self.corrupt_prob: self.corrupt_prob_value}
                        loss_bt, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                        cl, ae, kl, weight = self.sess.run(
                            [self.clique_loss, self.ae_loss, self.kl_loss, self.self_weight_decay_J],
                            feed_dict=feed_dict)

                        #print("cl_loss:%.10f, ae_loss:%.7f, kl_loss:%.20f, weight_l2_loss:%.7f" % (cl, ae, kl, weight))
                        bt_loss += loss_bt
                    #print("epoch %d loss:%.4f" % (epoch, bt_loss))
            # self.W_list_value = []
            # self.b_list_value = []
            # for w in self.W_list:
            #     self.W_list_value.append(w.eval())
            # for b in self.b_list:
            #     self.b_list_value.append(b.eval())
