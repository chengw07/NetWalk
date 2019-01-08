"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import tensorflow as tf


def corrupt(x):
    r = tf.add(tf.cast(x, tf.float32), tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=0.1, dtype=tf.float32), tf.float32))
    # r = tf.multiply(x,tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0.5, maxval=1.5, dtype=tf.float32), tf.float32))
    return r


# def kl_divergence(p, p_hat):
#     return tf.reduce_mean(p * tf.log(tf.abs(p)) - p * tf.log(tf.abs(p_hat)) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat))

def autoencoder(data, corrupt_prob, dimensions, beta=0.01, rho = 0.4, activation = tf.nn.sigmoid, lamb = 0.01, gamma =0.01):
    # init_random = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=24, dtype=tf.float32)
    #
    # init_truncated = tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=24, dtype=tf.float32)
    #
    # init_uniform = tf.random_uniform_initializer(minval=0, maxval=1, seed=24, dtype=tf.float32)

    init_uniform_unit = tf.uniform_unit_scaling_initializer(factor=1.0, seed=24, dtype=tf.float32)

    # init_variance_scaling_normal = tf.variance_scaling_initializer(scale=1.0, mode="fan_in",
    #                                                                distribution="normal", seed=24, dtype=tf.float32)
    # init_variance_scaling_uniform = tf.variance_scaling_initializer(scale=1.0, mode="fan_in",
    #                                                                 distribution="uniform", seed=24, dtype=tf.float32)
    # init_orthogonal = tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32)
    # init_glorot_uniform = tf.glorot_uniform_initializer()
    # init_glorot_normal = tf.glorot_normal_initializer()


    # x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    x = tf.cast(data, tf.float32)

    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)
    noise_input = current_input

    weight_decay_J = 0


    # Build the encoder
    print("========= encoder begin ==========")
    encoder = []
    encoder_b = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[0])
        print("encoder : layer_i - n_output - n_input", layer_i, n_output, n_input)

        #W = tf.Variable(tf.random_uniform([n_output, n_input], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))

        W_name = "W1_" + str(layer_i)
        W = tf.get_variable(W_name, shape=[n_output, n_input], initializer=init_uniform_unit)

        b = tf.Variable(tf.zeros([1, n_output]))
        encoder.append(W)
        encoder_b.append(b)
        output = activation(tf.transpose(tf.transpose(tf.matmul(W, current_input))+ b))
        current_input = output
        weight_decay_J += (lamb / 2.0) * (tf.reduce_mean(W ** 2))
    print("========= encoder finish =========")
    # latent representation
    encoder_out = current_input
    print(encoder_out.shape)
    #encoder.reverse()
    # Build the decoder using the same weights
    print("========= decoder begin ==========")
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        print("decoder : layer_i - n_output", layer_i, n_output)
        n_input = int(current_input.get_shape()[0])
        #W = tf.transpose(encoder[layer_i])  # transpose of the weights

        #W = tf.Variable(tf.random_uniform([n_output, n_input], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))

        W_name = "W2_" + str(layer_i)
        W = tf.get_variable(W_name, shape=[n_output, n_input], initializer=init_uniform_unit)

        b = tf.Variable(tf.zeros([1, n_output]))

        output = activation(tf.transpose(tf.transpose(tf.matmul(W, current_input)) + b))
        current_input = output
        weight_decay_J += (lamb / 2.0) * (tf.reduce_mean(W ** 2))
    print("========= decoder finish =========")
    # now have the reconstruction through the network
    reconstruction = current_input
    # kl = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=z/0.01))
    #encoder.reverse()
    rhohats = tf.reduce_mean(tf.transpose(encoder_out), 0)

    #p = np.repeat([rho], encoder_out.get_shape().as_list()[0]).astype(np.float32)
    kl = tf.reduce_mean(
        rho * tf.log(rho / rhohats) + (1 - rho) * tf.log((1 - rho) / (1 - rhohats)))

    #m = data.get_shape().as_list()[1] * 1.0
    ae_loss = (gamma / 2.0) * tf.reduce_mean(tf.square(reconstruction - x))


    kl_loss = beta * kl
    cost = ae_loss + kl_loss + weight_decay_J
    # cost = 0.5 * tf.reduce_sum(tf.square(y - x))

    return {
        'x': x,
        'encoder_out': encoder_out,
        'reconstruction': reconstruction,
        'corrupt_prob': corrupt_prob,
        'cost': cost,
        'noise_input': noise_input,
        'kl': kl,
        'weight_decay_J': weight_decay_J,
        'ae_loss': ae_loss,
        'kl_loss': kl_loss,
        'W_list': encoder,
        'b_list': encoder_b
    }
