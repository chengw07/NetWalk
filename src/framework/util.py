"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
from imports import *


# An empty class for sticking attributes onto
class Empty(object):
    pass


# Given a function and a function input, compute the gradient at that
# input, numerically.
#
# - `fn` - the function
# - `theta` - the point at which to compute the gradient
# - `epsilon` - the step-size
# - returns: \\( \\nabla_{\\mathrm{theta}} \\mathrm{fn}(\\mathrm{theta}) \\)
def compute_numerical_gradient(fn, theta, epsilon=1e-4):
    numgrad = np.zeros(theta.shape)

    for i in range(theta.size):
        theta_minus = theta.copy()
        theta_plus = theta.copy()
        theta_minus[i] = theta_minus[i] - epsilon
        theta_plus[i] = theta_plus[i] + epsilon
        numgrad[i] = (fn(theta_plus) - fn(theta_minus)) / (2 * epsilon)

    return numgrad


# Display a network or array.
#
# - `arr` - the data, one entry per column
# - `title` - optional title
# - `show` - if true, pause and display (call `plt.show()`)
def display_network(arr, title=None, show=True):
    arr = arr - np.mean(arr)
    L, M = arr.shape
    sz = int(np.sqrt(L))
    buf = 1

    # Figure out pleasant grid dimensions
    if M == np.floor(np.sqrt(M)) ** 2:
        n = m = int(np.sqrt(M))
    else:
        n = int(np.ceil(np.sqrt(M)))
        while (M % n) and n < 1.2 * np.sqrt(M):
            n += 1
        m = int(np.ceil(M / n))

    array = np.zeros([buf + m * (sz + buf), buf + n * (sz + buf)])

    k = 0
    for i in range(0, int(m)):
        for j in range(0, int(n)):
            if k >= M:
                continue
            cmax = np.max(arr[:, k])
            cmin = np.min(arr[:, k])
            r = buf + i * (sz + buf)
            c = buf + j * (sz + buf)
            array[r:r + sz, c:c + sz] = (arr[:, k].reshape([sz, sz], order='F') - cmin) / (cmax - cmin)
            k = k + 1
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(array, interpolation='nearest', cmap=plt.cm.gray)
    if show:
        plt.show()
