# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from collections import deque

# ToDo:
#   (Base-)Optimizer import
#   apply_gradients:
#       possible multiple grads and vars -> individual fifo queues?
#   proper handling of dtypes?
#   distributed training?
#

class Mfac(optimizer):
    """

    """

    def __init__(self, m, learning_rate=0.001, name="mfac", dtype="float64", **kwargs):
        """
        Initialize M-FAC optimizer.
        """
        super().__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.m = m
        self.dtype =dtype
        self.gradient_fifo = deque([], maxlen=m)
        self.D = None
        self.B = None

    def apply_gradients(self, grads_and_vars, name=None):
        """
        Apply second order approximation with last m gradients.
        """

        # extract gradient and loss

        # compute new gradient
        len_fifo = len(self.gradient_fifo)
        if len_fifo < self.m:
            # do normal GDS
            pass
        elif len_fifo == self.m:
            # setup matrices (with last m gradients) and compute scaled gradients
            self.D, self.B = _setup_matrices(gradient)
            last_m_gradients = _fifo_to_grads(self.gradient_fifo)
            scaled_gradient = _compute_InvMatVec(last_m_gradients, gradient, self.D, self.B, dtype=self.dtype)
        else:
            # compute scaled gradient
            last_m_gradients = _fifo_to_grads(self.gradient_fifo)
            scaled_gradient = _compute_InvMatVec(last_m_gradients, gradient, self.D, self.B, dtype=self.dtype)
        # update fifo queue
        self.gradient_fifo.append(scaled_gradient)
        # apply new gradient


def _setup_matrices(grads, damp=1e-8, dtype="float64"):
    """
    Initialize D and B (Algorithm1)
    """
    G = tf.convert_to_tensor(grads, dtype=dtype)
    # init matrices
    m = G.shape[0]
    D = tf.Variable(tf.math.scalar_mul(damp, tf.matmul(G, tf.transpose(G))), dtype=dtype)
    B = tf.math.scalar_mul(damp, tf.Variable(tf.linalg.eye(m, dtype=dtype), dtype=dtype))
    B = tf.Variable(B)
    # Compute D
    for idx in range(1, m):
        denominator = tf.math.pow((m + D[idx - 1, idx - 1]), -1)
        test = D[idx:, idx:] - denominator * tf.matmul(tf.transpose(D[idx - 1:, idx:]), D[idx - 1:, idx:])
        D[idx:, idx:].assign(test)
    D = tf.linalg.band_part(D, 0, -1)
    # Compute B
    for idx in range(1, m):
        denominator = m + tf.linalg.diag_part(D)[:idx]
        tmp = tf.math.divide(-D[:idx, idx], denominator)
        tmp = tf.transpose(tmp)
        to_assign = tf.linalg.matvec(B[:idx, :idx], tmp)
        B[idx, :idx].assign(to_assign)
    return D, B


def _compute_InvMatVec(grads, x, D, B, damp=1e-8, dtype="float64"):
    """
    Compute \hat{F_{m}}\bm{x} for precomputed D and B
    """
    G = tf.convert_to_tensor(grads, dtype=dtype)
    m = G.shape[0]
    q = tf.Variable(tf.linalg.matvec(G, x))
    q0 = q[0] / (m + D[0, 0])
    q[0].assign(q0)
    for idx in range(1, m):
        tmp = q[idx:] - tf.math.scalar_mul(q[idx - 1], tf.transpose(D[idx - 1, idx:]))
        q[idx:].assign(tmp)
    denominator = m + tf.linalg.diag_part(D)
    q = q / denominator
    tmp = tf.linalg.matvec(B, q)
    a = tf.transpose(tf.linalg.matvec(G, tmp, transpose_a=True))
    b = tf.math.scalar_mul(damp, x)
    result = a - b
    return result


def _fifo_to_grads(fifo):
    """

    """
    grads = []
    for grad in fifo:
        grads.append(grad)
    return grads