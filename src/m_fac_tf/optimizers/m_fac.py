# -*- coding: utf-8 -*-

import tensorflow as tf


class MFAC(tf.keras.optimizers.Optimizer):
    """
    Implements MFAC optimizer.
    """
    pass




def setup_matrices(self, grads, damp=1e-8, dtype = "float64"):
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
    for idx in range(2, m):
        frac = tf.math.pow((m + D[idx - 1, idx - 1]), -1)
        test = D[idx:, idx:] - frac * tf.matmul(tf.transpose(D[idx - 1:, idx:]), D[idx - 1:, idx:])
        D[idx:, idx:].assign(test)
    D = tf.linalg.band_part(D, 0, -1)
    # Compute B
    for idx in range(2, m):
        frac = m + tf.linalg.diag_part(D)[:idx]
        tmp = tf.math.divide(-D[:idx, idx], frac)
        tmp = tf.transpose(tmp)
        to_assign = tf.linalg.matvec(B[:idx, :idx], tmp)
        B[idx, :idx].assign(to_assign)
    return D, B

def compute_InvMatVec(self, grads, x, D, B,damp = 1e-8, dtype = "float64"):
    """
    Compute \hat{F_{m}}\bm{x} for precomputed D and B
    """
    G = tf.convert_to_tensor(grads, dtype=dtype)
    m =  G.shape[0]
    q = tf.linalg.matvec(G, x)
    q = tf.Variable(tf.math.scalar_mul(damp, q))
    for idx in range(2,m):
        q[idx:] = q[idx:] - tf.math.scalar_mul(q[idx-1], tf.transpose(D[idx-1, idx:]))
    frac = m + tf.linalg.band_part(D, 0, 1)
    q = q / frac
    result = tf.matmul(tf.matmul(tf.transpose(q, B)), G)
    result = tf.math.scalar_mul(damp, x) - tf.transpose(result)
    return result

if __name__ == "__main__":
    grad1 = tf.constant([1,2,3,4,5])
    grad2 = tf.constant([5,6,7,8,9])
    grad3 = tf.constant([1,5,7,3,9])
    grad4 = tf.constant([8,2,4,6,2])
    tf.queue.FIFOQueue()