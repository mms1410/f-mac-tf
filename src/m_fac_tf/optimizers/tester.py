# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from collections import deque

class TestOptimizer(tf.keras.optimizers.Optimizer):
    """
    Implements my new fancy Test Optimizer. Will it work?
    """

    def __init__(self, m, damp = 1e-8,base_optim = None, learning_rate=0.001, name="mfac",**kwargs):
        super(TestOptimizer, self).__init__(name=name, **kwargs)
        self._build_learning_rate(learning_rate)
        self._learning_rate = learning_rate
        self.damp = damp
        self.m = m
        self.grad_fifo = MatrixFifo(m)
        self.D = None
        self.B = None
        self.G = None
        self.base_grad = None
        self.scaled_grad = None
        self.lambd = 1 / damp
        self.test_counter = 0
        self.test_counter2 = 0


    def _setupMatrices(self):
        """
        """
        # init matrices
        self.G = self.grad_fifo.values
        self.D = tf.Variable(tf.math.scalar_mul(self.damp, tf.matmul(self.G, tf.transpose(self.G))))
        self.B = tf.Variable(tf.math.scalar_mul(self.damp, tf.Variable(tf.linalg.eye(self.m))))
        # Compute D
        for idx in range(1, self.m):
            denominator = tf.math.pow((self.m + self.D[idx - 1, idx - 1]), -1)
            test = self.D[idx:, idx:] - denominator * tf.matmul(tf.transpose(self.D[idx - 1:, idx:]), self.D[idx - 1:, idx:])
            self.D[idx:, idx:].assign(test)
        self.D = tf.linalg.band_part(self.D, 0, -1)
        # Compute B
        for idx in range(1, self.m):
            denominator = self.m + tf.linalg.diag_part(self.D)[:idx]
            tmp = tf.math.divide(-self.D[:idx, idx], denominator)
            tmp = tf.transpose(tmp)
            to_assign = tf.linalg.matvec(self.B[:idx, :idx], tmp)
            self.B[idx, :idx].assign(to_assign)


    def _compute_InvMatVec(self, x):
        """
        Compute \hat{F_{m}}\bm{x} for precomputed D and B
        """
        self.G = self.grad_fifo.values
        q = tf.Variable(tf.linalg.matvec(self.G, x))
        q0 = q[0] / (m + D[0, 0])
        q[0].assign(q0)
        for idx in range(1, self.m):
            tmp = q[idx:] - tf.math.scalar_mul(q[idx - 1], tf.transpose(self.D[idx - 1, idx:]))
            q[idx:].assign(tmp)
        denominator =self.m + tf.linalg.diag_part(self.D)
        q = q / denominator
        tmp = tf.linalg.matvec(self.B, q)
        a = tf.transpose(tf.linalg.matvec(self.G, tmp, transpose_a=True))
        b = tf.math.scalar_mul(self.damp, x)
        result = a - b
        return result



    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        self.base_grad = gradient
        self.grad_fifo.append(gradient)
        self.test_counter += 1
        if self.grad_fifo.counter < self.m:
            self.grad_fifo.append(gradient)
            variable.assign_add(-gradient * self.lr)
        else:
            self.test_counter2 += 1
            scaled_grad = gradient + 100
            self.scaled_grad = scaled_grad
            variable.assign_add(-scaled_grad * self.lr)




class MatrixFifo:
    """
    Implements idea of fifo queue for tensorflow matrix.
    """
    def __init__(self, ncol):
        self.values = None
        self.ncol=ncol
        self.counter = 0
    
    def append(self, vector):
        """
        For k by m matrix and vecotr of dimension k by 1 move columns 2,...,m 'to left by one position' and substitute column m with vector.
        """
        if self.values is None:
            # first vector to append will determine nrow
            self.values = tf.Variable(tf.zeros(shape=[vector.shape[1], self.ncol]))
            self.values[:,-1].assign(tf.cast(vector, dtype=self.values.dtype))
        else:
            # TODO: throw error message if vector has incompatible shape
            tmp = tf.identity(self.values)
            # update last column with new vector
            self.values[:,-1].assign(tf.cast(vector, dtype=self.values.dtype))
            # move columns to left
            self.values[:,:-1].assign(tmp[:,1:])
        self.counter += 1




if __name__ == "__main__":
    import sys
    "Toy example for 8 covariates and sec order approximation with last 3 gradients."
    n_variables = 8
    m = 3
    grad1 = tf.Variable(np.full((1, n_variables), 1))
    grad2 = tf.Variable(np.full((1, n_variables), 2))
    grad3 = tf.Variable(np.full((1, n_variables), 3))
    grad4 = tf.Variable(np.full((1, n_variables), 4))
    grad5 = tf.Variable(np.full((1, n_variables), 5))
    grad6 = tf.Variable(np.full((1, n_variables), 6))
    tgrad1 = tf.Tensor(np.full((1, n_variables), 1))
    G = MatrixFifo(m)
    G.append(grad1)
    print(G.values)







