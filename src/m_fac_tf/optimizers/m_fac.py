# -*- coding: utf-8 -*-
import tensorflow as tf

class MFAC(tf.keras.optimizers.Optimizer):
    """Implements MFAC Optimizer"""

    def __init__(self, m, base_optim = None, learning_rate=0.001, damp=1e-8, name="mfac", **kwargs):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.m=m
        self.base_optim = base_optim
        self.damp=damp
        self.lambd = 1 / damp
        self.D = None
        self.B = None
        self.G = None
        self.MatFifo = MatrixFifo(m)
        self.grads = None
        self.scaled_grads = None


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

    def update_step(self, gradient, variable):
        """update"""
        if self.base_optim is not None:
            return self.base_optim.update_step(gradient, variable)
        else:
            lr = tf.cast(self.learning_rate, variable.dtype)
            variable.assign_add(-gradient * lr)
        
    def minimize(self, loss, var_list, tape=None):
        """minimize"""
        grads_and_vars = self.compute_gradients(loss, var_list, tape)
        grads, vars = (zip(*grads_and_vars))
        flatten_grads = list(map(lambda x: tf.reshape(x, [-1]), grads))
        flatten_grads = tf.concat(flatten_grads)
        self.MatFifo.append(flatten_grads)
        if self.MatFifo.counter >= self.m:
            self._setupMatrices()
            self.scaled_grads = self._compute_InvMatVec(flatten_grads)
        else:
            self.scaled_grads = grads



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