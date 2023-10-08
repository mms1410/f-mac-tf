import tensorflow as tf
from utils.utils import MatrixFifo, deflatten

class SimpleOptim(tf.keras.optimizers.SGD):
    def __init__(self, steps, m=3, **kwargs):
        super(SimpleOptim, self).__init__()
        self.m = 3
        self.D = None
        self.B = None
        self.test_slot_1 = None
        self.test_slot_2 = None
        self.test_slot_3 = None
        self.GradFifo = MatrixFifo(self.m)
        self.steps = steps
    
    def update_step(self, gradient, variable):
        return super().update_step(gradient, variable)
    
    def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False, **kwargs):
        return super().apply_gradients(grads_and_vars, name, skip_gradients_aggregation, **kwargs)
    

    def compute_gradients(self, loss, var_list, tape=None):
        return super().compute_gradients(loss, var_list, tape)
    
    def minimize(self, loss, var_list, tape=None):
        grads_and_vars = self.compute_gradients(loss, var_list, tape)
        gradients_list= [gradient for gradient, _ in grads_and_vars]
        var_list = [var for _, var in grads_and_vars]
        # flatten grads and append int fifo matrix
        flatten_grads = list(map(lambda x: tf.reshape(x, [-1]), gradients_list))
        flatten_grads = tf.concat(flatten_grads, axis = 0)
        self.GradFifo.append(flatten_grads)
        if self.GradFifo.counter >= self.m:
            # Do second order approximation
            self._setupMatrices()
            scaled_grads = self._compute_InvMatVec(flatten_grads)
            # Reshape scaled grads into original format and combine with vars
            # to create an altered version of grads_and_vars
            shapes_grads = list(map(lambda x: x.shape, gradients_list))
            scaled_grads = deflatten(scaled_grads, shapes_grads)
            grads_and_vars.assing(list(zip(scaled_grads, var_list)))

        self.apply_gradients(grads_and_vars)


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
        q0 = q[0] / (self.m + self.D[0, 0])
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
    