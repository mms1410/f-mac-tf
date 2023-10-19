"""Contains class Mfac."""
import tensorflow as tf
from src.utils.helper_functions import deflatten, RowWiseMatrixFifo

matmul = tf.linalg.matmul
scalmul = tf.math.scalar_mul
matvec = tf.linalg.matvec
band_part = tf.linalg.band_part
diag_part = tf.linalg.diag_part

class Mfac(tf.keras.optimizers.SGD):
    def __init__(self, m, damp, name="MFAC", **kwargs):
        super(Mfac, self).__init__(name=name)
        self.m = m
        self.damp = damp
        self.D = None
        self.B = None
        self.GradFifo = RowWiseMatrixFifo(self.m)
        self.test_slot_1 = tf.Variable(0, dtype=tf.int32)
        self.test_slot_2 = tf.Variable(0, dtype=tf.int32)

    def update_step(self, gradient, variable):
        return super().update_step(gradient, variable)

    def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False, **kwargs):  # noqa E501
        return super().apply_gradients(grads_and_vars, name, skip_gradients_aggregation, **kwargs)  # noqa E501

    def compute_gradients(self, loss, var_list, tape=None):
        return super().compute_gradients(loss, var_list, tape)

    def minimize(self, loss, var_list, tape=None):
        grads_and_vars = self.compute_gradients(loss, var_list, tape)
        gradients_list = [gradient for gradient, _ in grads_and_vars]
        var_list = [var for _, var in grads_and_vars]
        # flatten grads and append fifo matrix
        # due to nested structure of shape flatten 2times
        flatten_grads = list(map(lambda x: tf.reshape(x, [-1]), gradients_list))  # noqa E501
        flatten_grads = tf.concat(flatten_grads, axis=0)
        self.GradFifo.append(flatten_grads)
        self.test_slot_2.assign_add(1)
        if self.GradFifo.counter >= self.m:
            self.test_slot_1.assign_add(1)
            # Do second order approximation
            # self.D, self.B = self._setupMatrices()
            self._setupMatrices()
            scaled_grads = self._compute_InvMatVec(vec=flatten_grads)
            # Reshape scaled grads into original format and combine with vars
            # to create an altered version of grads_and_vars
            shapes_grads = list(map(lambda x: x.shape, gradients_list))
            scaled_grads = deflatten(scaled_grads, shapes_grads)
            grads_and_vars = (list(zip(scaled_grads, var_list)))

        self.apply_gradients(grads_and_vars)

    def _setupMatrices(self):
        """Implements Algorithm1 from paper."""
        self.D = matmul(self.GradFifo.values,
                        self.GradFifo.values,
                        transpose_b=True)
        self.D = tf.Variable(scalmul(self.damp, self.D))
        self.B = tf.eye(self.m, self.m)
        self.B = tf.Variable(scalmul(self.damp, self.B))

        for idx in range(1, self.m):
            to_subtract = matmul(self.D[idx-1:, idx:],
                                 self.D[idx-1:, idx:],
                                 transpose_a=True)
            to_subtract = scalmul(1/(self.m + self.D[idx-1, idx-1]),
                                  to_subtract)
            self.D[idx:, idx:].assign(self.D[idx:, idx:] - to_subtract)

        # 0 for upper and -1 for all elements
        self.D.assign(band_part(self.D, 0, -1))

        for idx in range(1, self.m):
            to_multiply = self.D[:idx, idx] / (-self.m + diag_part(self.D)[:idx])  # noqa E501
            to_multiply = tf.expand_dims(to_multiply, axis=1)
            to_assign = matmul(to_multiply,
                               self.B[:idx, :idx],
                               transpose_a=True)
            self.B[idx, :idx].assign(to_assign)

    def _compute_InvMatVec(self, vec: tf.Tensor):
        """Implements Algorithm2 from paper.

        Compute the Matrix vector product using precomputed matrices D and B.

        Args:
            vec: tf.Tensor

        Returns:
            scaled gradient vector
        """
        q_vec = matvec(self.GradFifo.values, vec)
        q_vec = tf.Variable(scalmul(self.damp, q_vec))
        q_vec[0].assign(q_vec[0] / (self.m + self.D[0, 0]))

        for idx in range(1, self.m):
            to_subtract = scalmul(q_vec[idx-1],
                                  tf.transpose(self.D[idx-1, idx:]))
            q_vec[idx:].assign(q_vec[idx:] - to_subtract)

        q_vec = q_vec / (self.m + diag_part(self.D))
        q_vec = tf.expand_dims(q_vec, axis=1)
        to_subtract = matmul(q_vec, self.B, transpose_a=True)
        to_subtract = matmul(to_subtract, self.GradFifo.values)
        to_subtract = tf.transpose(to_subtract)
        result = scalmul(self.damp, vec) - tf.squeeze(to_subtract)
        return result
