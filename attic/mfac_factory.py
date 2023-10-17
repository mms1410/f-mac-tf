"""Contains the MfacFactoy cass."""
import tensorflow as tf
from src.m_fac_tf.utils.helper_functions import deflatten, flatten_grads
from src.m_fac_tf.utils.helper_functions import MatrixFifo


class MfacFactory:
    def __init__(self, parent_optimizer):
        self.parent = parent_optimizer

    def get_mfac_optimizer(self, m, damp):
        class MFAC(self.parent):
            def __init__(self, m=m, damp=damp, **kwargs):
                super(MFAC, self).__init__()
                self.m = m
                self.damp = damp
                self.D = None
                self.B = None
                self.test_slot_1 = None
                self.test_slot_2 = None
                self.test_slot_3 = None
                self.GradFifo = MatrixFifo(self.m)
                self.steps = ""

            def update_step(self, gradient, variable):
                return super().update_step(gradient, variable)
    
            def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False, **kwargs):
                return super().apply_gradients(grads_and_vars, name, skip_gradients_aggregation, **kwargs)
            
            def compute_gradients(self, loss, var_list, tape=None):
                return super().compute_gradients(loss, var_list, tape)
            
            def minimize(self, loss, var_list, tape=None):
                grads_and_vars = self.compute_gradients(loss, var_list, tape)
                gradients_list = [gradient for gradient, _ in grads_and_vars]
                var_list = [var for _, var in grads_and_vars]
                # flatten grads and append int fifo matrix
                flatten_grads = list(map(lambda x: tf.reshape(x, [-1]), gradients_list))
                flatten_grads = tf.concat(flatten_grads, axis=0)
                self.GradFifo.append(flatten_grads)
                if self.GradFifo.counter >= self.m:
                    # Do second order approximation
                    self.D, self.B = self._setupMatrices()
                    scaled_grads = self._compute_InvMatVec(x=flatten_grads,
                                                           G=self.GradFifo.values)
                    # Reshape scaled grads into original format and combine with vars
                    # to create an altered version of grads_and_vars
                    shapes_grads = list(map(lambda x: x.shape, gradients_list))
                    scaled_grads = deflatten(scaled_grads, shapes_grads)
                    grads_and_vars.assing(list(zip(scaled_grads, var_list)))
                self.apply_gradients(grads_and_vars)

            def _setupMatrices(self):
                # init matrices
                D = tf.math.scalar_mul(self.damp, tf.matmul(self.G, tf.transpose(self.G)))
                B = tf.math.scalar_mul(self.damp, tf.linalg.eye(self.m))
                # Compute D
                for idx in range(1, self.m):
                    denominator = tf.math.pow((self.m + D[idx - 1, idx - 1]), -1)
                    test = D[idx:, idx:] - denominator * tf.matmul(tf.transpose(D[idx - 1:, idx:]), D[idx - 1:, idx:])
                    D = tf.concat([D[:idx, :], tf.concat([D[idx:, :idx], test], axis=1)], axis=0)
                D = tf.linalg.band_part(D, 0, -1)
                # Compute B
                for idx in range(1, self.m):
                    denominator = self.m + tf.linalg.diag_part(D)[:idx]
                    tmp = tf.math.divide(-D[:idx, idx], denominator)
                    tmp = tf.transpose(tmp)
                to_assign = tf.linalg.matvec(B[:idx, :idx], tmp)
                B_row = tf.concat([to_assign, B[idx, idx:]], axis=0)
                B = tf.concat([B[:idx, :], B_row[None, :], B[idx+1:, :]], axis=0)
                return D, B

            def _compute_InvMatVec(self, x, G):
                """
                Compute \hat{F_{m}}\bm{x} for precomputed D and B
                """
                q = tf.linalg.matvec(G, x)  #self.G (Gradient_länge, self.m); x (Gradient_länge, 1)
                q = tf.math.scalar_mul(self.damp, q)
                q0 = q[0] / (self.m + self.D[0, 0])
                q = tf.concat([[q0], q[1:]], axis=0)
                for idx in range(1, self.m):
                    tmp = q[idx:] - tf.math.scalar_mul(q[idx - 1], tf.transpose(self.D[idx - 1, idx:]))
                    q = tf.concat([q[:idx], tmp], axis=0)
                denominator =self.m + tf.linalg.diag_part(self.D)
                q  = q / denominator
                q = tf.reshape(q, [1,self.m])
                tmp = tf.matmul(q, self.B)
                a  = tf.math.scalar_mul(self.damp, x) #
                b = tf.transpose(tf.matmul(tmp, G))
                result = a - b
                return result
        return MFAC()