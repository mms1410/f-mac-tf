# -*- coding: utf-8 -*-
import tensorflow as tf
from collections import deque
import abc


# ToDo:
#   (Base-)Optimizer import from tensorflow
#   apply_gradients:
#       possible multiple grads and vars -> individual fifo queues?
#   proper handling of dtypes?
#   distributed training?


class Mfac(tf.keras.optimizers.Optimizer):
    """

    """

    @abc.abstractmethod
    def __init__(self, m, damp = 1e-8, learning_rate=0.001, name="mfac", **kwargs):
        """Initialize M-FAC optimizer.

        Args:
            m: integer for last m gradients to use for second order approximation.
            learning_rate: float of learning rate
            name: string of optimizes name.
            dtype: datatype used in tensors.


        """
        super(Mfac, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.m = m
        self.damp = damp
        self.gradient_fifo = deque(cpacity=m)
        self.scaled_gradient = None
        self.D = None
        self.B = None

    @abc.abstractmethod
    def apply_gradients(
            self,
            grads_and_vars,
            name=None,
            skip_gradients_aggregation=False,
            **kwargs,
    ):
        """Apply gradients to variables.

        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.
          skip_gradients_aggregation: If true, gradients aggregation will not be
            performed inside optimizer. Usually this arg is set to True when you
            write custom code aggregating gradients outside the optimizer.
          **kwargs: keyword arguments only used for backward compatibility.

        Returns:
          A `tf.Variable`, representing the current iteration.
          """
        pass

    @abc.abstractmethod
    def minimize(self, loss, var_list, tape=None):
        """Minimize `loss` by updating `var_list`.

        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before applying
        then call `tf.GradientTape` and `apply_gradients()` explicitly instead
        of using this function.

        Args:
          loss: `Tensor` or callable. If a callable, `loss` should take no
            arguments and return the value to minimize.
          var_list: list or tuple of `Variable` objects to update to minimize
            `loss`, or a callable returning the list or tuple of `Variable`
            objects.  Use callable when the variable list would otherwise be
            incomplete before `minimize` since the variables are created at the
            first time `loss` is called.
          tape: (Optional) `tf.GradientTape`.

        Returns:
          None
        """
        # compute normal gradient
        grads_and_vars = self.compute_gradients(loss, var_list, tape)
        # if there are enough m past gradients adjust computed normal gradient with past m gradients.
        if len(self.gradient_fifo) < self.gradient_fifo.maxlen:
            # ToDo: add (normal) gradient to fifo
        elif len(self.gradient_fifo) == self.gradient_fifo.maxlen:
            self.D, self.B = self._setup_matrices(self._fifo_to_grads(self.gradient_fifo))
            scaled_gradient = self._compute_InvMatVec(gradient, self.D, self.B, self.damp)
            self.gradient_fifo.append(scaled_gradient)
            # ToDo: replace gradient with scaled_gradient in grads_and_vars
        else:
            scaled_gradient = self._compute_InvMatVec(gradient, self.D, self.B, self.damp)
            self.gradient_fifo.append(scaled_gradient)
            # ToDo: replace gradient with scaled_gradient in grads_and_vars
        self.apply_gradients(grads_and_vars)

    @abc.abstractmethod
    def update_step(self, gradient, variable):
        """Function to update variable value based on given gradients.

        This method must be implemented in customized optimizers.

        Args:
          gradient: backpropagated gradient of the given variable.
          variable: variable whose value needs to be updated.

        Returns:
          An `Operation` that applies the specified gradients.

        """
        # since in minimize

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
        """Fifo queue to gradient tensor"""
        grads = []
        for grad in fifo:
            grads.append(grad)
        return grads


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    model = tf.keras.applications.resnet50.ResNet50()
