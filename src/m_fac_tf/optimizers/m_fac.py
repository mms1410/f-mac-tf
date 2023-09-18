# -*- coding: utf-8 -*-
import tensorflow as tf
from collections import deque
import abc


class MFAC(tf.keras.optimizers.Optimizer):

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.0,
            m=5,
            nesterov=False,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="MFAC",
            **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.m = m
        self.nesterov = nesterov
        # initialize fifo queue
        self.fifo = deque(maxlen=m)
        if isinstance(momentum, (int, float)) and (
                momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]

        # TODO(b/204321487): Add nesterov acceleration.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            add_value = tf.IndexedSlices(
                -gradient.values * lr, gradient.indices
            )
            if m is not None:
                m.assign(m * momentum)
                m.scatter_add(add_value)
                if self.nesterov:
                    variable.scatter_add(add_value)
                    variable.assign_add(m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.scatter_add(add_value)
        else:
            # Dense gradients
            if m is not None:
                m.assign(-gradient * lr + m * momentum)
                if self.nesterov:
                    variable.assign_add(-gradient * lr + m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.assign_add(-gradient * lr)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            }
        )
        return config

    def compute_gradients(self, loss, var_list, tape=None):
        """Compute gradients of loss on trainable variables.

        Args:
          loss: `Tensor` or callable. If a callable, `loss` should take no
            arguments and return the value to minimize.
          var_list: list or tuple of `Variable` objects to update to minimize
            `loss`, or a callable returning the list or tuple of `Variable`
            objects. Use callable when the variable list would otherwise be
            incomplete before `minimize` since the variables are created at the
            first time `loss` is called.
          tape: (Optional) `tf.GradientTape`. If `loss` is provided as a
            `Tensor`, the tape that computed the `loss` must be provided.

        Returns:
          A list of (gradient, variable) pairs. Variable is always present, but
          gradient can be `None`.
        """
        if not callable(loss) and tape is None:
            raise ValueError(
                "`tape` is required when a `Tensor` loss is passed. "
                f"Received: loss={loss}, tape={tape}."
            )
        if tape is None:
            tape = tf.GradientTape()
        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()

        grads = tape.gradient(loss, var_list)

        if len(self.fifo) == self.m:
            self.fifo.append(grads)
            fifo = self._fifo_to_grads(self.fifo)
            d, b = self._setup_matrices(fifo, grads)
            scaled_grads = self._compute_InvMatVec(fifo, grads, d, b)
            return list(zip(scaled_grads, var_list))
        else:
            self.fifo.append(grads)
            return list(zip(grads, var_list))


    def _setup_matrices(self, grads, damp=1e-8, dtype="float64"):
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

    def _compute_InvMatVec(self, grads, x, D, B, damp=1e-8, dtype="float64"):
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

    def _fifo_to_grads(self, fifo):
        """Fifo queue to gradient tensor"""
        grads = []
        for grad in fifo:
            grads.append(grad)
        return grads


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    model = tf.keras.applications.resnet50.ResNet50()
