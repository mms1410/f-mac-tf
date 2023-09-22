# -*- coding: utf-8 -*-
import tensorflow as tf
from collections import deque
import abc
import traceback



class MyAdam(tf.keras.optimizers.Optimizer):
    r"""Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
      learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to `0.001`.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 2nd moment estimates. Defaults to
        `0.999`.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        `1e-7`.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      {{base_optimizer_keyword_args}}

    Reference:
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=0.001,
        m = 5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adam",
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.m = m
        self.plane_grad = None

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build Adam variables on.
        """
        super().build(var_list)
        print("Call build")
        traceback.print_stack()
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        self.plane_grad = gradient

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        print("Call get_config")
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


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
        self.D = None
        self.B = None
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

        if len(self.fifo) == self.m:
            self.fifo.append(gradient)
            fifo = self._fifo_to_grads(self.fifo)
            self.D, self.B = self._setup_matrices(grads=fifo)
            gradient = self._compute_InvMatVec(grads=fifo, x=gradient, D=self.D, B=self.B)
            print("scaled grads: ")
        else:
            self.fifo.append(gradient)
            print("grads: ")

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

    def _setup_matrices(self):
        """
        Initialize D and B (Algorithm1)
        """
        # init matrices
        self.D = tf.Variable(tf.math.scalar_mul(self.damp, tf.matmul(self.G, tf.transpose(self.G))))
        self.B = tf.math.scalar_mul(self.damp, tf.Variable(tf.linalg.eye(self.m)))
        self.B = tf.Variable(B)
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
        return None

    def _compute_InvMatVec(self, x):
        """
        Compute \hat{F_{m}}\bm{x} for precomputed D and B
        """
        self.G = tf.convert_to_tensor(self.grads)
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

    def _fifo_to_grads(self, fifo):
        """Fifo queue to gradient tensor"""
        grads = []
        for grad in fifo:
            grads.append(grad)
        return grads
    
if __name__ == "__main__":
    pass