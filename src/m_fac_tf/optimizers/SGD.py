import tensorflow as tf
from collections import deque
class MFAC(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        m=5,
        damp = 1e-8,
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
        self.nesterov = nesterov
        self.damp = damp
        self.i = 0
        self.m = m
        self.grad_fifo = MatrixFifo(m)
        self.D = None
        self.B = None
        self.G = None
        self.lambd = 1 / damp
        self.base_grad = None
        self.scaled_grad = None
        self.identity_matrix = tf.constant(tf.linalg.eye(7960), dtype=tf.float32)
        self.test_counter = 0
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

         # Initialize self.D and self.B
        self.D = tf.Variable(tf.zeros(shape=(7960, 7960)), trainable=False)
        self.B = tf.Variable(tf.zeros(shape=(7960, 7960)), trainable=False)

    def minimize(self, loss, var_list, tape=None):
        grads_and_vars = self.compute_gradients(loss, var_list, tape)

        if grads_and_vars is not None:
            grads, vars = zip(*grads_and_vars)
            reconstructed_tensors = self.scale_grads(grads)
            grads_and_vars = list(zip(reconstructed_tensors, vars))

        self.apply_gradients(grads_and_vars)

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

    def scale_grads(self, grads):
        print("start grads", grads)
        #Array zum speichern der alten shapes
        original_shapes = []
        #Array für eindimensionalen Gradienten
        gradient = []
        for grad in grads:
          original_shapes.append(grad.shape)
          gr = tf.reshape(grad, [-1])
          gradient.append(gr)

        gradient = tf.concat(gradient, axis = 0)
        print("Gradient (eindimensional)", gradient)

        #Gradienten skalieren
        self.grad_fifo.append(gradient)
        if self.grad_fifo.counter == self.m:
          self.D, self.B = self._setupMatrices()
          gradient = self._compute_InvMatVec(x=gradient)

        #Array für Rückformattierung
        reconstructed_tensors = []

        start_index = 0

        for shape in original_shapes:
            # Berechnen der Anzahl der Elemente in der ursprünglichen Shape
            num_elements = tf.reduce_prod(shape)
            # Extrahieren des entsprechenden Teils des Arrays
            sub_array = gradient[start_index:start_index + num_elements]
            # Umformen des Teil-Arrays in die ursprüngliche Shape
            reconstructed = tf.reshape(sub_array, shape)
            # Hinzufügen des rekonstruierten Tensors zur Liste
            reconstructed_tensors.append(reconstructed)
            # Aktualisieren des Startindex für den nächsten Tensor
            start_index += num_elements

        print("final_tensor", reconstructed_tensors)
        return reconstructed_tensors

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


    def _setupMatrices(self):
        # init matrices
        self.G = self.grad_fifo.values
        self.D.assign(tf.math.scalar_mul(self.damp, tf.matmul(self.G, tf.transpose(self.G))))
        self.B.assign(tf.math.scalar_mul(self.damp, self.identity_matrix))
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
        return self.D, self.B

    def _compute_InvMatVec(self, x):
        """
        Compute \hat{F_{m}}\bm{x} for precomputed D and B
        """
        self.G = self.grad_fifo.values
        print("self.g", self.G)
        q = tf.linalg.matvec(self.G, x, transpose_a=True)  #self.G (Gradient_länge, self.m); x (Gradient_länge, 1)
        q = tf.math.scalar_mul(self.damp, q)
        print("q", q)
        q0 = q[0] / (self.m + self.D[0, 0])
        q = tf.concat([[q0], q[1:]], axis=0)
        for idx in range(1, self.m):
            tmp = q[idx:] - tf.math.scalar_mul(q[idx - 1], tf.transpose(self.D[idx - 1, idx:]))
            q = tf.concat([q[:idx], tmp], axis=0)
        denominator =self.m + tf.linalg.diag_part(self.D)
        q = q / denominator
        tmp = tf.linalg.matvec(self.B, q)
        print("tmp", tmp)
        a = tf.math.scalar_mul(self.damp, x)
        print("a", a)
        b = tf.transpose(tf.linalg.matvec(self.G, tmp, transpose_a=True))
        print("b", b)
        result = a - b
        return result
