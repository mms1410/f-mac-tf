"""Class for custom MFAC-SGD optimizer."""

import tensorflow as tf

from src.utils.helper_functions import RowWiseMatrixFifo, deflatten

matmul = tf.linalg.matmul
scalmul = tf.math.scalar_mul
matvec = tf.linalg.matvec
band_part = tf.linalg.band_part
diag_part = tf.linalg.diag_part


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
        m=512,
        damp=1e-8,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="f-mfac-sgd",
        **kwargs,
    ):
        """Initialize the optimizer and all variables.

        Args:
            learning_rate: float or learning rate schedule function
            momentum: float or momentum schedule function
            nesterov: bool, whether to use Nesterov momentum
            weight_decay: float or weight decay schedule function
            clipnorm: float or global norm gradient clipping rate
            clipvalue: float or individual gradient clipping rate
            global_clipnorm: float or global norm gradient clipping rate
            use_ema: bool, whether to use exponential moving average
            m: int, number of gradients to store
            damp: float, damping factor
            ema_momentum: float or ema momentum schedule function
            ema_overwrite_frequency: int or ema overwrite frequency schedule function
            jit_compile: bool, whether to jit compile the optimizer
            name: string, name of the optimizer
            **kwargs: for backwards compatibility



        """
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
            **kwargs,
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.damp = damp
        self.m = m
        self.GradFifo = RowWiseMatrixFifo(self.m)
        self.D = None
        self.B = None
        self.G = None
        self.lambd = 1 / damp
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
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
                self.add_variable_from_reference(model_variable=var, variable_name="m")
            )
        self._built = True

    def minimize(self, loss, var_list, tape=None):
        """Minimize the loss function.

        Compute the Gradients for each variable in var_list. If the Fifo-Que is full,
        the gradients are scaled using the MFAC algorithm and be applied to the variables.

        Args:
            loss: loss function
            var_list: list of variables to compute gradients for
            tape: gradient tape for automatic differentiation

        Returns:
            None

        """
        grads_and_vars = self.compute_gradients(loss, var_list, tape)

        if grads_and_vars is not None:
            grads, vars = zip(*grads_and_vars)
            reconstructed_tensors = self.scale_grads(grads)
            grads_and_vars = list(zip(reconstructed_tensors, vars))

        self.apply_gradients(grads_and_vars)

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable.

        Args:
            gradient: gradient of the model variable
            variable: model variable

        Returns:
            None
        """
        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]

        # TODO(b/204321487): Add nesterov acceleration.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            add_value = tf.IndexedSlices(-gradient.values * lr, gradient.indices)
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
        """Scales the Gradients using the MFAC algorithm.

        This Function formats all the Gradients into one singe one-dimensional Tensor and
        append it to the fifo-que. If the fifo-que is full, the algorithm computes the scaled
        gradients with the MFAC algorithm and returns the scaled gradients in the original shape.


        Args:
            grads: list of gradients for each variable

        Returns:
            list of scaled gradients in the original shape

        """
        # Array zum speichern der alten shapes
        original_shapes = []
        # Array für eindimensionalen Gradienten
        gradient = []
        for grad in grads:
            original_shapes.append(grad.shape)
            gr = tf.reshape(grad, [-1])
            gradient.append(gr)

        gradient = tf.concat(gradient, axis=0)

        # Gradienten skalieren
        self.GradFifo.append(gradient)
        if self.GradFifo.counter >= self.m:
            self._setupMatrices()
            gradient = self._compute_InvMatVec(vec=gradient)

        # Array für Rückformattierung
        reconstructed_tensors = []

        start_index = 0

        for shape in original_shapes:
            # Berechnen der Anzahl der Elemente in der ursprünglichen Shape
            num_elements = tf.reduce_prod(shape)
            # Extrahieren des entsprechenden Teils des Arrays
            sub_array = gradient[start_index : start_index + num_elements]
            # Umformen des Teil-Arrays in die ursprüngliche Shape
            reconstructed = tf.reshape(sub_array, shape)
            # Hinzufügen des rekonstruierten Tensors zur Liste
            reconstructed_tensors.append(reconstructed)
            # Aktualisieren des Startindex für den nächsten Tensor
            start_index += num_elements

        return reconstructed_tensors

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            }
        )
        return config

    def _setupMatrices(self):
        """Implements Algorithm1 from paper.
        
        Here the matrices B and D are set up and calculated according 
        to Algorithm 1 so that they can be used for the function _compute_InvMatVec.
        
        """
        self.D = matmul(self.GradFifo.values, self.GradFifo.values, transpose_b=True)
        self.D = tf.Variable(scalmul(self.damp, self.D))
        self.B = tf.eye(self.m, self.m)
        self.B = tf.Variable(scalmul(self.damp, self.B))

        for idx in range(1, self.m):
            to_subtract = matmul(self.D[idx - 1 :, idx:], self.D[idx - 1 :, idx:], transpose_a=True)
            to_subtract = scalmul(1 / (self.m + self.D[idx - 1, idx - 1]), to_subtract)
            self.D[idx:, idx:].assign(self.D[idx:, idx:] - to_subtract)

        # 0 for upper and -1 for all elements
        self.D.assign(band_part(self.D, 0, -1))

        for idx in range(1, self.m):
            to_multiply = self.D[:idx, idx] / (-self.m + diag_part(self.D)[:idx])  # noqa E501
            to_multiply = tf.expand_dims(to_multiply, axis=1)
            to_assign = matmul(to_multiply, self.B[:idx, :idx], transpose_a=True)
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
            to_subtract = scalmul(q_vec[idx - 1], tf.transpose(self.D[idx - 1, idx:]))
            q_vec[idx:].assign(q_vec[idx:] - to_subtract)

        q_vec = q_vec / (self.m + diag_part(self.D))
        q_vec = tf.expand_dims(q_vec, axis=1)
        to_subtract = matmul(q_vec, self.B, transpose_a=True)
        to_subtract = matmul(to_subtract, self.GradFifo.values)
        to_subtract = tf.transpose(to_subtract)
        result = scalmul(self.damp, vec) - tf.squeeze(to_subtract)
        return result
