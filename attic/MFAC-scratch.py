from collections import deque
import tensorflow as tf
import numpy as np
tf.random.set_seed(22)


class MFAC(tf.Module):
    """
    Implements MFAC optimizer.
    """

    def __init__(self, m, learning_rate=1e-3):
        """
        Initialize MFAC optimizer.

        Initializing will create the attribute gradient_queue, a FIFO queue of capacity m.
        After the first m steps, the last m gradients will be used to rescale gradients in order to  account for second
        order approximation terms. Each newly computed gradient will then replace the oldest gradient.
        in gradient_queue.

        Args:
            m: An integer indicating the number of last m gradients to be used.
        """
        self.gradient_fifo = deque([], maxlen=m)
        self.learning_rate = learning_rate
        self.B = None
        self.D = None

    def apply_gradients(self, grads, vars):
        """
        Apply M-FAC algorithm to update list of variables ('vars') with the last m gradients in 'grads'.
        """
        # Compute gradient

        if len(self.gradient_fifo) < self.gradient_fifo.maxlen:
            for grad, var in zip(grads, vars):
                new_grad = self.learning_rate * grad
        elif len(self.gradient_fifo) == self.gradient_fifo.maxlen:
            self.D, self.B = _setup_matrices(self.gradient_fifo)
            new_grad = _compute_InvMatVec(self.gradient_fifo, x, self.D, self.B)
        elif len(self.gradient_fifo) > self.gradient_fifo.maxlen:
            pass

        # add gradient into queue
        self.gradient_fifo.append(new_grad)
        # assign gradient to variable
        var.assign_sub(new_grad)
        print(f"assigned new gradient {new_grad} \n")


########################################################################################################################

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


def fifo_to_grads(fifo):
    """

    """
    grads = []
    for grad in fifo:
        grads.append(grad)
    return grads
########################################################################################################################
def loss(x):
    return 2 * (x ** 4) + 3 * (x ** 3) + 2


def grad(f, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        result = f(x)
    return tape.gradient(result, x)


def convergence_test(optimizer, loss_fn, grad_fn=grad, init_val=2., max_iters=2000):
    # Function for optimizer convergence test
    print("-------------------------------")
    # Initializing variables and structures
    x_star = tf.Variable(init_val)
    param_path = []
    converged = False

    for iter in range(1, max_iters + 1):
        x_grad = grad_fn(loss_fn, x_star)

        # Case for exploding gradient
        if tf.math.is_nan(x_grad):
            print(f"Gradient exploded at iteration {iter}\n")
            return []

        # Updating the variable and storing its old-version
        x_old = x_star.numpy()
        optimizer.apply_gradients([x_grad], [x_star])
        param_path.append(x_star.numpy())

        # Checking for convergence
        if x_star == x_old:
            print(f"Converged in {iter} iterations\n")
            converged = True
            break

    # Print early termination message
    if not converged:
        print(f"Exceeded maximum of {max_iters} iterations. Test terminated.\n")
    else:
        print(f"Termination within {max_iters} iterations.\n")
    return param_path


def test_procedure(test_optimizer):
    param_map_gd = {}
    learning_rates = [1e-3, 1e-2, 1e-1]
    for learning_rate in learning_rates:
        param_map_gd[learning_rate] = (convergence_test(
            test_optimizer, loss_fn=loss))


########################################################################################################################
if __name__ == "__main__":
    x_vals = tf.linspace(-2, 2, 201)
    x_vals = tf.cast(x_vals, tf.float32)
    m = 3
    dtype="float64"
    grad1 = tf.constant(np.random.normal(5), shape=(5,), dtype=dtype)
    grad2 = tf.constant(np.random.normal(5), shape=(5,), dtype=dtype)
    grad3 = tf.constant(np.random.normal(5), shape=(5,), dtype=dtype)
    grad4 = tf.constant(np.random.normal(5), shape=(5,), dtype=dtype)

    fifo =deque([], maxlen=m)
    fifo.append(grad1)
    fifo.append(grad2)
    print(len(fifo))
    fifo.append(grad3)
    print(len(fifo))
    print(fifo)
    fifo.append(grad4)
    print(fifo)

    grads = fifo_to_grads(fifo)
    # D, B = _setup_matrices(grads, dtype=dtype)
    # scaled_grad4 = _compute_InvMatVec(grads,grad4, D, B)