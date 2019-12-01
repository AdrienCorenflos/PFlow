import tensorflow as tf

__all__ = ['sinkhorn', 'symmetric_sinkhorn', 'sinkhorn_divergence']


@tf.function
def _cost(x, y, p):
    x_minus_y = tf.expand_dims(x, 0) - tf.expand_dims(y, 1)
    if p == 1:
        return tf.reduce_sum(tf.abs(x_minus_y), -1)
    if p == 2:
        return tf.reduce_sum(tf.square(x_minus_y), -1)
    else:
        return tf.linalg.norm(x_minus_y, p, -1)


@tf.function
def scal(alpha, vec):
    return tf.tensordot(alpha, vec, 1)


@tf.function
def sinkhorn(w_x, x, w_y, y, p=2, eps=1e-1, tol=1e-3, max_iter=100, use_implicit_function_theorem=True):
    """
    Sinkhorn iterates, returns the dual vectors for the transport associated.
    :param w_x:
        weights for position x
    :param x:
        location of the first points
    :param w_y:
        weights for position y
    :param y:
        location of the second (target) points
    :param p:
        dimension of the norm used
    :param eps:
        regularisation parameter
    :param tol:
        stop iterations when potentials don't move more than tol anymore
    :param max_iter:
        max number of iterations
    :param use_implicit_function_theorem:
        method used to compute the resulting gradients -> unroll the iterates or compute at convergence
    """
    log_w_x = tf.math.log(w_x)
    log_w_y = tf.math.log(w_y)

    if not use_implicit_function_theorem:
        cost = _cost(x, y, p) / eps
        cost_t = tf.transpose(cost)
    else:
        cost = _cost(x, tf.stop_gradient(y), p) / eps
        cost_t = _cost(y, tf.stop_gradient(x), p) / eps

    v = tf.zeros_like(w_x)

    break_clause = tf.constant(True)
    i = 0
    while break_clause and i < max_iter - 1:
        v_prev = v
        temp_v = tf.reshape(v + log_w_x, (1, -1))
        u = -tf.reduce_logsumexp(temp_v - cost_t, 1)
        temp_u = tf.reshape(u + log_w_y, (1, -1))
        v = -tf.reduce_logsumexp(temp_u - cost, 1)
        err = eps * tf.reduce_mean(tf.abs(v - v_prev))
        break_clause = not(err < tol)
        i += 1

    if not use_implicit_function_theorem:
        temp_v = tf.reshape(v + log_w_x, (1, -1))
        u = -tf.reduce_logsumexp(temp_v - cost_t, 1)
        temp_u = tf.reshape(u + log_w_y, (1, -1))
        v = -tf.reduce_logsumexp(temp_u - cost, 1)

    else:
        temp_v = tf.reshape(tf.stop_gradient(v + log_w_x), (1, -1))
        u = -tf.reduce_logsumexp(temp_v - cost_t, 1)
        temp_u = tf.reshape(tf.stop_gradient(u + log_w_y), (1, -1))
        v = -tf.reduce_logsumexp(temp_u - cost, 1)

    return eps * u, eps * v


@tf.function
def symmetric_sinkhorn(w_x, x, p=2, eps=1e-1, tol=1e-3, max_iter=100, use_implicit_function_theorem=True):
    """
    Sinkhorn iterates, returns the dual vector for the transport associated of (w_x, x) w.r.t itself
    :param w_x:
        weights for position x
    :param x:
        location of the first points
    :param p:
        dimension of the norm used
    :param eps:
        regularisation parameter
    :param tol:
        stop iterations when potentials don't move more than tol anymore
    :param max_iter:
        max number of iterations
    :param use_implicit_function_theorem:
        method used to compute the resulting gradients -> unroll the iterates or compute at convergence
    """
    log_w_x = tf.math.log(w_x)

    if not use_implicit_function_theorem:
        cost = _cost(x, x, p) / eps
    else:
        cost = _cost(x, tf.stop_gradient(x), p) / eps

    u = tf.zeros_like(w_x)
    trigger = tf.constant(True)
    i = 0
    while trigger and i < max_iter - 1:
        u_prev = u
        temp_u = tf.reshape(u + log_w_x, (1, -1))
        u = 0.5 * (u - tf.reduce_logsumexp(temp_u - cost, 1))
        err = eps * tf.reduce_mean(tf.abs((u - u_prev)))
        trigger = err < tol
        i += 1

    if not use_implicit_function_theorem:
        temp_u = tf.reshape(u + log_w_x, (1, -1))
        u = -tf.reduce_logsumexp(temp_u - cost, 1)
    else:
        temp_u = tf.reshape(tf.stop_gradient(u) + log_w_x, (1, -1))
        u = -tf.reduce_logsumexp(temp_u - cost, 1)

    return eps * u


@tf.function
def sinkhorn_divergence(w_x, x, w_y, y, p=2, eps=1e-1, tol=1e-3, max_iter_cross=100, max_iter_sym=100,
                        use_implicit_function_theorem=True):
    """
    Sinkhorn divergence
    :param w_x:
        weights for position x
    :param x:
        location of the first points
    :param p:
        dimension of the norm used
    :param eps:
        regularisation parameter
    :param tol:
        stop iterations when potentials don't move more than tol anymore
    :param max_iter_cross:
        max number of iterations for OT(x, y)
    :param max_iter_sym:
        max number of iterations for OT(x, x) and OT(y, y)
    :param use_implicit_function_theorem:
        method used to compute the resulting gradients -> unroll the iterates or compute at convergence
    """
    a_y, b_x = sinkhorn(w_x, x, w_y, y, p, eps, tol, max_iter_cross,
                        use_implicit_function_theorem)
    a_x = symmetric_sinkhorn(w_x, x, p, eps, tol, max_iter_sym,
                             use_implicit_function_theorem)
    b_y = symmetric_sinkhorn(w_y, y, p, eps, tol, max_iter_sym,
                             use_implicit_function_theorem)
    cost = scal(w_x, b_x - a_x) + scal(w_y, a_y - b_y)

    return cost


def main():
    import numpy as np
    import ot
    import geomloss.sinkhorn_samples as losses
    np.random.seed(0)
    N, M = 200, 200
    x = np.random.normal(0., 1., (N, 1))
    w_x = np.random.uniform(0.25, 0.75, N)
    w_x /= w_x.sum()

    y = np.random.normal(1., 3., (M, 1))
    w_y = np.full(M, 1 / M)

    print(ot.bregman.empirical_sinkhorn2(x, y, 0.5, w_x, w_y, 'euclidean'))
    a_y, b_x = sinkhorn(w_x, x, w_y, y, 2, 0.5, 1e-9, 10000)
    print(scal(w_x, b_x) + scal(w_y, a_y))
    print(losses.sinkhorn_tensorized(w_x, x, w_y, y, 0.5))



if __name__ == '__main__':
    main()
