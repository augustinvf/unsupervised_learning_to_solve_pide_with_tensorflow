import tensorflow as tf

def lambda_left_term(theta, sigma, nu):
    return tf.sqrt((tf.square(theta) / tf.square(sigma)**4) + (2 / (tf.square(sigma)**2 * nu)))

def lambda_right_term(theta, sigma):
    return (theta / tf.square(sigma))

def compute_lambda_p(theta, sigma, nu):
    return lambda_left_term(theta, sigma, nu) - lambda_right_term(theta, sigma)

def compute_lambda_n(theta, sigma, nu):
    return lambda_left_term(theta, sigma, nu) + lambda_right_term(theta, sigma)

def k(y, lambda_n, lambda_p, nu):
    output_tensor = tf.zeros_like(y, dtype=tf.float32)
    is_positive = tf.greater(y, 0)
    is_negative = tf.logical_not(is_positive)
    output_tensor = tf.where(is_positive, tf.exp(-lambda_p * y) / (nu * y), output_tensor)
    output_tensor = tf.where(is_negative, -tf.exp(lambda_n * y) / (nu * y), output_tensor)

    return output_tensor

def gamma(x) :
    # x is always positive
    return tf.math.exp(tf.math.lgamma(x))

def iupper_gamma(x, a) :
    return gamma(a) * tf.math.igammac(x, a)

def g_1(x, alpha):
    if alpha == 0 :
        return tf.math.exp(-x)
    elif 0 < alpha < 1 :
        # Here we want to compute the inverse upper gamma function
        # there is only the normalized upper gamma function
        # so we compute gamma * normalized upper gamma
        # and gamma = exp(log(gamma)) because tf has the loggamma function implemented
        return iupper_gamma(x, 1 - alpha)

def g_2(x, alpha):
    #return 0
    if alpha == 0 :
        return tf.math.special.expint(x)
    elif 0 < alpha < 1 :
        return ((tf.math.exp(-x) * tf.pow(x, -alpha))/ alpha) - (1/alpha) * iupper_gamma(x, 1-alpha)

def sigma_square(nu, epsilon, lambda_n, lambda_p, Y):
    # we put 0.0 because some operations doesn't support int32
    term1 = (1 / nu) * tf.pow(lambda_p, Y - 2) * (-(tf.pow(lambda_p * epsilon, 1 - Y) * tf.exp(-lambda_p * epsilon)) + (1 - Y) * (g_1(0.0, Y) - g_1(lambda_p * epsilon, Y)))
    term2 = (1 / nu) * tf.pow(lambda_n, Y - 2) * (-(tf.pow(lambda_n * epsilon, 1 - Y) * tf.exp(-lambda_n * epsilon)) + (1 - Y) * (g_1(0.0, Y) - g_1(lambda_n * epsilon, Y)))
    return term1 + term2

def omega(epsilon, nu, lambda_n, lambda_p, Y):
    term1 = (1 / nu) * (tf.pow(lambda_p, Y) * g_2(lambda_p * epsilon, Y) - tf.pow(lambda_p - 1, Y) * g_2((lambda_p - 1) * epsilon, Y))
    term2 = (1 / nu) * (tf.pow(lambda_n, Y) * g_2(lambda_n * epsilon, Y) - tf.pow(lambda_n + 1, Y) * g_2((lambda_n + 1) * epsilon, Y))
    return term1 + term2

def y_j(j):
    if 1 <= j < 50:
        return 0.01 * j
    elif 50 <= j < 60:
        return 0.05 * (j - 50) + 0.5
    elif 60 <= j < 75:
        return 0.2 * (j - 60) + 1
    elif -75 <= j <= -1:
        return -y_j(-j)
    else:
        return 0.0

def compute_integral_with_rectangles(w, w_y_plus_j, w_y_minus_j, y_plus_j, y_minus_j, lambda_n, lambda_p, nu):
    batch_size = tf.shape(w)[0]
    integral = tf.zeros_like(w, dtype=tf.float32)

    for i in tf.range(batch_size):
        term1 = (1/2) * (w_y_plus_j[i, 0] - w[i]) * k(y_plus_j[0], lambda_n[i], lambda_p[i], nu[i]) * (y_plus_j[1] - y_plus_j[0])
        term2 = tf.reduce_sum((1/2) * (w_y_plus_j[i, 1:73] - w[i]) * k(y_plus_j[1:73], lambda_n[i], lambda_p[i], nu[i]) * (y_plus_j[2:] - y_plus_j[:72]))
        term3 = (1/2) * (w_y_plus_j[i, 73] - w[i]) * k(y_plus_j[73], lambda_n[i], lambda_p[i], nu[i]) * (y_plus_j[73] - y_plus_j[72])

        term4 = (1/2) * (w_y_minus_j[i, 0] - w[i]) * k(y_minus_j[0], lambda_n[i], lambda_p[i], nu[i]) * (y_minus_j[0] - y_minus_j[1])
        term5 = tf.reduce_sum((1/2) * (w_y_minus_j[i, 1:73] - w[i]) * k(y_minus_j[1:73], lambda_n[i], lambda_p[i], nu[i]) * (y_minus_j[:72] - y_minus_j[2:]))
        term6 = (1/2) * (w_y_minus_j[i, 73] - w[i]) * k(y_minus_j[73], lambda_n[i], lambda_p[i], nu[i]) * (y_minus_j[72] - y_minus_j[73])

        integral = tf.tensor_scatter_nd_add(integral, [[i, 0]], term1 + term2 + term3 + term4 + term5 + term6)

    return integral

def compute_inner_part(nu, epsilon, lambda_n, lambda_p, ddw_dx_dx, dw_dx):
    return (1/2) * (ddw_dx_dx - dw_dx) * sigma_square(nu, epsilon, lambda_n, lambda_p, 0)

def compute_outer_part(w, w_y_plus_j, w_y_minus_j, y_plus_j, y_minus_j, nu, dw_dx, epsilon, lambda_n, lambda_p):
    integral = compute_integral_with_rectangles(w, w_y_plus_j, w_y_minus_j, y_plus_j, y_minus_j, lambda_n, lambda_p, nu)
    # numerical explosion of omega
    omega_ = omega(epsilon, nu, lambda_n, lambda_p, 0)
    condition_nan = tf.math.logical_not(tf.math.is_nan(omega_))
    LEFT_BORN = -10000
    RIGHT_BORN = 10000
    condition_not_gt_100 = tf.math.logical_and(tf.math.greater(omega_, LEFT_BORN), tf.math.less_equal(omega_, RIGHT_BORN))
    appropriate_values = tf.math.logical_and(condition_nan, condition_not_gt_100)
    appropriate_index = tf.where(appropriate_values)[:, 0]

    integral = tf.gather(integral, appropriate_index)
    omega_ = tf.reshape(tf.gather(omega_, appropriate_index), [tf.shape(integral)[0], 1])

    return omega_, integral, appropriate_index

def compute_integral_term(nu, epsilon, lambda_n, lambda_p, w, dw_dx, ddw_dx_dx, w_y_plus_j, w_y_minus_j, y_plus_j, y_minus_j):
    # be careful, in tensorflow, (200, 1) + (200, ) gives a (200, 200) tensor !
    # same for the "*" operation...
    inner_part = tf.reshape(compute_inner_part(nu, epsilon, lambda_n, lambda_p, ddw_dx_dx, dw_dx), [200, 1])
    omega_, integral, appropriate_index = compute_outer_part(w, w_y_plus_j, w_y_minus_j, y_plus_j, y_minus_j, nu, dw_dx, epsilon, lambda_n, lambda_p)

    w = tf.gather(w, appropriate_index)
    dw_dx = tf.gather(dw_dx, appropriate_index)
    inner_part = tf.gather(inner_part, appropriate_index)
    outer_part = integral + tf.reshape(tf.reshape(omega_, [-1]) * tf.reshape(dw_dx, [-1]), [tf.shape(omega_)[0],1])

    return w, dw_dx, inner_part + outer_part, appropriate_index
