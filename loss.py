import tensorflow as tf

def compute_loss(entry, x_min, K, w, dw_dtau, dw_dx, w_x_zero, w_xmin, w_xmax, integral_term_for_mini_batch):
    x = entry[:, 0]
    tau = entry[:, 1]
    r = entry[:, 5]
    q = entry[:, 6]

    term_1 = tf.square(integral_term_for_mini_batch - dw_dtau + (r - q) * dw_dx - r * w)
    term_2 = tf.square(w_x_zero - tf.maximum(K - tf.exp(x), tf.zeros_like(x)))
    term_3 = tf.square(w_xmin - (K * tf.exp(-r * tau) - tf.exp(x_min - q * tau)))
    term_4 = tf.square(w_xmax)

    loss = term_1 + term_2 + term_3 + term_4

    print("t1 loss:", tf.reduce_mean(term_1))
    print("t2 loss:", tf.reduce_mean(term_2))
    print("t3 loss:", tf.reduce_mean(term_3))
    print("t4 loss:", tf.reduce_mean(term_4))

    return tf.reduce_mean(loss)
