import tensorflow as tf
import numpy as np

from integral import compute_integral_term
from loss import compute_loss
from intermediate_data import generate_entries_trapezoidal_rule

def forward_pass(model, optimizer, mini_batch, batch_size, x_max, x_min, strike_K, epsilon, y_plus_j, y_minus_j, device, is_training, last_eval):
    loss_epoch = 0
    rmse = 0
    mae = 0

    if is_training:
        model.trainable = True
    else:
        model.trainable = False

    # X where tau = 0
    zeros_column = tf.zeros_like(mini_batch[:, 1])
    entry_x_zero_partial = tf.concat([mini_batch[:, 0:1], zeros_column[:, tf.newaxis]], axis=1)
    entry_x_zero = tf.Variable(tf.concat([entry_x_zero_partial, mini_batch[:, 2:]], axis = 1))

    # X where x = x_min
    x_min_column = tf.constant(x_min, shape=tf.shape(mini_batch[:, 0]), dtype=tf.float32)
    entry_xmin_tau = tf.Variable(tf.concat([x_min_column[:, tf.newaxis], mini_batch[:, 1:]], axis = 1))

    # X where x = x_max
    x_max_column = tf.constant(x_max, shape=tf.shape(mini_batch[:, 0]), dtype=tf.float32)
    entry_xmax_tau = tf.Variable(tf.concat([x_max_column[:, tf.newaxis], mini_batch[:, 1:]], axis = 1))

    # X (x + y_j) and X (x - y_j)
    entry_plus_y = tf.Variable(generate_entries_trapezoidal_rule(mini_batch, y_plus_j))
    entry_minus_y = tf.Variable(generate_entries_trapezoidal_rule(mini_batch, y_minus_j))

    # standard X
    entry = tf.Variable(mini_batch)

    with tf.GradientTape() as g_loss:
        with tf.GradientTape() as g_second_derivative:
            g_second_derivative.watch(entry)
            with tf.GradientTape() as g_derivatives:
                g_derivatives.watch(entry)
                w = model(entry)
            # computing the derivatives
            dw = g_derivatives.gradient(w, entry)
        ddw = g_second_derivative.gradient(dw, entry)

        dw_dx = dw[:, 0]
        dw_dtau = dw[:, 1]

        ddw_dxdx = ddw[:, 0]

        sigma = entry[:, 2]
        nu = entry[:, 3]
        theta = entry[:, 4]
        lambda_p = tf.sqrt((tf.square(theta) / sigma ** 4) + (2 / (tf.square(sigma) * nu))) - (theta / tf.square(sigma))
        lambda_n = tf.sqrt((tf.square(theta) / sigma ** 4) + (2 / (tf.square(sigma) * nu))) + (theta / tf.square(sigma))
        
        w_plus_j_list = []
        w_minus_j_list = []

        for i in range(batch_size) :
            X_plus_j = tf.reshape(entry_plus_y[i, :, :], (74, 7))
            X_minus_j = tf.reshape(entry_minus_y[i, :, :], (74, 7))

            # we first reshape the output in a solo vector of shape (74)
            # then we add a dimension, useful when we will concatenate the results for all the examples
            w_plus = tf.expand_dims(tf.reshape(model(X_plus_j), [-1]), axis = 0)
            w_minus = tf.expand_dims(tf.reshape(model(X_minus_j), [-1]), axis = 0)

            w_plus_j_list.append(w_plus)
            w_minus_j_list.append(w_minus)

        # w_plus_j contains the result of the 74 x+y_j for all entries
        # So w_plus_j has the shape : (200, 74)
        w_plus_j = tf.concat(w_plus_j_list, axis = 0)
        w_minus_j = tf.concat(w_minus_j_list, axis = 0)

        w, dw_dx, integral_term_for_mini_batch, appropriate_index = compute_integral_term(nu, epsilon, lambda_n, lambda_p, w, dw_dx, ddw_dxdx, w_plus_j, w_minus_j, y_plus_j, y_minus_j)
        # computing the boundary conditions
        w_x_zero = model(entry_x_zero)
        w_xmin = model(entry_xmin_tau)
        w_xmax = model(entry_xmax_tau)

        # appropriate outputs
        entry = tf.gather(entry, appropriate_index)
        dw_dtau = tf.gather(dw_dtau, appropriate_index)
        w_x_zero = tf.gather(w_x_zero, appropriate_index)
        w_xmin = tf.gather(w_xmin, appropriate_index)
        w_xmax = tf.gather(w_xmax, appropriate_index)

        loss = compute_loss(entry, x_min, strike_K, w, dw_dtau, dw_dx, w_x_zero, w_xmin, w_xmax, integral_term_for_mini_batch)

    loss_epoch += loss
    print(loss)

    # if last_eval, we compute the RMSE and the MAE
    if last_eval:
        rmse = tf.square(loss)
        mae = tf.abs(loss)

    # if is_training, we do the backward_pass
    if is_training:
        gradients = g_loss.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_epoch, rmse, mae
