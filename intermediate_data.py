import tensorflow as tf

# In this file, we will create the x + x_j vectors and the boundary conditions vectors

def generate_entries_trapezoidal_rule(mini_batch, y_j):
    # mini_batch is a tensor by assumption
    tensor_zeros = tf.zeros((74))

    tensor_list = [y_j]
    for _ in range(6):
        tensor_list.append(tensor_zeros) 
    new_tensor = tf.stack(tensor_list)
    new_tensor = tf.expand_dims(new_tensor, 0)
    dimension_modifier = tf.constant([200, 1, 1], tf.int32)
    new_tensor = tf.tile(new_tensor, dimension_modifier)

    entry = mini_batch
    entry = tf.expand_dims(entry, 2)
    dimension_modifier = tf.constant([1, 1, 74], tf.int32)
    entry = tf.tile(entry, dimension_modifier)
    
    return tf.transpose(tf.add(entry, new_tensor), perm = [0, 2, 1])
