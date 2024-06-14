import tensorflow as tf
import numpy as np

class EntryInitializer:
    def __init__(self, x_train_range, x_test_range, tau_range, sigma_range, nu_range, theta_range, r_range, q_range):
        self.x_train_range = x_train_range
        self.x_test_range = x_test_range
        self.tau_range = tau_range
        self.sigma_range = sigma_range
        self.nu_range = nu_range
        self.theta_range = theta_range
        self.r_range = r_range
        self.q_range = q_range

    def initialize_all_except_x(self):
        tau = tf.random.uniform(shape=[1], minval=self.tau_range[0], maxval=self.tau_range[1], dtype=tf.float32)
        sigma = tf.random.uniform(shape=[1], minval=self.sigma_range[0], maxval=self.sigma_range[1], dtype=tf.float32)
        nu = tf.random.uniform(shape=[1], minval=self.nu_range[0], maxval=self.nu_range[1], dtype=tf.float32)
        theta = tf.random.uniform(shape=[1], minval=self.theta_range[0], maxval=self.theta_range[1], dtype=tf.float32)
        r = tf.random.uniform(shape=[1], minval=self.r_range[0], maxval=self.r_range[1], dtype=tf.float32)
        q = tf.random.uniform(shape=[1], minval=self.q_range[0], maxval=self.q_range[1], dtype=tf.float32)

        return tf.concat([tau, sigma, nu, theta, r, q], axis = 0)

    def initialize_one_entry_for_training(self):
        x = tf.random.uniform(shape=[1], minval=self.x_train_range[0], maxval=self.x_train_range[1], dtype=tf.float32)
        other_parameters = self.initialize_all_except_x()
        return tf.concat([x, other_parameters], axis=0)
    
    def initialize_n_entries_for_training(self, n):
        return tf.stack([self.initialize_one_entry_for_training() for _ in range(n)])

    def initialize_one_entry_for_test(self):
        x = tf.random.uniform(shape=[1], minval=self.x_test_range[0], maxval=self.x_test_range[1], dtype=tf.float32)
        other_parameters = self.initialize_all_except_x()
        return tf.concat([x, other_parameters], axis=0)
    
    def initialize_n_entries_for_test(self, n):
        return tf.stack([self.initialize_one_entry_for_test() for _ in range(n)])

def generate_training_dataset(data, batch_size):
    return tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(batch_size)

def generate_test_dataset(data, batch_size):
    return tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
