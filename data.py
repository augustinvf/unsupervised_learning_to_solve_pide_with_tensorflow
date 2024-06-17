import tensorflow as tf

class EntryInitializer:
    def __init__(self, x_train_range, x_test_range, tau_range, sigma_range, nu_range, theta_range, r_range, q_range, training_size):
        self.x_train_range = x_train_range
        self.x_test_range = x_test_range
        self.tau_range = tau_range
        self.sigma_range = sigma_range
        self.nu_range = nu_range
        self.theta_range = theta_range
        self.r_range = r_range
        self.q_range = q_range
        self.nb_parameters = 7
        self.parameters_ranges = [
            self.tau_range,
            self.sigma_range,
            self.nu_range,
            self.theta_range,
            self.r_range,
            self.q_range,
        ]
        self.training_size = training_size

        # training samples
        self.sobol_samples = tf.math.sobol_sample(7, training_size)


    def initialize_all_except_x(self):
        # Generate Sobol samples
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
    
    def initialize_entries_for_training(self):
        # Initialize a TensorArray for x_entries
        x_entries = tf.TensorArray(tf.float32, size=self.training_size)

        for i in range(self.training_size):
            x = tf.random.uniform(
                shape=[1], 
                minval=self.x_train_range[0], 
                maxval=self.x_train_range[1], 
                dtype=tf.float32
            )
            x_entries = x_entries.write(i, x)

        # Convert TensorArray to tensor
        x_entries = x_entries.stack()
        x_entries = tf.reshape(x_entries, [self.training_size, 1])  # Reshape to match dimensions

        # Scale the Sobol samples to the parameter ranges
        scaled_samples = []
        for i, (min_val, max_val) in enumerate(self.parameters_ranges):
            scaled_samples.append(min_val + (max_val - min_val) * self.sobol_samples[:, i])

        other_parameters = tf.stack(scaled_samples, axis=1)

        # Concatenate along axis 1 to get samples with shape (training_size, num_params + 1)
        return tf.concat([x_entries, other_parameters], axis=1)

    def initialize_one_entry_for_test(self):
        x = tf.random.uniform(shape=[1], minval=self.x_test_range[0], maxval=self.x_test_range[1], dtype=tf.float32)
        other_parameters = self.initialize_all_except_x()
        return tf.concat([x, other_parameters], axis=0)
    
    def initialize_n_entries_for_test(self, n):
        return tf.stack([self.initialize_one_entry_for_test() for _ in range(n)])

    def generate_training_dataset(self, data, batch_size):
        return tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=self.training_size).batch(batch_size)

    def generate_test_dataset(data, batch_size):
        return tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
