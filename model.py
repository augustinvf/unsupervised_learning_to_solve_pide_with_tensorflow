import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self, x_dimension, layer_dimensions, activation_function, initialisation_distribution):
        super(MLP, self).__init__()

        L = len(layer_dimensions)
        assert L >= 1, "At least one layer"

        self.L = L
        self.initialisation_distribution = initialisation_distribution

        initializer = self.compute_initializer()

        self.model = tf.keras.Sequential()

        # Add input layer
        self.model.add(tf.keras.layers.Dense(layer_dimensions[0], input_dim=x_dimension, activation=activation_function))

        # Add hidden layers
        for i in range(1, L):
           self.model.add(tf.keras.layers.Dense(layer_dimensions[i], kernel_initializer=initializer, activation = activation_function))

        # Add output layer
        self.model.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

    def call(self, inputs):
        return self.model(inputs)
    
    def compute_initializer(self):
        if self.initialisation_distribution == "He-normal" :
            return tf.keras.initializers.HeNormal()
        if self.initialisation_distribution == "He-uniform" :
            return tf.keras.initializers.HeUniform()
        if self.initialisation_distribution == "LeCun-normal" :
            return tf.keras.initializers.LecunNormal()
        if self.initialisation_distribution == "LeCun-uniform" :
            return tf.keras.initializers.LecunUniform()
        if self.initialisation_distribution == "Glorot-normal" :
            return tf.keras.initializers.GlorotNormal()        
        if self.initialisation_distribution == "Glorot-uniform" :
            return tf.keras.initializers.GlorotUniform()

