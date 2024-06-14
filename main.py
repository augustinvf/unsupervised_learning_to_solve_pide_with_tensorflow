import time
import pickle
import tensorflow as tf
from math import log, sqrt
from omegaconf import OmegaConf
from model import MLP
from data import EntryInitializer, generate_training_dataset, generate_test_dataset
from training import forward_pass
from integral import y_j

start_time = time.time()

# To load the hyperparameters
config = OmegaConf.load("config.yaml")

# Check GPU availability
if tf.config.experimental.list_physical_devices('GPU'):
    device = '/GPU:0'
else:
    device = '/CPU:0'

# Hyperparameters to generate the instance of the model
strike_K = config.K

batch_size = config.batch_size
training_size = config.training_size
test_size = config.test_size

x_train_range = [log(strike_K / 40), log(2*strike_K)]
x_test_range = [log(strike_K / 2), log(2*strike_K)]
tau_range = config.tau_range
sigma_range = config.sigma_range
nu_range = config.nu_range
theta_range = config.theta_range
r_range = config.r_range
q_range = config.q_range

data_initializer = EntryInitializer(x_train_range, x_test_range, tau_range, sigma_range, nu_range, theta_range, r_range, q_range)
data_training = data_initializer.initialize_n_entries_for_training(training_size)
training_dataset = generate_training_dataset(data_training, batch_size)

data_test = data_initializer.initialize_n_entries_for_training(test_size)
test_dataset = generate_test_dataset(data_test, batch_size)

x_dimension = 7

layer_dimensions = config.layer_dimensions

activation_function = config.activation_function

initialisation_distribution = config.initialisation_distribution

# Generating the model
model = MLP(x_dimension, layer_dimensions, activation_function, initialisation_distribution)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=config.optimizer.params.lr, 
                                     weight_decay=config.optimizer.params.weight_decay)

training_epochs = config.training_epochs

# Special points for computing the integral term

y_plus_j = tf.TensorArray(tf.float32, size=74)
y_minus_j = tf.TensorArray(tf.float32, size=74)

for j in range(1, 75):
    y_plus_j = y_plus_j.write(j-1, y_j(j))
    y_minus_j = y_minus_j.write(j-1, y_j(-j))

# Convertir TensorArray en Tensor
y_plus_j = y_plus_j.stack()
y_minus_j = y_minus_j.stack()

# Parameters for the integral
epsilon = config.epsilon
x_max = config.x_max
x_min = config.x_min


list_loss_training = []
list_loss_test = []
last_eval = False

for epoch in range(training_epochs):
    print("1 more epoch")

    # TRAINING
    is_training = True
    for mini_batch in training_dataset:
        loss_training, _, _ = forward_pass(model, optimizer, mini_batch, batch_size, x_max, x_min, strike_K, epsilon, y_plus_j, y_minus_j, device, is_training, last_eval)
        list_loss_training.append(loss_training.numpy())

#     # TEST
#     is_training = False
#     for mini_batch in test_dataset:
#         loss_test, _, _ = forward_pass(model, optimizer, mini_batch, batch_size, strike_K, epsilon, y_plus_j, y_minus_j, device, is_training, last_eval)
#         list_loss_test.append(loss_test.numpy())

# # COMPUTE RMSE and MAE
# last_eval = True
# rmse = 0
# mae = 0

# for mini_batch in test_dataset:
#     _, mini_batch_rmse, mini_batch_mae = forward_pass(model, optimizer, mini_batch, batch_size, strike_K, epsilon, y_plus_j, y_minus_j, device, is_training, last_eval)
#     rmse += mini_batch_rmse
#     mae = mini_batch_mae if mae < mini_batch_mae else mae

# number_of_mini_batches = test_size // batch_size
# rmse = sqrt(rmse / number_of_mini_batches)

# end_time = time.time()
# execution_time = end_time - start_time

# result = {
#     "list loss training": list_loss_training,
#     "list loss test": list_loss_test,
#     "rmse": rmse,
#     "mae": mae,
#     "execution time": execution_time,
#     "training size": training_size,
#     "test_size": test_size,
#     "nb of epochs": training_epochs,
#     "activation function": config.activation_function,
#     "initial distribution": initialisation_distribution
# }

# with open('result/result_saver.pkl', 'wb') as file:
#     pickle.dump(result, file)
