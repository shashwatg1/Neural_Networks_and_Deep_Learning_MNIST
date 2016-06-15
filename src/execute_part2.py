#this part makes use of network2.py

# we first import the mnist dataset using the mnist_loader.py file
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2

# we create a network of 784 input, 30 hidden, 10 output neurons
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# initialise weights and biases
net.large_weight_initializer()

# epochs = 20, mini batch size = 10, learning rate = 0.5, lambda = 1.0
net.SGD(training_data, 20, 10, 0.5, lmbda = 1.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# by using more number of hidden units say 100 and more epochs say 30, accuracy will improve
