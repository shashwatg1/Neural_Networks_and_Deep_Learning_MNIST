#this part makes use of network.py

# we first import the mnist dataset using the mnist_loader.py file
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
# we create a network of 784 input, 30 hidden, 10 output neurons
net = network.Network([784, 30, 10])

# we use stochastic gradient descent to learn training data over 20 epochs, mini batch size of 10, learning rate 3.0
net.SGD(training_data, 20, 10, 3.0, test_data=test_data)

# by using more number of hidden units say 100 and more epochs say 30, accuracy will improve