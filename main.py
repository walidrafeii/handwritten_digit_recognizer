import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 784 inputs, 30 neurons in hidden layer, 10 outputs.
net = network.Network([784, 30, 10])
net.SGD(list(training_data), 30, 10, 3, test_data=list(test_data))
