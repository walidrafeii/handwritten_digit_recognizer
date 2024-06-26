import numpy as np
import random

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
  """Derivative of the sigmoid function"""
  return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
  """
  Defines a simple deep learning network.
  A deep learning network for simplicity consists mainly of the following:
  1. Input Layer: this is the first item in `sizes`
  2. Hidden Layer: this is the layers between sizes[1: len(sizes) - 1]
  3. Output Layer: this is the last layer sizes[-1]
  4. Biases: A vector that defines the bias from one layer to the next.
  5. Weights: A vector that defines the weights from one layer to the next.

  Args:
      object : _description_
  """
  def __init__(self, sizes) -> None:
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feed_forward(self, a):
    """ Return the output of the network if `a` is input"""
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a
  
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """
    Train the neural network using mini-batch stochastic gradient descent.
    The `training_data` is a list of tuples (x,y) representing the training inputs
    and the desired outputs. If `test_data` is provided then the network will be
    evaluated against the test data after each epoch, and partial progress printed
    out. This is useful for tracking progress, but slow things down substantially.

    Args:
        training_data: The training data
        epochs : Number of times for training data to pass through the algorithm.
        mini_batch_size : size of mini-batches to use when sampling
        eta : Learning rate
        test_data: Test data to evaluate after each epoch
    """
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
      else:
        print(f"Epoch {j} complete")

  def update_mini_batch(self, mini_batch, eta):
    """
    Update the network's weights and biases by applying the gradient descent
    using backpropagation to a single mini batch.
    Args:
        mini_batch: is a list of tuples `(x,y)
        eta : Learning rate
    """
    biases = [np.zeros(b.shape) for b in self.biases]
    weights = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_biases, delta_weights = self.backprop(x,y)
      biases = [new_biases + delta_biases for new_biases, delta_biases in zip(biases, delta_biases)]
      weights = [new_weights + delta_weights for new_weights, delta_weights in zip(weights, delta_weights)]

    self.weights = [weights - (eta / len(mini_batch)) * new_weights 
                    for weights, new_weights in zip(self.weights, weights)]
    
    self.biases = [biases - (eta / len(mini_batch)) * new_biases 
                    for biases, new_biases in zip(self.biases, biases)]
    
  def backprop(self, x, y):
    """
    Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
    `nabla_b` and `nabla_w` are layer-by-layer lists of numpy arrays, similar to `self.biases`
    and `self.weights`
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * \
        sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)


  def evaluate(self, test_data):
    """
    Return The number of test inputs for which the neural network outputs
    the correct result. Note that the neural network's output is assumed
    to be the index of whichever neuron in the final layer has the highest
    activation.
    """
    test_results = [(np.argmax(self.feed_forward(x)), y) for (x,y) in test_data]
    return sum(int(x == y) for (x,y) in test_results)
  
  def cost_derivative(self, output_activations, y):
    """
    Returns the vector of partial derivatives \partial C_x / \partial a for the
    output of activations.
    """
    return (output_activations - y)
    
