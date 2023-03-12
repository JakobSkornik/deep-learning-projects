import numpy as np
from .activation_functions import ReLU, SoftmaxCCE


class DenseLayer:
    """
    Dense layer class. This class performs base math operations
    between weights and biases.
    """

    weights = None
    biases = None
    d_weights = None
    d_biases = None
    d_inputs = None

    def __init__(self, num_of_inputs, num_of_neurons):
        """Initialize DenseLayer class."""

        self.weights = 2 * np.random.random((num_of_inputs, num_of_neurons)) - 1
        self.biases = np.zeros((1, num_of_neurons))

    def forward(self, inputs):
        """Forward pass."""

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, inputs):
        """Backward pass."""

        self.d_weights = np.dot(self.inputs.T, inputs)
        self.d_biases = np.sum(inputs, axis=0, keepdims=True)
        self.d_inputs = np.dot(inputs, self.weights.T)


class HiddenLayer:
    """Hidden layer class."""

    dense_layer = None
    activation_function = ReLU()

    def __init__(
        self,
        num_of_inputs: int = 10,
        num_of_neurons: int = 10,
    ) -> None:
        """Initializes a layer object."""

        self.dense_layer = DenseLayer(num_of_inputs, num_of_neurons)

    def forward(self, inputs):
        """Forward pass."""

        self.dense_layer.forward(inputs)
        self.activation_function.forward(self.dense_layer.output)
        return self.activation_function.output

    def backward(self, inputs):
        """Backward pass."""

        # print(np.round(inputs, 2))
        # print("_________")
        self.activation_function.backward(inputs)
        # print(np.round(inputs, 2))
        # print("_________")
        # print(np.round(self.activation_function.d_inputs, 2))
        # print("===============================")
        self.dense_layer.backward(self.activation_function.d_inputs)


class OutputLayer:
    """Output layer class."""

    dense_layer = None
    activation_function = SoftmaxCCE()

    def __init__(
        self,
        num_of_inputs: int = 10,
        num_of_neurons: int = 10,
    ) -> None:
        """Initializes a layer object."""

        self.dense_layer = DenseLayer(num_of_inputs, num_of_neurons)

    def forward(self, inputs, truths=None):
        """Forward pass."""

        self.dense_layer.forward(inputs)
        loss = self.activation_function.forward(self.dense_layer.output, truths)
        return loss

    def backward(self, inputs, truths):
        """Backward pass."""

        self.activation_function.backward(inputs, truths)
        self.dense_layer.backward(self.activation_function.d_inputs)
