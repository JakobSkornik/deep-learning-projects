import numpy as np
from typing import Union
from .layers import HiddenLayer, OutputLayer
from .optimizers import AdaGrad, StochasticGradientDescent


class NeuralNetwork:
    """
    Basic neural network class. Contains code for a simple
    neural network.
    """

    layers = []
    output_layer = None

    def __init__(
        self,
        input_size: int,
        output_size: int = 2,
        iterations: int = 10000,
        hidden_layers: int = 1,
        layer_size: Union[int, list] = 5,
        alpha: float = 1,
        alpha_decay: float = 0.05,
        optimizer: str = "SGD",
        momentum: float = 0.0,
        epsilon: float = 1e-3,
        logs: bool = False,
        log_frequency=1000,
    ) -> None:

        # Validate output size
        if output_size < 2:
            raise Exception(f"Invalid outpud size: {output_size}.")

        # Validate input size
        if input_size < 1:
            raise Exception(f"Invalid input size: {input_size}.")

        # Validate number of hidden layers
        if hidden_layers < 1:
            print("Invalid number of hidden layers. Selecting default (1).")
            hidden_layers = 1
        self.layers = []

        # If layer_size parameter is a list
        if type(layer_size) == "list":

            # Validate input size
            if input_size != layer_size[0]:
                raise Exception("First layer size must match input size.")

            # Validate hidden_layers against layer_size
            if hidden_layers != len(layer_size):
                raise Exception(
                    "Number of layers must match length of layer_size parameter - 1."
                )

            # Create layers
            prev_layer_size = input_size
            for layer in layer_size:
                self.layers.append(HiddenLayer(prev_layer_size, layer))
                prev_layer_size = layer

            # Append output layer
            self.output_layer = OutputLayer(layer_size, output_size)

        # If layer_size parameter is an integer
        else:

            # Create layers
            prev_layer_size = input_size
            for layer in range(hidden_layers):
                self.layers.append(HiddenLayer(prev_layer_size, layer_size))
                prev_layer_size = layer_size

            # Append output layer
            self.output_layer = OutputLayer(layer_size, output_size)

        # Set max iterations
        self.max_iters = iterations

        # Set optimizer
        if optimizer == "adagrad":
            self.optimizer = AdaGrad(alpha, alpha_decay, epsilon)
        self.optimizer = StochasticGradientDescent(
            alpha, alpha_decay, momentum=momentum
        )

        # Configure logs
        self.logs = logs
        self.log_frequency = log_frequency

    def train(self, dataset: list, truths: list) -> None:
        """Train method."""

        dataset = np.array(dataset)
        truths = np.array(truths)

        for i in range(self.max_iters):
            loss, predictions = self.forward(dataset, truths)
            accuracy = self.__accuracy(predictions, truths)

            self.backward(predictions, truths)
            self.optimize()

            if self.logs:
                self.__log(i, accuracy, loss)

        print(f"Finished learning. Accuracy: {accuracy}.")

    def forward(self, val, truths):
        """Forward method."""

        for layer in self.layers:
            val = layer.forward(val)

        loss = self.output_layer.forward(val, truths)
        predictions = self.output_layer.activation_function.output
        return loss, predictions

    def backward(self, val, truths):
        """Backward method."""

        self.output_layer.backward(val, truths)
        prev = self.output_layer.dense_layer.d_inputs

        for layer in reversed(self.layers):
            layer.backward(prev)
            prev = layer.dense_layer.d_inputs

    def optimize(self):
        """Optimize method."""

        self.optimizer.pre_update_params()

        for layer in self.layers:
            self.optimizer.update_params(layer.dense_layer)

        self.optimizer.update_params(self.output_layer.dense_layer)

        self.optimizer.post_update_params()

    def predict(self, input, encoding: dict = None):
        """Predict method."""

        for layer in self.layers:
            layer.forward(input)
            input = layer.activation_function.output

        prediction = self.output_layer.forward(input)
        prediction = np.argmax(prediction, axis=1)[0]

        if not encoding:
            return prediction

        return encoding[prediction]

    def __log(self, i, accuracy, loss):
        """Log method."""

        if i % self.log_frequency == 0:
            print(
                f"{i}: loss: {round(loss, 4)}, accuracy: {round(accuracy, 4)}, learning_rate: {round(self.optimizer.current_learning_rate, 4)}"
            )

    def __accuracy(self, predictions, truths):
        """Calculate accuracy."""

        truths.copy()
        predictions = np.argmax(predictions, axis=1)

        if len(truths.shape) == 2:
            truths = np.argmax(truths, axis=1)

        return np.mean(predictions == truths)
