import numpy as np
from .loss_functions import CategoricalCrossEntropy


class ReLU:
    """Rectified linear activation function."""

    inputs = None
    output = None
    d_inputs = None

    def forward(self, inputs):
        """Forward pass."""

        self.inputs = inputs.copy()
        self.output = np.maximum(0, inputs)

    def backward(self, inputs):
        """Backward pass."""

        self.d_inputs = inputs.copy()
        self.d_inputs[self.inputs <= 0] = 0


class Softmax:
    """Softmax activation function."""

    inputs = None
    output = None
    d_inputs = None

    def forward(self, inputs):
        """Forward pass."""

        self.inputs = inputs
        e_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = e_x / np.sum(e_x, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, inputs):
        """Backward pass"""

        self.d_inputs = np.empty_like(inputs)
        for index, (single_output, single_inputs) in enumerate(
            zip(self.output, inputs)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.d_inputs[index] = np.dot(jacobian_matrix, single_inputs)


class SoftmaxCCE:
    """Softmax with 'categorical cross-entropy loss' activation function."""

    activation = Softmax()
    loss = CategoricalCrossEntropy()
    output = None
    d_inputs = None

    def forward(self, inputs, truths=None):
        """Forward pass."""

        self.activation.forward(inputs)
        self.output = self.activation.output

        if truths is not None:
            return self.loss.calculate(self.output, truths)

        return self.output

    def backward(self, inputs, truths):
        """Backward pass"""

        samples = len(inputs)
        if len(truths.shape) == 2:
            truths = np.argmax(truths, axis=1)

        self.d_inputs = inputs.copy()
        self.d_inputs[range(samples), truths] -= 1
        self.d_inputs = self.d_inputs / samples
