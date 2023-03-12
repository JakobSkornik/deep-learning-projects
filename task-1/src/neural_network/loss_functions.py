import numpy as np


class Loss:
    """Simple loss function."""

    def calculate(self, output, truths):
        """Calculate loss based on truths"""
        return np.mean(self.forward(output, truths))


class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy loss function."""

    def forward(self, predictions, truths):
        """Forward pass."""

        samples = len(predictions)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        if len(truths.shape) == 1:
            correct_confidences = predictions_clipped[range(samples), truths]
        elif len(truths.shape) == 2:
            correct_confidences = np.sum(predictions_clipped * truths, axis=1)

        return -np.log(correct_confidences + 1e-7)

    def backward(self, inputs, truths):
        """Backward pass."""

        samples = len(inputs)
        labels = len(inputs[0])

        if len(truths.shape) == 1:
            truths = np.eye(labels)[truths]

        self.d_inputs = -truths / inputs
        self.d_inputs = self.d_inputs / samples
