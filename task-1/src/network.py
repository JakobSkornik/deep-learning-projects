import time
import numpy as np
from typing import Union

from .activations import softmax, sigmoid, sigmoid_prime, softmax_dLdZ
from .losses import cross_entropy
from .util import shuffle


class Network(object):
    weights = []
    biases = []
    eta = 0.001
    eta_decay = None
    lamda = None
    input_size = 0
    num_hidden_layers = 0
    optimizer = "sgd"
    validation_k = 5
    activation = sigmoid
    act_prime = sigmoid_prime
    shuffle = True

    def train(
        self,
        training_data: np.ndarray,
        training_class: np.ndarray,
        val_data: np.ndarray,
        val_class: np.ndarray,
        epochs: int,
        mini_batch_size: int = 16,
    ):
        i = 0
        eta_current = self.eta
        n = training_data.shape[1]

        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []

        for j in range(epochs):
            loss_avg = 0.0
            acc_avg = 0.0
            start_time = time.time()

            if self.shuffle:
                training_data, training_class = shuffle(training_data, training_class)

            mini_batches = [
                (
                    training_data[:, k : k + mini_batch_size],
                    training_class[:, k : k + mini_batch_size],
                )
                for k in range(0, n, mini_batch_size)
            ]

            for idx, mini_batch in enumerate(mini_batches):
                data = mini_batch[0]
                truths = mini_batch[1]
                Y, Zs, As = self.forward_pass(data)
                gw, gb, rw = self.backward_pass(Y, truths, Zs, As)
                self.update_network(gw, gb, rw, eta_current)

                if self.eta_decay:
                    eta_current = self.eta * np.exp(-self.eta_decay * i)

                loss = cross_entropy(truths, Y)

                loss_avg += loss
                acc = self.accuracy(truths, Y)
                acc_avg += acc
                i += 1
                # print(f"Batch {idx}/{len(mini_batches)}...", end="\r")

            loss_norm = loss_avg / len(mini_batches)
            acc_norm = acc_avg / len(mini_batches)
            losses.append(loss_norm)
            accuracies.append(acc_norm)

            if j % self.validation_k == 0:
                val_loss, val_acc = self.eval_network(val_data, val_class)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                print(
                    f"[{j}] L: {loss_norm:.4f} A: {acc_norm:.4f} VL: {val_loss:.4f} VA: {val_acc:.4f} t: {time.time() - start_time:.2f}s",
                    end="\r\n\r",
                )
            else:
                print(
                    f"[{j}] L: {loss_norm:.4f} A: {acc_norm:.4f} t: {time.time() - start_time:.2f}s",
                    end="\r\n\r",
                )

        return losses, accuracies, val_losses, val_accuracies

    def update_network(self, gw, gb, rw, eta):
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * (gw[i] + rw[i])
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            b1 = self.beta_1
            b2 = self.beta_2
            e = self.epsilon
            for i in range(len(self.weights)):
                self.w_momentums[i] = b1 * self.w_momentums[i] + (1 - b1) * gw[i]
                self.w_cache[i] = b2 * self.w_cache[i] + (1 - b2) * np.square(gw[i])

                self.b_momentums[i] = b1 * self.b_momentums[i] + (1 - b1) * gb[i]
                self.b_cache[i] = b2 * self.b_cache[i] + (1 - b2) * np.square(gb[i])

                aw = self.w_momentums[i] / (np.sqrt(self.w_cache[i]) + e)
                ab = self.b_momentums[i] / (np.sqrt(self.b_cache[i]) + e)

                self.weights[i] -= eta * (aw + rw[i])
                self.biases[i] -= eta * ab
        else:
            raise ValueError("Unknown optimizer:" + self.optimizer)

    def forward_pass(self, input: tuple[np.ndarray, np.ndarray]):
        Zs = []
        activations = [input]
        output_layer_idx = len(self.weights) - 1

        for i in range(len(self.weights)):
            prev_layer_output = activations[-1]
            z_i = np.dot(self.weights[i], prev_layer_output) + self.biases[i]

            if i == output_layer_idx:
                a_i = softmax(z_i)
            else:
                a_i = self.activation(z_i)

            Zs.append(z_i)
            activations.append(a_i)
        Y = activations[-1]

        return Y, Zs, activations

    def backward_pass(
        self,
        output: tuple[np.ndarray, np.ndarray],
        target: tuple[np.ndarray, np.ndarray],
        Zs: list[tuple[np.ndarray, np.ndarray]],
        activations: list[tuple[np.ndarray, np.ndarray]],
    ):
        batch_len = target.shape[1]
        n_layers = len(self.weights)

        gw = [np.zeros(x.shape) for x in self.weights]
        rw = [np.zeros(x.shape) for x in self.weights]
        gb = [np.zeros(x.shape) for x in self.biases]

        for i in reversed(range(n_layers)):
            if i == n_layers - 1:
                d = softmax_dLdZ(output, target)
            else:
                d = np.dot(np.transpose(self.weights[i + 1]), d) * self.act_prime(Zs[i])
            gw[i] = np.dot(d, np.transpose(activations[i])) / batch_len
            gb[i] = np.sum(d, axis=1, keepdims=True) / batch_len
            if self.lamda:
                rw[i] = (self.lamda / self.input_size) * self.weights[i]
        return gw, gb, rw

    def accuracy(self, truths, predictions):
        if len(truths.shape) == 2:
            predictions = np.argmax(predictions, axis=0)
            truths = np.argmax(truths, axis=0)
        return np.sum(predictions == truths) / len(truths)

    def eval_network(self, validation_data, validation_class):
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:, i], -1)
            example_class = np.expand_dims(validation_class[:, i], -1)
            example_class_num = np.argmax(validation_class[:, i], axis=0)
            output, Zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            loss_avg += loss
        return loss_avg / n, tp / n

    def add_layer(self, neurons: int = 10, type: str = "hidden") -> None:
        if type == "input":
            self.input_size = neurons
        elif type == "hidden":
            if len(self.weights) == 0:
                prev_size = self.input_size
            else:
                prev_size = len(self.weights[-1])
            self.weights.append(
                ((2 / prev_size) ** 0.5) * np.random.randn(neurons, prev_size)
            )
            self.biases.append(np.zeros((neurons, 1)))
        elif type == "output":
            if len(self.weights) == 0:
                raise TypeError("Missing hidden layers.")
            else:
                prev_size = len(self.weights[-1])
                self.weights.append(
                    ((2 / prev_size) ** 0.5) * np.random.randn(neurons, prev_size)
                )
                self.biases.append(np.zeros((neurons, 1)))
        else:
            raise TypeError(f"Invalid layer type: {type}")

    def set(self, property: str, val: Union[float, int, dict, bool]) -> None:
        if property == "optimizer":
            self.optimizer = val["type"]
            if self.optimizer == "adam":
                self.epsilon = val["epsilon"]
                self.beta_1 = val["beta_1"]
                self.beta_2 = val["beta_2"]
                self.w_momentums = [np.zeros_like(x) for x in self.weights]
                self.w_cache = [np.zeros_like(x) for x in self.weights]
                self.b_momentums = [np.zeros_like(x) for x in self.biases]
                self.b_cache = [np.zeros_like(x) for x in self.biases]

        elif property == "learning_rate":
            self.eta = val

        elif property == "decay":
            self.eta_decay = val

        elif property == "validation_k":
            self.validation_k = val

        elif property == "activation":
            self.activation = val["activation"]
            self.act_prime = val["act_prime"]

        elif property == "lambda":
            self.lamda = val

        elif property == "shuffle":
            self.shuffle = val
