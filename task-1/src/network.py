import random
from typing import Union
import pandas as pd
import numpy as np
import pickle


class SGD_Params:
    def __init__(self, eta: float, eta_decay: float = 0.0):
        self.eta = eta
        self.eta_decay = eta_decay


class Adam_Params:
    def __init__(
        self,
        eta: float = 0.001,
        eta_decay: float = 0.00001,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        self.eta = eta
        self.eta_decay = eta_decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


class Network(object):
    def __init__(
        self,
        sizes: list[int],
        optimizer: str = "sgd",
        params: Union[SGD_Params, Adam_Params] = SGD_Params(0.001, 0.00001),
    ):
        self.weights = [
            ((2 / sizes[i - 1]) ** 0.5) * np.random.randn(sizes[i], sizes[i - 1])
            for i in range(1, len(sizes))
        ]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer
        self.params = params

        if self.optimizer == "adam":
            self.mw = [np.zeros_like(x) for x in self.weights]
            self.vw = [np.zeros_like(x) for x in self.weights]
            self.mb = [np.zeros_like(x) for x in self.biases]
            self.vb = [np.zeros_like(x) for x in self.biases]

    def train(
        self,
        training_data: np.ndarray,
        training_class: np.ndarray,
        val_data: np.ndarray,
        val_class: np.ndarray,
        epochs: int,
        mini_batch_size: int,
    ):
        i = 0
        eta_current = self.params.eta
        n = training_data.shape[1]

        losses = []
        accuracies = []

        for j in range(epochs):
            loss_avg = 0.0
            training_data, training_class = shuffle(training_data, training_class)
            mini_batches = [
                (
                    training_data[:, k : k + mini_batch_size],
                    training_class[:, k : k + mini_batch_size],
                )
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                Y, Zs, As = self.forward_pass(mini_batch[0])
                gw, gb = self.backward_pass(Y, mini_batch[1], Zs, As)
                self.update_network(gw, gb, eta_current)

                if self.params.eta_decay:
                    eta_current = self.params.eta * np.exp(-self.params.eta_decay * i)
                else:
                    eta_current = self.params.eta
                i += 1

                loss = cross_entropy(mini_batch[1], Y)
                loss_avg += loss

            if j % 1 == 0:
                loss, acc = self.eval_network(val_data, val_class)
                losses.append(loss)
                accuracies.append(acc)
                print(
                    f"Epoch {j} | Loss: {loss:.4f} | Eta: {eta_current:.8f} | Acc: {acc:.4f}",
                    end="\n\r",
                )
            else:
                print(
                    f"Epoch {j} | Loss: {loss_avg / len(mini_batches):.8f} | Eta: {eta_current:.4f}",
                    end="\r",
                )

        return losses, accuracies

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

    def update_network(self, gw, gb, eta):
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * gw[i]
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            b1 = self.params.beta_1
            b2 = self.params.beta_2
            e = self.params.epsilon
            for i in range(len(self.weights)):
                self.mw[i] = b1 * self.mw[i] + (1 - b1) * gw[i]
                self.vw[i] = b2 * self.vw[i] + (1 - b2) * np.square(gw[i])
                
                self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
                self.vb[i] = b2 * self.vb[i] + (1 - b2) * np.square(gb[i])

                aw = self.mw[i] / (np.sqrt(self.vw[i]) + e)
                ab = self.mb[i] / (np.sqrt(self.vb[i]) + e)

                self.weights[i] -= eta * aw
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
                a_i = sigmoid(z_i)

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

        gw = [np.zeros(w.shape) for w in self.weights]
        gb = [np.zeros(b.shape) for b in self.biases]

        for i in reversed(range(n_layers)):
            if i == n_layers - 1:
                d = softmax_dLdZ(output, target)
            else:
                d = np.dot(np.transpose(self.weights[i + 1]), d) * sigmoid_prime(Zs[i])
            gw[i] = np.dot(d, np.transpose(activations[i])) / batch_len
            gb[i] = np.sum(d, axis=1, keepdims=True) / batch_len

        return gw, gb


def softmax(Z: np.ndarray):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def softmax_dLdZ(output, target):
    return output - target


def cross_entropy(y_true, y_pred, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def relu(z: np.ndarray):
    return np.maximum(0, z)


def relu_dLdz(z):
    z[z < 0] = 0
    return z


def sigmoid(z: np.ndarray):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def load_data_cifar(train_file: str, test_file: str):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict["data"]) / 255.0
    train_class = np.array(train_dict["labels"])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict["data"]) / 255.0
    test_class = np.array(test_dict["labels"])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return (
        train_data.transpose(),
        train_class_one_hot.transpose(),
        test_data.transpose(),
        test_class_one_hot.transpose(),
    )


def validation_split(
    dataset: np.ndarray, ground_truths: np.ndarray, pct: float
) -> tuple:
    val_size = int(len(dataset) * pct)
    train_data = dataset[..., val_size:]
    train_class = ground_truths[..., val_size:]
    val_data = dataset[..., :val_size]
    val_class = ground_truths[..., :val_size]
    return train_data, train_class, val_data, val_class


def shuffle(dataset: np.ndarray, truths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idxs = np.arange(dataset.shape[1])
    np.random.shuffle(idxs)
    return dataset[:, idxs], truths[:, idxs]
