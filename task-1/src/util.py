import pickle
import numpy as np
import matplotlib.pyplot as plt


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


def plot_model(acc: list, loss: list, val_acc: list, val_loss: list, k: int) -> None:
    plt.plot(acc)
    plt.plot(np.repeat(val_acc, k))
    plt.title("Accuracy and Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy"], loc="upper right")
    plt.show()

    plt.plot(loss)
    plt.plot(np.repeat(val_loss, k))
    plt.title("Loss and Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"], loc="upper right")
    plt.show()