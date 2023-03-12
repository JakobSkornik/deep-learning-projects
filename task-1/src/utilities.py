import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


def get_dummies(dataset: pd.DataFrame, cols: list) -> pd.DataFrame:
    """This method calls the pd.get_dummies method."""

    return pd.get_dummies(dataset, columns=cols, drop_first=True)

def evaluate_custom(X, y, nn):
    correct = 0
    confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    y = np.argmax(y, axis=1)

    for i in range(len(X)):
        truth = y[i]
        entry = X[i]
        predicted = nn.predict(entry)

        if predicted == truth:
            correct += 1

            if predicted == 1:
                confusion_matrix["TP"] += 1
            else:
                confusion_matrix["TN"] += 1

        else:
            if predicted == 1:
                confusion_matrix["FP"] += 1
            else:
                confusion_matrix["FN"] += 1

    print(
        f"""=======================
RESULTS:

    TP: {confusion_matrix["TP"]},
    TN: {confusion_matrix["TN"]},
    FP: {confusion_matrix["FP"]},
    FN: {confusion_matrix["FN"]}
    accuracy: {correct} / {len(X)} = {round((correct / len(X)) * 100, 2)}%
    sensitivity: {round(confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"]), 2)}
    specificity: {round(confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"]), 2)}
    """
    )
    return confusion_matrix


def visualize_custom(cm_dict: dict) -> None:
    """Visualize confusion matrix."""

    cm = pd.DataFrame(
        np.array([[cm_dict["TN"], cm_dict["FP"]], [cm_dict["FN"], cm_dict["TP"]]])
    )
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")


def split_at_x_percent(dataset: np.ndarray, x: int) -> tuple:
    """Returns two pd.DataFrames by splitting original one at x%."""

    rows = len(dataset)
    idx = int((rows * x) / 100)

    if len(dataset.shape) == 1:
        return dataset[:idx], dataset[idx + 1 :]

    return dataset[:idx, :], dataset[idx + 1 :, :]


def one_hot(a, num_classes):
    """Transforms vector to one-hot encoded vector."""

    if type(a) is list:
        a = np.array(a)

    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
