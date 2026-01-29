from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt


def plot_losses_and_accuracies(
    inner_losses_dict: dict[str, list[float]],
    outer_losses_dict: dict[str, list[float]],
    acc_training_list: list[float],
    average_acc_diff: list[float],
    kmeans_acc: float,
    output_dir: Union[str, Path],
):
    """
    Plot training losses and accuracies.

    :param inner_losses_dict: Dictionary mapping dataset names to inner loss lists.
    :param outer_losses_dict: Dictionary mapping dataset names to outer loss lists.
    :param acc_training_list: List of training accuracies.
    :param average_acc_diff: List of average accuracy differences.
    :param kmeans_acc: KMeans accuracy value.
    :param output_dir: Directory to save plots (will create 'plots/' subdirectory)
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for dataset_name in inner_losses_dict:
        plt.plot(inner_losses_dict[dataset_name], label=f"{dataset_name} - inner loss")
        plt.plot(
            outer_losses_dict[dataset_name],
            linestyle="--",
            label=f"{dataset_name} - outer loss",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Losses per Dataset")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(acc_training_list, label="Target network accuracy")
    kmeans_acc_line = [kmeans_acc] * len(acc_training_list)
    plt.plot(kmeans_acc_line, label="Kmeans accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "accuracy.png")
    plt.close()

    if len(average_acc_diff) > 0:
        plt.figure()
        plt.plot(average_acc_diff, label="Accuracy difference train vs no_train")

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracies averaged")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / "accuracy_diff.png")
        plt.close()
