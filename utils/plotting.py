from pathlib import Path
import matplotlib.pyplot as plt

def plot_losses_and_accuracies(inner_losses_dict, outer_losses_dict, acc_training_list, average_acc_diff, kmeans_acc):
    Path(Path(__file__).parent.parent / "visualization" / "plots").mkdir(parents=True, exist_ok=True)
    plt.figure()
    for dataset_name in inner_losses_dict:
        plt.plot(
            inner_losses_dict[dataset_name],
            label=f"{dataset_name} - inner loss"
        )
        plt.plot(
            outer_losses_dict[dataset_name],
            linestyle="--",
            label=f"{dataset_name} - outer loss"
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Losses per Dataset")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(__file__).parent.parent / "visualization" / "plots" / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(acc_training_list, label = "Target network accuracy")
    kmeans_acc = [kmeans_acc] * len(acc_training_list)
    plt.plot(kmeans_acc, label = "Kmeans accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(__file__).parent.parent / "visualization" / "plots" / "accuracy.png")
    plt.close()

    if len(average_acc_diff) > 0:
        plt.figure()
        plt.plot(average_acc_diff, label="Accuracy difference train vs no_train")

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracies averaged")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(__file__).parent.parent / "visualization" / "plots" / "accuracy_diff.png")
        plt.close()
