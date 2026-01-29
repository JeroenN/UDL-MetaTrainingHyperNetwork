from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt



def plot_losses_and_accuracies(
    inner_losses_dict, 
    outer_losses_dict, 
    acc_training_list, 
    average_acc_diff, 
    kmeans_acc,
    output_dir: Union[str, Path],
    name_addition= "_average"
):

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    

    plt.figure()
    if inner_losses_dict is not None:
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
        plt.savefig(plots_dir / f"loss{name_addition}.png")
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
    plt.savefig(plots_dir/ f"accuracy{name_addition}.png")
    plt.close()

    if len(average_acc_diff) > 0:
        plt.figure()
        plt.plot(average_acc_diff, label="Accuracy difference train vs no_train")

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Accuracies averaged")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / f"accuracy_diff{name_addition}.png")
        plt.close()
