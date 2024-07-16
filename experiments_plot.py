"""
Script that plot losses and metrics of selected experiments.
"""
import pickle
from pathlib import Path
from matplotlib import pyplot as plt


if __name__ == "__main__":
    paths_to_plot = [Path("results_mc_kan_search_best_candidate", "kan_training_dataset_men",
                          "68035e02-e891-4b0c-bc3a-b5023f3534e8", "4", "kan_res_1.pickle")]
    for experiment_path in paths_to_plot:
        with open(experiment_path, "rb") as f:
            experiment_results = pickle.load(f)
        uar = [(recall + specificity) / 2 for recall, specificity in zip(experiment_results["test_recall"],
                                                                         experiment_results["test_specificity"])]
        # Create subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

        # First subplot for train and test accuracy
        ax1.plot(experiment_results["train_acc"], label='Train Accuracy', marker='o')
        ax1.plot(experiment_results["test_acc"], label='Test Accuracy', marker='o')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Train and Test Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Second subplot for test specificity and test recall
        ax2.plot(experiment_results["test_specificity"], label='Test Specificity', marker='o')
        ax2.plot(experiment_results["test_recall"], label='Test Recall', marker='o')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metrics')
        ax2.set_title('Test Specificity and Test Recall')
        ax2.legend()
        ax2.grid(True)

        # Third subplot for UAR
        ax3.plot(uar, label='Test UAR', marker='o')

        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Metrics')
        ax3.set_title('Test UAR')
        ax3.legend()
        ax3.grid(True)

        # Fourth subplot for losses
        ax4.plot(experiment_results["train_loss"], label='Train loss', marker='o')
        ax4.plot(experiment_results["test_loss"], label='Test loss', marker='o')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Metrics')
        ax4.set_title('Losses')
        ax4.legend()
        ax4.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()
