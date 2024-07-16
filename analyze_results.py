"""
Script to analyze results of kan_arch_mc.py.
"""
import pickle
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    # path to results to analyze. It is expected that directory contains multiple
    # folders with different dataset and each folder contains subfolder
    # representing various architectures (kan_arch_mc.py).
    pickled_results_path = Path(".", "results_mc_kan_search_best_candidate_2", "kan_training_dataset_men")
    for dataset in sorted(pickled_results_path.iterdir()):
        for arch_result in dataset.iterdir():
            print(f"evaluating architecture {arch_result}")
            best_uar = []
            for result in arch_result.glob("*.pickle"):
                # read result for a single train/test split
                with open(result, "rb") as f:
                    experiment_results = pickle.load(f)
                # compute UAR
                uar = [(recall + specificity) / 2 for recall, specificity in
                       zip(experiment_results["test_recall"],
                           experiment_results["test_specificity"])]
                # select best UAR for this train/test split
                best_uar.append(max(uar))
            # prepare results to print
            TO_PRINT = " ".join([f"{num:.4f}" for num in best_uar])
            # print the BEST UAR for each split and MEAN of BEST UAR
            print(f"The best UAR for each split: {TO_PRINT}")
            print(f"Mean uar: {np.mean(best_uar)}")
