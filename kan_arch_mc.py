"""
KAN arch search script.
"""
import pickle
from pathlib import Path
import torch
import numpy as np
from kan import KAN
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.base import BaseSampler
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

N_SEED = 42
np.random.seed(N_SEED)


class CustomSMOTE(BaseSampler):
    """Class that implements KMeansSMOTE oversampling. Due to initialization of KMeans
    there are 10 tries to resample the dataset. Then standard SMOTE is applied.
    """
    _sampling_type = "over-sampling"

    def __init__(self, kmeans_args=None, smote_args=None):
        super().__init__()
        self.kmeans_args = kmeans_args if kmeans_args is not None else {}
        self.smote_args = smote_args if smote_args is not None else {}
        self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
        self.smote = SMOTE(**self.smote_args)

    def _fit_resample(self, X, y):
        resample_try = 0
        while resample_try < 10:
            try:
                X_res, y_res = self.kmeans_smote.fit_resample(X, y)
                return X_res, y_res
            except Exception:
                # dont care about exception, KmeansSMOTE failed
                self.kmeans_smote = KMeansSMOTE(random_state=resample_try)
                resample_try += 1
        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res


def train_acc():
    """
    Train accuracy. That is how the PyKAN needs the metric functions.
    """
    return torch.mean((torch.argmax(model(dataset["train_input"]),
                                    dim=1) == dataset["train_label"]).float())


def test_acc():
    """
    Test accuracy. That is how the PyKAN needs the metric functions.
    """
    return torch.mean((torch.argmax(model(dataset["test_input"]),
                                    dim=1) == dataset["test_label"]).float())


def test_recall():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    # Calculate recall
    recall = tp / (tp + fn)
    return recall


def test_specificity():
    """
    Specificity for the test. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    # Calculate specificity
    specificity = tn / (tn + fp)
    return specificity


# since PyKAN 0.1.2 it is necessary to magically set torch default type to float64
# to avoid issues with matrix inversion during training with the LBFGS optimizer
torch.set_default_dtype(torch.float64)
# path to training datasets
datasets = Path("", "kan_training_dataset_men")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
print(f"The {DEVICE} will be used for the computation..")
for dataset in datasets.iterdir():
    print(f"evaluating dataset {dataset}")
    # load dataset
    with open(dataset.joinpath("dataset.pk"), "rb") as f:
        dataset_file = pickle.load(f)
    X = np.array(dataset_file["data"])
    y = np.array(dataset_file["labels"])
    # path where to store results
    results_path = Path(".", "results_mc_kan_search_no_lambda_no_clsweight", dataset)
    # get the number of features
    input_size = X.shape[1]
    # define KAN architecture
    kan_archs = [[input_size, input_size, input_size, 2],
                 [input_size, input_size * 2, 2],
                 [input_size, int(input_size / 2), int(input_size / 4), 2],
                 [input_size, input_size * 2, int(input_size / 4), 2],
                 [input_size, input_size + int(0.5 * input_size), 2],
                 [input_size, input_size - int(0.5 * input_size), 2],
                 [input_size, input_size + int(0.5 * input_size), int(0.5 * input_size), 2]]
    # iterate over KAN architectures and train for each dataset
    for arch in kan_archs:
        torch.manual_seed(0)

        # create results directory for each dataset and evaluated architecture
        result_dir = results_path.joinpath(str(arch).replace(",", "_").replace(" ", ""))
        result_dir.mkdir(parents=True, exist_ok=True)
        # Monte Carlo cross-validation = split train/test 10 times
        print(f"evaluating {str(arch)}")
        for idx in range(10):
            # use random_state for reproducible split (KmeansSMOTE is not reproducible anyway...)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=idx)
            # KMeansSMOTE resampling. if fails 10x SMOTE resampling
            X_resampled, y_resampled = CustomSMOTE(kmeans_args={"random_state": N_SEED}).fit_resample(X_train, y_train)
            # KAN dataset format, load it to device
            dataset = {"train_input": torch.from_numpy(X_resampled).to(DEVICE),
                       "train_label": torch.from_numpy(y_resampled).type(
                           torch.LongTensor).to(DEVICE),
                       "test_input": torch.from_numpy(X_test).to(DEVICE),
                       "test_label": torch.from_numpy(y_test).type(torch.LongTensor).to(DEVICE)}
            # feature dimension sanity check
            # print(dataset["train_input"].dtype)
            # create KAN model
            model = KAN(width=arch, grid=5, k=3, device=DEVICE, auto_save=False)
            # load model to device
            model.to(DEVICE)
            # although the dataset is balanced, KAN tends to overfit to unhealthy...
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            # generally it should be hyperparameter to optimize
            class_weights = torch.tensor(class_weights, dtype=torch.float64).to(DEVICE)
            # train model
            results = model.fit(dataset, opt="LBFGS",
                                  steps=12, batch=-1,
                                  metrics=(train_acc, test_acc, test_specificity, test_recall),
                                  loss_fn=torch.nn.CrossEntropyLoss(),
                                  device=DEVICE)
            # infotainment
            print(f"final test acc: {results['test_acc'][-1]}"
                  f" mean test acc: {np.mean(results['test_acc'])}")
            # dump results
            with open(result_dir.joinpath(f'kan_res_{idx}.pickle'), "wb") as output_file:
                pickle.dump(results, output_file)
