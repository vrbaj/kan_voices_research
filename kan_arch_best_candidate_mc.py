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

from utilities import CustomSMOTE  # pylint:disable=import-error


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
dataset = Path("", "kan_training_dataset_men", "68035e02-e891-4b0c-bc3a-b5023f3534e8")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device("cpu")
print(f"The {DEVICE} will be used for the computation..")

# load dataset
with open(dataset.joinpath("dataset.pk"), "rb") as f:
    dataset_file = pickle.load(f)
X = np.array(dataset_file["data"])
y = np.array(dataset_file["labels"])
# path where to store results
results_path = Path(".", "results_mc_kan_search_best_candidate", dataset)
# get the number of features
input_size = X.shape[1]
# define KAN architecture

model1 = KAN(width=[input_size, input_size * 2, 2], grid=7, k=3, device=DEVICE, auto_save=False)
model2 = KAN(width=[input_size, input_size * 2, 2], grid=7, k=5, device=DEVICE, auto_save=False)
model3 = KAN(width=[input_size, input_size * 2, 2], grid=9, k=3, device=DEVICE, auto_save=False)
model4 = KAN(width=[input_size, input_size * 2, 2], grid=9, k=5, device=DEVICE, auto_save=False)
model5 = KAN(width=[input_size, int(input_size * 2 + 10), 2], grid=5, k=3, device=DEVICE, auto_save=False)
model6 = KAN(width=[input_size, int(input_size / 2), int(10 + input_size / 4), 2], grid=5, k=3, device=DEVICE, auto_save=False)
model7 = KAN(width=[input_size, int(input_size * 2 + 10), 2], grid=5, k=3, device=DEVICE, auto_save=False)
models = [model1, model2, model3, model4, model5, model6, model7]
# iterate over KAN architectures and train for each dataset
for idx_m, model in enumerate(models):
    # create results directory for each dataset and evaluated architecture
    result_dir = results_path.joinpath(str(idx_m))
    result_dir.mkdir(parents=True, exist_ok=True)
    # Monte Carlo cross-validation = split train/test 10 times
    for idx in range(10):
        # use random_state for reproducible split (KmeansSMOTE is not reproducible anyway...)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=idx)
        # KMeansSMOTE resampling. if fails 10x SMOTE resampling
        X_resampled, y_resampled = CustomSMOTE().fit_resample(X_train, y_train)
        # KAN dataset format, load it to device
        dataset = {"train_input": torch.from_numpy(X_resampled).to(DEVICE),
                   "train_label": torch.from_numpy(y_resampled).type(
                       torch.LongTensor).to(DEVICE),
                   "test_input": torch.from_numpy(X_test).to(DEVICE),
                   "test_label": torch.from_numpy(y_test).type(torch.LongTensor).to(DEVICE)}
        # feature dimension sanity check
        print(dataset["train_input"].dtype)
        # load model to device
        model.to(DEVICE)
        # although the dataset is balanced, KAN tends to overfit to unhealthy...
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        # generally it should be hyperparameter to optimize
        class_weights = torch.tensor(class_weights, dtype=torch.float64).to(DEVICE)
        # train model
        results = model.train(dataset, opt="LBFGS",
                              steps=12, batch=-1,
                              metrics=(train_acc, test_acc, test_specificity, test_recall),
                              loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
                              device=DEVICE, lamb=0.001)
        # infotainment
        print(f"final test acc: {results['test_acc'][-1]}"
              f" mean test acc: {np.mean(results['test_acc'])}")
        # dump results
        with open(result_dir.joinpath(f'kan_res_{idx}.pickle'), "wb") as output_file:
            pickle.dump(results, output_file)
