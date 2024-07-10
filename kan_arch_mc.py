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

from utilities import CustomSMOTE


def train_acc():
    return torch.mean((torch.argmax(model(dataset["train_input"]), dim=1) == dataset["train_label"]).float())


def test_acc():
    return torch.mean((torch.argmax(model(dataset["test_input"]), dim=1) == dataset["test_label"]).float())


def test_recall():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    # Calculate recall
    recall = tp / (tp + fn)
    return recall


def test_specificity():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    # Calculate specificity
    specificity = tn / (tn + fp)
    return specificity


datasets = Path("", "kan_training_dataset_men")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The {device} will be used for the computation..")
for dataset in datasets.iterdir():
    # load dataset
    with open(dataset.joinpath("dataset.pk"), "rb") as f:
        dataset_file = pickle.load(f)
    X = np.array(dataset_file["data"])
    y = np.array(dataset_file["labels"])
    # path where to store results
    results_path = Path(".", "results_mc_kan_search", dataset)
    # get the number of features
    input_size = X.shape[1]
    # define KAN architecture
    kan_archs = [[input_size, input_size, input_size, 2],
                 [input_size, input_size * 2, 2],
                 [input_size, int(input_size / 2), int(input_size / 4), 2],
                 [input_size, input_size * 2, int(input_size / 4), 2]]
    # iterate over KAN architectures and train for each dataset
    for arch in kan_archs:
        # create results directory for each dataset and evaluated architecture
        result_dir = results_path.joinpath(str(arch).replace(",", "_").replace(" ", ""))
        result_dir.mkdir(parents=True, exist_ok=True)
        # Monte Carlo cross-validation = split train/test 10 times
        for idx in range(10):
            # use random_state for reproducible split (KmeansSMOTE is not reproducible anyway...)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=idx)

            X_resampled, y_resampled = CustomSMOTE().fit_resample(X_train, y_train)
            # KAN dataset format, load it to device
            dataset = {"train_input": torch.from_numpy(X_resampled.astype(np.float32)).to(device),
                       "train_label": torch.from_numpy(y_resampled).type(torch.LongTensor).to(device),
                       "test_input": torch.from_numpy(X_test.astype(np.float32)).to(device),
                       "test_label": torch.from_numpy(y_test).type(torch.LongTensor).to(device)}
            # create KAN model
            model = KAN(width=arch, grid=5, k=3, device=device, auto_save=False)
            # load model to device
            model.to(device)
            # although the dataset is balanced, KAN tends to overfit to unhealthy...
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            # generally it should be hyperparameter to optimize
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            # try:
            # train model
            results = model.train(dataset, opt="LBFGS",
                                  steps=12, batch=-1,
                                  metrics=(train_acc, test_acc, test_specificity, test_recall),
                                  loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights), device=device, lamb=0.001)
            print(f"final train acc {results['train_acc']}final test acc: {results['test_acc'][-1]} mean test acc: {np.mean(results['test_acc'])}")
            # dump results
            with open(result_dir.joinpath(f'kan_res_{idx}.pickle'), "wb") as output_file:
                pickle.dump(results, output_file)
            # except Exception as ex:
            #     # this should happen only if something is really FUCKED UP
            #     print(f"FUCKED {idx} - {ex}")
