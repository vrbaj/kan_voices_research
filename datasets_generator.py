"""
Script to generate all evaluated datasets, with various features combinations
"""
import csv
from pathlib import Path
import itertools
from multiprocessing import freeze_support
import pickle
import json
import random

from tqdm.contrib.concurrent import process_map

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# possible feature combinations
diff_pitch = [True]
stdev_f0 = [True]
spectral_centroid = [True]
spectral_contrast = [True]
spectral_flatness = [True]
spectral_rolloff = [True]
zcr = [True]
mfccs = [13]
var_mfccs = [False]
formants = [False]
lfccs = [True]
skewness = [True]
shannon_entropy = [True]
sexes = [0, 1]

def dataset_config_to_json(experiment_config, file_path: Path):
    """Dump a dataset config to json."""
    with file_path.open("w") as f:
        json.dump(experiment_config, f)


def dump_to_pickle(data, file_path):
    """Dump whatever to pickle."""
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def compose_dataset(dataset_params: dict) -> None:
    """
    Function that compose a dataset based on features specified in dataset_params dictionary.
    The dataset is dumped to unique folder. Dataclass was used back in the time.. Due to
    limited audience, I went back to the dict usage.
    :param dataset_params: dictionary that specify the features to be used
    :return: None. Dataset is dumped to a file
    """
    X = []
    y = []
    data_to_dump = {"data": X, "labels": y}
    folder_to_dump = dataset_params.pop("folder_to_dump")

    with open("features.csv", newline="", encoding="utf-8") as csv_file:
        dataset = csv.DictReader(csv_file, dialect="unix")
        patient: dict
        for patient in dataset:
            patient_features = []
            nan_in_data = False
            if int(patient["sex"]) == dataset_params["sex"]:
                y.append(int(patient["pathology"]))
                patient_features.append(int(patient["age"]))
                if dataset_params["diff_pitch"]:
                    if patient["diff_pitch"] == "nan":
                        patient_features.append(0.0)
                        nan_in_data = True
                    else:
                        patient_features.append(float(patient["diff_pitch"]))

                if patient["mean_f0"] == "nan":
                    patient_features.append(0.0)
                    nan_in_data = True
                else:
                    patient_features.append(float(patient["mean_f0"]))

                if dataset_params["stdev_f0"]:
                    if patient["stdev_f0"] == "nan":
                        patient_features.append(0.0)
                        nan_in_data = True
                    else:
                        patient_features.append(float(patient["stdev_f0"]))

                patient_features.append(float(patient["hnr"]))

                if patient["jitter"] == "nan":
                    patient_features.append(0.0)
                    nan_in_data = True
                else:
                    patient_features.append(float(patient["jitter"]))

                if patient["shimmer"] == "nan":
                    patient_features.append(0.0)
                    nan_in_data = True
                else:
                    patient_features.append(float(patient["shimmer"]))

                all_mfcc = json.loads(patient["mfcc"])
                patient_features += all_mfcc[:dataset_params["mfcc"]]

                all_delta_mfcc = json.loads(patient["delta_mfcc"])
                patient_features += all_delta_mfcc[:dataset_params["mfcc"]]

                all_delta2_mfcc = json.loads(patient["delta2_mfcc"])
                patient_features += all_delta2_mfcc[:dataset_params["mfcc"]]

                if dataset_params["var_mfcc"]:
                    all_var_mfcc = json.loads(patient["var_mfcc"])
                    patient_features += all_var_mfcc[:dataset_params["mfcc"]]

                    all_var_delta_mfcc = json.loads(patient["var_delta_mfcc"])
                    patient_features += all_var_delta_mfcc[:dataset_params["mfcc"]]

                    all_var_delta2_mfcc = json.loads(patient["var_delta2_mfcc"])
                    patient_features += all_var_delta2_mfcc[:dataset_params["mfcc"]]

                if dataset_params["spectral_centroid"]:
                    patient_features.append(float(patient["spectral_centroid"]))

                if dataset_params["spectral_contrast"]:
                    all_contrasts = json.loads(patient["spectral_contrast"])
                    patient_features += all_contrasts

                if dataset_params["spectral_flatness"]:
                    patient_features.append(float(patient["spectral_flatness"]))

                if dataset_params["spectral_rolloff"]:
                    patient_features.append(float(patient["spectral_rolloff"]))

                if dataset_params["zero_crossing_rate"]:
                    patient_features.append(float(patient["zero_crossing_rate"]))

                if dataset_params["formants"]:
                    all_formants = json.loads(patient["formants"])
                    patient_features += all_formants

                if dataset_params["shannon_entropy"]:
                    patient_features.append(float(patient["shannon_entropy"]))

                if dataset_params["lfcc"]:
                    all_lfcc = json.loads(patient["lfcc"])
                    patient_features += all_lfcc

                if dataset_params["skewness"]:
                    patient_features.append(float(patient["skewness"]))

                if nan_in_data:
                    patient_features.append(1)
                else:
                    patient_features.append(0)

                X.append(patient_features)

    subdir_name = "women"
    if dataset_params["sex"] == 0:
        subdir_name = "men"

    dataset_path = Path(".").joinpath("training_data", subdir_name, folder_to_dump)
    dataset_path.mkdir(parents=True)
    dataset_file = dataset_path.joinpath("dataset.pk")
    dump_to_pickle(data_to_dump, dataset_file)
    dataset_config_to_json(dataset_params, dataset_path.joinpath("config.json"))
# pylint: enable=too-many-locals,too-many-branches,too-many-statements

# pylint: disable=too-many-locals
def main(max_workers: int = 8) -> None:
    """
    Main function that creates all possible datasets based on the features specified in the script.
    :param max_workers: number of workers to be used in the process_map function
    :return: None
    """
    for sex_of_interest in sexes:
        configurations = []

        for folder_to_dump, settings in enumerate(itertools.product(diff_pitch, stdev_f0, spectral_centroid,
                                                         spectral_contrast, spectral_flatness,
                                                         spectral_rolloff, zcr, mfccs, var_mfccs,
                                                         formants, lfccs, skewness, shannon_entropy)):
            diff, stdev, centroid, contrast, flatness, rolloff, zrc_param, mfcc, var_mfcc, \
            formant, lfcc, skew, shannon  = settings
            # dataset configuration
            dataset_config = {"sex": sex_of_interest,
                              "diff_pitch": diff,
                              "stdev_f0": stdev,
                              "mfcc": mfcc,
                              "var_mfcc": var_mfcc,
                              "spectral_centroid": centroid,
                              "spectral_contrast": contrast,
                              "spectral_flatness": flatness,
                              "spectral_rolloff": rolloff,
                              "zero_crossing_rate": zrc_param,
                              "formants": formant,
                              "lfcc": lfcc,
                              "skewness": skew,
                              "shannon_entropy": shannon,
                              "folder_to_dump": str(folder_to_dump),}
            configurations.append(dataset_config)

        print(f"Totally {len(configurations)} datasets will be created for sex {sex_of_interest}")

        # use max_workes possible, according to your CPU, it speeds up set creation significantly
        # i.e. my Ryzen with 12 cores (max_workers=24) creates all datasets in 2 minutes
        # i.e. Intel Core i5 7th gen. with 4 workers took hours to finish.
        process_map(compose_dataset, configurations, max_workers=max_workers, chunksize=4)

# pylint: enable=too-many-locals

if __name__ == "__main__":
    freeze_support()
    main()
