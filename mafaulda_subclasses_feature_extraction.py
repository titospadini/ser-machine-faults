import os
import numpy as np
import pandas as pd
import soundfile as sf
import logging
from tqdm import tqdm

from features import feature_extractor
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# configuration and constants
class Config:
    BASE_DIR        = "/home/tito/ser-machine-faults/data/MAFAULDA_24khz_16bits"
    OUTPUT_CSV_PATH = "new_mafaulda_subclasses_24khz_500ms_20ms_40-mfcc.csv"

    SEGMENT_DURATION    = 500   # 500 ms
    OVERLAP_DURATION    = 20    # 100 ms
    SAMPLING_FREQUENCY  = 24000 # 24 kHz

    FRAME_LENGTH    = int(SEGMENT_DURATION / 1000 * SAMPLING_FREQUENCY)
    HOP_LENGTH      = int((SEGMENT_DURATION - OVERLAP_DURATION) / 1000 * SAMPLING_FREQUENCY)

    FEATURES_LST = [
        "mean", "median", "variance", "standard_deviation", "skewness", "kurtosis",
        "energy", "power", "min", "max", "peak_to_peak", "root_mean_square", "zero_crossing_rate",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
        "spectral_contrast", "mfcc", "delta_mfcc", "delta_delta_mfcc"
    ]

    DIRS = {
        "normal": f"{BASE_DIR}/normal",

        "horizontal_misalignment_05": f"{BASE_DIR}/horizontal-misalignment/0.5mm",
        "horizontal_misalignment_10": f"{BASE_DIR}/horizontal-misalignment/1.0mm",
        "horizontal_misalignment_15": f"{BASE_DIR}/horizontal-misalignment/1.5mm",
        "horizontal_misalignment_20": f"{BASE_DIR}/horizontal-misalignment/2.0mm",

        "vertical_misalignment_051": f"{BASE_DIR}/vertical-misalignment/0.51mm",
        "vertical_misalignment_063": f"{BASE_DIR}/vertical-misalignment/0.63mm",
        "vertical_misalignment_127": f"{BASE_DIR}/vertical-misalignment/1.27mm",
        "vertical_misalignment_140": f"{BASE_DIR}/vertical-misalignment/1.40mm",
        "vertical_misalignment_178": f"{BASE_DIR}/vertical-misalignment/1.78mm",
        "vertical_misalignment_190": f"{BASE_DIR}/vertical-misalignment/1.90mm",

        "imbalance_06":  f"{BASE_DIR}/imbalance/6g",
        "imbalance_10":  f"{BASE_DIR}/imbalance/10g",
        "imbalance_15":  f"{BASE_DIR}/imbalance/15g",
        "imbalance_20":  f"{BASE_DIR}/imbalance/20g",
        "imbalance_25":  f"{BASE_DIR}/imbalance/25g",
        "imbalance_30":  f"{BASE_DIR}/imbalance/30g",
        "imbalance_35":  f"{BASE_DIR}/imbalance/35g",

        "overhang_ball_00": f"{BASE_DIR}/overhang/ball_fault/0g",
        "overhang_ball_06": f"{BASE_DIR}/overhang/ball_fault/6g",
        "overhang_ball_20": f"{BASE_DIR}/overhang/ball_fault/20g",
        "overhang_ball_35": f"{BASE_DIR}/overhang/ball_fault/35g",

        "overhang_cage_00": f"{BASE_DIR}/overhang/cage_fault/0g",
        "overhang_cage_06": f"{BASE_DIR}/overhang/cage_fault/6g",
        "overhang_cage_20": f"{BASE_DIR}/overhang/cage_fault/20g",
        "overhang_cage_35": f"{BASE_DIR}/overhang/cage_fault/35g",

        "overhang_outer_00": f"{BASE_DIR}/overhang/outer_race/0g",
        "overhang_outer_06": f"{BASE_DIR}/overhang/outer_race/6g",
        "overhang_outer_20": f"{BASE_DIR}/overhang/outer_race/20g",
        "overhang_outer_35": f"{BASE_DIR}/overhang/outer_race/35g",

        "underhang_ball_00": f"{BASE_DIR}/underhang/ball_fault/0g",
        "underhang_ball_06": f"{BASE_DIR}/underhang/ball_fault/6g",
        "underhang_ball_20": f"{BASE_DIR}/underhang/ball_fault/20g",
        "underhang_ball_35": f"{BASE_DIR}/underhang/ball_fault/35g",

        "underhang_cage_00": f"{BASE_DIR}/underhang/cage_fault/0g",
        "underhang_cage_06": f"{BASE_DIR}/underhang/cage_fault/6g",
        "underhang_cage_20": f"{BASE_DIR}/underhang/cage_fault/20g",
        "underhang_cage_35": f"{BASE_DIR}/underhang/cage_fault/35g",

        "underhang_outer_00": f"{BASE_DIR}/underhang/outer_race/0g",
        "underhang_outer_06": f"{BASE_DIR}/underhang/outer_race/6g",
        "underhang_outer_20": f"{BASE_DIR}/underhang/outer_race/20g",
        "underhang_outer_35": f"{BASE_DIR}/underhang/outer_race/35g"
    }

    CLASSES = {
        "normal":                       0,
        "horizontal_misalignment_05":   1,
        "horizontal_misalignment_10":   2,
        "horizontal_misalignment_15":   3,
        "horizontal_misalignment_20":   4,
        "vertical_misalignment_051":    5,
        "vertical_misalignment_063":    6,
        "vertical_misalignment_127":    7,
        "vertical_misalignment_140":    8,
        "vertical_misalignment_178":    9,
        "vertical_misalignment_190":    10,
        "imbalance_06":                 11,
        "imbalance_10":                 12,
        "imbalance_15":                 13,
        "imbalance_20":                 14,
        "imbalance_25":                 15,
        "imbalance_30":                 16,
        "imbalance_35":                 17,
        "overhang_ball_00":             18,
        "overhang_ball_06":             19,
        "overhang_ball_20":             20,
        "overhang_ball_35":             21,
        "overhang_cage_00":             22,
        "overhang_cage_06":             23,
        "overhang_cage_20":             24,
        "overhang_cage_35":             25,
        "overhang_outer_00":            26,
        "overhang_outer_06":            27,
        "overhang_outer_20":            28,
        "overhang_outer_35":            29,
        "underhang_ball_00":            30,
        "underhang_ball_06":            31,
        "underhang_ball_20":            32,
        "underhang_ball_35":            33,
        "underhang_cage_00":            34,
        "underhang_cage_06":            35,
        "underhang_cage_20":            36,
        "underhang_cage_35":            37,
        "underhang_outer_00":           38,
        "underhang_outer_06":           39,
        "underhang_outer_20":           40,
        "underhang_outer_35":           41
    }


def read_and_process_audio_file(file_path, config):
    """
    Reads and processes an audio file, extracting features from the audio data.

    Parameters:
        file_path (str): Path to the audio file.
        config (Config): Configuration object with parameters.

    Returns:
        tuple: Extracted features and a list of component counts for each feature.
    """
    # Read audio file using SoundFile
    audio, _ = sf.read(file_path)

    # Feature extraction (with segmentation)
    features, n_components_lst = feature_extractor(audio, features=config.FEATURES_LST, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH, center=False, pad_mode="constant")

    return features, n_components_lst


def process_directory(dir_path, class_idx, config, base_dir):
    """
    Processes all audio files in a directory, extracting features and labeling them.

    Parameters:
        dir_path (str): Path to the directory containing audio files.
        class_idx (int): Class index for the audio files in this directory.
        config (Config): Configuration object with parameters.
        base_dir (str): Base directory for relative path calculation.

    Returns:
        tuple: Feature matrix X, class vector y, filenames vector, and potentially expanded feature list.
    """
    X = []
    y = []
    filenames = []
    new_features_lst = config.FEATURES_LST.copy()

    audio_files = [file for file in os.listdir(dir_path) if file.endswith(".wav")]

    relative_dir_path = os.path.relpath(dir_path, base_dir)

    for file_counter, file in tqdm(enumerate(audio_files), total=len(audio_files), desc=f"Processing {relative_dir_path}"):
        try:
            file_path = os.path.join(dir_path, file)
            features, n_components_lst = read_and_process_audio_file(file_path, config)

            if file_counter == 0 and sum(n_components_lst) != len(config.FEATURES_LST):
                new_features_lst = expand_feature_lst(config.FEATURES_LST, n_components_lst)

            X.append(features)
            relative_path = os.path.relpath(file_path, base_dir)
            filenames.extend([relative_path] * features.shape[1])
            y.extend([class_idx] * features.shape[1])

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    if X:
        return np.hstack(X), np.array(y).astype(int), filenames, new_features_lst
    else:
        return np.array([]), np.array([]), [], new_features_lst


def main(config):
    """
    Main function to process all directories and compile features into a CSV file.

    Parameters:
        config (Config): Configuration object with parameters.
    """
    base_dir = config.BASE_DIR
    output_csv_path = config.OUTPUT_CSV_PATH

    X = []
    y = []
    filenames = []
    all_features_lst = None

    def process_all_dirs(dirs, class_idx):
        nonlocal X, y, filenames, all_features_lst
        if isinstance(dirs, dict):
            for dir_name, dir_path in dirs.items():
                process_all_dirs(dir_path, class_idx)
        else:
            features, labels, files, features_lst = process_directory(dirs, class_idx, config, base_dir)
            if features.size > 0:
                X.append(features)
                y.append(labels)
                filenames.extend(files)
                if all_features_lst is None:
                    all_features_lst = features_lst

    for class_name, dir_paths in tqdm(config.DIRS.items(), desc="Processing MAFAULDA"):
        class_idx = config.CLASSES[class_name]
        process_all_dirs(dir_paths, class_idx)

    if X:
        X = np.hstack(X)
        y = np.concatenate(y)

        df = pd.DataFrame(X.T, columns=all_features_lst)
        df["filename"] = filenames
        df["class"] = y
        df["label"] = df["class"].map({
            0: "normal",
            1: "horizontal_misalignment_05",
            2: "horizontal_misalignment_10",
            3: "horizontal_misalignment_15",
            4: "horizontal_misalignment_20",
            5: "vertical_misalignment_051",
            6: "vertical_misalignment_063",
            7: "vertical_misalignment_127",
            8: "vertical_misalignment_140",
            9: "vertical_misalignment_178",
            10: "vertical_misalignment_190",
            11: "imbalance_06",
            12: "imbalance_10",
            13: "imbalance_15",
            14: "imbalance_20",
            15: "imbalance_25",
            16: "imbalance_30",
            17: "imbalance_35",
            18: "overhang_ball_00",
            19: "overhang_ball_06",
            20: "overhang_ball_20",
            21: "overhang_ball_35",
            22: "overhang_cage_00",
            23: "overhang_cage_06",
            24: "overhang_cage_20",
            25: "overhang_cage_35",
            26: "overhang_outer_00",
            27: "overhang_outer_06",
            28: "overhang_outer_20",
            29: "overhang_outer_35",
            30: "underhang_ball_00",
            31: "underhang_ball_06",
            32: "underhang_ball_20",
            33: "underhang_ball_35",
            34: "underhang_cage_00",
            35: "underhang_cage_06",
            36: "underhang_cage_20",
            37: "underhang_cage_35",
            38: "underhang_outer_00",
            39: "underhang_outer_06",
            40: "underhang_outer_20",
            41: "underhang_outer_35"
            })

        df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    config = Config()
    main(config)