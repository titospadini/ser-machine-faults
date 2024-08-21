import os

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import logging
import argparse

from features import feature_extractor
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
parser = argparse.ArgumentParser(description='Process audio files and extract features.')
parser.add_argument('-s', '--segment', type=int, default=500, help='Segment duration in ms (default: 500)')
parser.add_argument('-l', '--overlap', type=int, default=20, help='Overlap duration in ms (default: 20)')
parser.add_argument('-f', '--frequency', type=int, default=24000, help='Sampling frequency in Hz (default: 24000)')
parser.add_argument('-m', '--mfcc', type=int, default=40, help='Number of MFCC coefficients (default: 40)')
parser.add_argument('-c', '--contrast', type=int, default=5, help='Number of spectral contrast bands (default: 5)')
parser.add_argument('-i', '--input_path', type=str, default="/home/tito/ser-machine-faults/corpora/MAFAULDA_24khz_16bits", help='Path to input directory (default: /home/tito/ser-machine-faults/corpora/MAFAULDA_24khz_16bits)')
parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to output CSV file (if not provided, it will be generated automatically)')

args = parser.parse_args()

# Configuration and constants
class Config:
    BASE_DIR            = args.input_path
    SEGMENT_DURATION    = args.segment
    OVERLAP_DURATION    = args.overlap
    SAMPLING_FREQUENCY  = args.frequency
    N_MFCC              = args.mfcc
    N_CONTRAST_BANDS    = args.contrast

    # Generate output path dynamically if not provided
    if args.output_path is None:
        OUTPUT_CSV_PATH = f"data/mafaulda_main_classes_{SAMPLING_FREQUENCY // 1000}khz_{SEGMENT_DURATION}ms_{OVERLAP_DURATION}ms_{N_MFCC}-mfcc.csv"
    else:
        OUTPUT_CSV_PATH = args.output_path

    FRAME_LENGTH    = int(SEGMENT_DURATION / 1000 * SAMPLING_FREQUENCY)
    HOP_LENGTH      = int((SEGMENT_DURATION - OVERLAP_DURATION) / 1000 * SAMPLING_FREQUENCY)

    FEATURES_LST = [
        "mean", "median", "variance", "standard_deviation", "skewness", "kurtosis", "entropy",
        "energy", "power", "min", "max", "peak_to_peak", "root_mean_square", "zero_crossing_rate",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
        "spectral_contrast", "mfcc", "delta_mfcc", "delta_delta_mfcc"
    ]

    DIRS = {
        "normal": f"{BASE_DIR}/normal",

        "horizontal_misalignment": {
            "0.5mm": f"{BASE_DIR}/horizontal-misalignment/0.5mm",
            "1.0mm": f"{BASE_DIR}/horizontal-misalignment/1.0mm",
            "1.5mm": f"{BASE_DIR}/horizontal-misalignment/1.5mm",
            "2.0mm": f"{BASE_DIR}/horizontal-misalignment/2.0mm"
        },

        "vertical_misalignment": {
            "0.51mm": f"{BASE_DIR}/vertical-misalignment/0.51mm",
            "0.63mm": f"{BASE_DIR}/vertical-misalignment/0.63mm",
            "1.27mm": f"{BASE_DIR}/vertical-misalignment/1.27mm",
            "1.40mm": f"{BASE_DIR}/vertical-misalignment/1.40mm",
            "1.78mm": f"{BASE_DIR}/vertical-misalignment/1.78mm",
            "1.90mm": f"{BASE_DIR}/vertical-misalignment/1.90mm"
        },

        "imbalance": {
            "6g":   f"{BASE_DIR}/imbalance/6g",
            "10g":  f"{BASE_DIR}/imbalance/10g",
            "15g":  f"{BASE_DIR}/imbalance/15g",
            "20g":  f"{BASE_DIR}/imbalance/20g",
            "25g":  f"{BASE_DIR}/imbalance/25g",
            "30g":  f"{BASE_DIR}/imbalance/30g",
            "35g":  f"{BASE_DIR}/imbalance/35g"
        },

        "overhang": {
            "ball_fault": {
                "0g":   f"{BASE_DIR}/overhang/ball_fault/0g",
                "6g":   f"{BASE_DIR}/overhang/ball_fault/6g",
                "20g":  f"{BASE_DIR}/overhang/ball_fault/20g",
                "35g":  f"{BASE_DIR}/overhang/ball_fault/35g"
            },
            "cage_fault": {
                "0g":   f"{BASE_DIR}/overhang/cage_fault/0g",
                "6g":   f"{BASE_DIR}/overhang/cage_fault/6g",
                "20g":  f"{BASE_DIR}/overhang/cage_fault/20g",
                "35g":  f"{BASE_DIR}/overhang/cage_fault/35g"
            },
            "outer_race": {
                "0g":   f"{BASE_DIR}/overhang/outer_race/0g",
                "6g":   f"{BASE_DIR}/overhang/outer_race/6g",
                "20g":  f"{BASE_DIR}/overhang/outer_race/20g",
                "35g":  f"{BASE_DIR}/overhang/outer_race/35g"
            }
        },

        "underhang": {
            "ball_fault": {
                "0g":   f"{BASE_DIR}/underhang/ball_fault/0g",
                "6g":   f"{BASE_DIR}/underhang/ball_fault/6g",
                "20g":  f"{BASE_DIR}/underhang/ball_fault/20g",
                "35g":  f"{BASE_DIR}/underhang/ball_fault/35g"
            },
            "cage_fault": {
                "0g":   f"{BASE_DIR}/underhang/cage_fault/0g",
                "6g":   f"{BASE_DIR}/underhang/cage_fault/6g",
                "20g":  f"{BASE_DIR}/underhang/cage_fault/20g",
                "35g":  f"{BASE_DIR}/underhang/cage_fault/35g"
            },
            "outer_race": {
                "0g":   f"{BASE_DIR}/underhang/outer_race/0g",
                "6g":   f"{BASE_DIR}/underhang/outer_race/6g",
                "20g":  f"{BASE_DIR}/underhang/outer_race/20g",
                "35g":  f"{BASE_DIR}/underhang/outer_race/35g"
            }
        }
    }

    CLASSES = {
        "normal":                   0,
        "horizontal_misalignment":  1,
        "vertical_misalignment":    2,
        "imbalance":                3,
        "overhang":                 4,
        "underhang":                5
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
    features, n_components_lst = feature_extractor(audio, features=config.FEATURES_LST, n_mfcc=config.N_MFCC, n_contrast_bands=config.N_CONTRAST_BANDS, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH, center=False, pad_mode="constant")

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

    audio_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]

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
            1: "horizontal_misalignment",
            2: "vertical_misalignment",
            3: "imbalance",
            4: "overhang",
            5: "underhang"
            })

        df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    logging.info(f"Segment Duration: {Config.SEGMENT_DURATION} ms")
    logging.info(f"Overlap Duration: {Config.OVERLAP_DURATION} ms")
    logging.info(f"Sampling Frequency: {Config.SAMPLING_FREQUENCY} Hz")
    logging.info(f"Number of MFCC Coefficients: {Config.N_MFCC}")
    logging.info(f"Number of Spectral Contrast Bands: {Config.N_CONTRAST_BANDS}")
    
    config = Config()
    main(config)
