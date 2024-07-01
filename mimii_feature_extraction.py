import os
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging
from features import feature_extractor
import tempfile
import shutil
import gc
from utils import expand_feature_lst

class Config:
    FEATURES_LST = [
        "mean", "median", "variance", "standard_deviation", "skewness", "kurtosis",
        "energy", "power", "min", "max", "peak_to_peak", "root_mean_square", "zero_crossing_rate",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
        "spectral_contrast", "mfcc", "delta_mfcc", "delta_delta_mfcc"
    ]
    FRAME_LENGTH_MS = 500  # Frame length in milliseconds
    OVERLAP_MS = 20        # Overlap in milliseconds
    SAMPLE_RATE = 16000    # Assuming a sample rate of 16kHz

    @property
    def FRAME_LENGTH(self):
        return int(self.FRAME_LENGTH_MS * self.SAMPLE_RATE / 1000)

    @property
    def HOP_LENGTH(self):
        overlap_samples = int(self.OVERLAP_MS * self.SAMPLE_RATE / 1000)
        return self.FRAME_LENGTH - overlap_samples

    BATCH_SIZE = 100  # Number of files to process at once

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

def process_files(file_paths, snr, part, machine_id, status, config, base_dir, temp_dir):
    """
    Processes a list of audio files, extracting features and labeling them.

    Parameters:
        file_paths (list): List of paths to audio files.
        snr (str): SNR scenario of the audio files.
        part (str): Machine part of the audio files.
        machine_id (str): ID of the machine.
        status (str): Status (normal/abnormal) of the audio files.
        config (Config): Configuration object with parameters.
        base_dir (str): Base directory for relative path calculation.
        temp_dir (str): Directory to save temporary files.

    Returns:
        None
    """
    data = []
    for file_counter, file_path in tqdm(enumerate(file_paths), total=len(file_paths), desc=f"Processing batch {part}/{machine_id}/{status}"):
        try:
            features, n_components_lst = read_and_process_audio_file(file_path, config)

            if file_counter == 0 and sum(n_components_lst) != len(config.FEATURES_LST):
                new_features_lst = expand_feature_lst(config.FEATURES_LST, n_components_lst)

            # Create DataFrame for features
            feature_df = pd.DataFrame(features.T, columns=new_features_lst)

            # Add metadata columns
            feature_df['filename'] = os.path.relpath(file_path, base_dir)
            feature_df['snr'] = snr
            feature_df['part'] = part
            feature_df['machine_id'] = machine_id
            feature_df['status'] = status

            data.append(feature_df)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    if data:
        batch_df = pd.concat(data, ignore_index=True)
        temp_file_path = os.path.join(temp_dir, f"{part}_{machine_id}_{status}_{snr}_{os.path.basename(file_paths[0])}.csv")
        batch_df.to_csv(temp_file_path, index=False)

        # Clear memory
        del data
        del batch_df
        gc.collect()

def process_directory(dir_path, snr, part, machine_id, status, config, base_dir, temp_dir):
    """
    Processes all audio files in a directory in batches, extracting features and labeling them.

    Parameters:
        dir_path (str): Path to the directory containing audio files.
        snr (str): SNR scenario of the audio files.
        part (str): Machine part of the audio files.
        machine_id (str): ID of the machine.
        status (str): Status (normal/abnormal) of the audio files.
        config (Config): Configuration object with parameters.
        base_dir (str): Base directory for relative path calculation.
        temp_dir (str): Directory to save temporary files.

    Returns:
        None
    """
    audio_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".wav")]
    relative_dir_path = os.path.relpath(dir_path, base_dir)

    num_batches = (len(audio_files) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    # Process in batches
    for i in tqdm(range(num_batches), desc=f"Processing directory {relative_dir_path}"):
        batch_files = audio_files[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE]
        process_files(batch_files, snr, part, machine_id, status, config, base_dir, temp_dir)

def main(config, input_path, output_csv_path):
    """
    Main function to process all directories and compile features into a CSV file.

    Parameters:
        config (Config): Configuration object with parameters.
        input_path (str): Path to the input dataset.
        output_csv_path (str): Path to the output CSV file.
    """
    base_dir = input_path

    # Create a temporary directory for intermediate CSV files
    temp_dir = tempfile.mkdtemp()

    try:
        snr_dirs = ['-6_dB', '0_dB', '6_dB']
        parts = ['fan', 'pump', 'slider', 'valve']
        machine_ids = ['id_00', 'id_02', 'id_04', 'id_06']
        statuses = ['abnormal', 'normal']

        total_steps = len(snr_dirs) * len(parts) * len(machine_ids) * len(statuses)
        step_counter = 0

        for snr in snr_dirs:
            for part in parts:
                for machine_id in machine_ids:
                    for status in statuses:
                        dir_path = os.path.join(base_dir, snr, part, machine_id, status)
                        if os.path.exists(dir_path):
                            process_directory(dir_path, snr, part, machine_id, status, config, base_dir, temp_dir)
                        step_counter += 1
                        tqdm.write(f"Progress: {step_counter}/{total_steps}")

        # Combine all temporary CSV files into one
        with open(output_csv_path, 'w') as outfile:
            for i, temp_file in enumerate(sorted(os.listdir(temp_dir))):
                temp_file_path = os.path.join(temp_dir, temp_file)
                with open(temp_file_path, 'r') as infile:
                    if i == 0:
                        outfile.write(infile.read())
                    else:
                        next(infile)  # Skip header line for subsequent files
                        outfile.write(infile.read())

    finally:
        # Remove the temporary directory and its contents
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from MIMII dataset and save to CSV.")
    parser.add_argument('--input', required=True, help="Path to the input dataset")
    parser.add_argument('--output', required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    config = Config()
    main(config, args.input, args.output)
