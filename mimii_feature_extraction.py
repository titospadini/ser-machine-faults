import os
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging
from features import feature_extractor

class Config:
    FEATURES_LST = [
        "mean", "median", "variance", "standard_deviation", "skewness", "kurtosis",
        "energy", "power", "min", "max", "peak_to_peak", "root_mean_square", "zero_crossing_rate",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
        "spectral_contrast", "mfcc", "delta_mfcc", "delta_delta_mfcc"
    ]
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

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

def process_directory(dir_path, snr, part, machine_id, status, config, base_dir):
    """
    Processes all audio files in a directory, extracting features and labeling them.

    Parameters:
        dir_path (str): Path to the directory containing audio files.
        snr (str): SNR scenario of the audio files.
        part (str): Machine part of the audio files.
        machine_id (str): ID of the machine.
        status (str): Status (normal/abnormal) of the audio files.
        config (Config): Configuration object with parameters.
        base_dir (str): Base directory for relative path calculation.

    Returns:
        DataFrame: A DataFrame containing extracted features and metadata.
    """
    data = []
    audio_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]

    relative_dir_path = os.path.relpath(dir_path, base_dir)

    for file_counter, file in tqdm(enumerate(audio_files), total=len(audio_files), desc=f"Processing {relative_dir_path}"):
        try:
            file_path = os.path.join(dir_path, file)
            features, _ = read_and_process_audio_file(file_path, config)

            # Create DataFrame for features
            feature_df = pd.DataFrame(features.T)

            # Add metadata columns
            feature_df['filename'] = os.path.relpath(file_path, base_dir)
            feature_df['snr'] = snr
            feature_df['part'] = part
            feature_df['machine_id'] = machine_id
            feature_df['status'] = status

            data.append(feature_df)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def main(config, input_path, output_csv_path):
    """
    Main function to process all directories and compile features into a CSV file.

    Parameters:
        config (Config): Configuration object with parameters.
        input_path (str): Path to the input dataset.
        output_csv_path (str): Path to the output CSV file.
    """
    base_dir = input_path

    data = []

    for snr in ['-6_dB', '0_dB', '6_dB']:
        for part in ['fan', 'pump', 'slider', 'valve']:
            for machine_id in ['id_00', 'id_02', 'id_04', 'id_06']:
                for status in ['abnormal', 'normal']:
                    dir_path = os.path.join(base_dir, snr, part, machine_id, status)
                    if os.path.exists(dir_path):
                        df = process_directory(dir_path, snr, part, machine_id, status, config, base_dir)
                        if not df.empty:
                            data.append(df)

    if data:
        final_df = pd.concat(data, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from MIMII dataset and save to CSV.")
    parser.add_argument('--input', required=True, help="Path to the input dataset")
    parser.add_argument('--output', required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    config = Config()
    main(config, args.input, args.output)
