import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import argparse

# List of SNR scenarios and machine parts
snr_levels = ['-6_dB', '0_dB', '6_dB']
machine_parts = ['fan', 'pump', 'slider', 'valve']
ids = ['id_00', 'id_02', 'id_04', 'id_06']
states = ['abnormal', 'normal']

# Function to count files in the dataset
def count_files_in_dataset(source_dataset_path):
    file_count = 0
    for snr in snr_levels:
        for part in machine_parts:
            part_path = os.path.join(source_dataset_path, snr, part)
            for id_folder in ids:
                for state in states:
                    state_path = os.path.join(part_path, id_folder, state)
                    file_count += len(os.listdir(state_path))
    return file_count

# Function to convert to mono, normalize and save the audio files
def convert_and_normalize_audio(source_dataset_path, destination_dataset_path):
    total_files = count_files_in_dataset(source_dataset_path)
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for snr in snr_levels:
            for part in machine_parts:
                part_path = os.path.join(source_dataset_path, snr, part)
                new_part_path = os.path.join(destination_dataset_path, snr, part)

                for id_folder in ids:
                    for state in states:
                        state_path = os.path.join(part_path, id_folder, state)
                        new_state_path = os.path.join(new_part_path, id_folder, state)

                        # Create directories if they do not exist
                        os.makedirs(new_state_path, exist_ok=True)

                        # Process and save files
                        for file_name in os.listdir(state_path):
                            full_file_name = os.path.join(state_path, file_name)
                            if os.path.isfile(full_file_name):
                                # Load the audio file
                                audio, sr = sf.read(full_file_name, always_2d=True)

                                # Convert to mono by summing all channels
                                mono_audio = np.sum(audio, axis=1)

                                # Normalize audio to -6 dBFS
                                rms = np.sqrt(np.mean(mono_audio**2))
                                scalar = 10**(-6/20) / rms
                                normalized_audio = mono_audio * scalar

                                # Save the new audio file
                                new_file_path = os.path.join(new_state_path, file_name)
                                sf.write(new_file_path, normalized_audio, sr)

                                pbar.update(1)

# Main function to parse arguments and execute the conversion process
def main():
    parser = argparse.ArgumentParser(description="Convert and normalize MIMII dataset audio to mono and -6 dBFS.")
    parser.add_argument('--input', required=True, help="Path to the source dataset")
    parser.add_argument('--output', required=True, help="Path to the destination dataset")
    args = parser.parse_args()

    source_dataset_path = args.input
    destination_dataset_path = args.output

    convert_and_normalize_audio(source_dataset_path, destination_dataset_path)

if __name__ == "__main__":
    main()
