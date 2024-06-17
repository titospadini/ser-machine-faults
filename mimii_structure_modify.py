import os
import shutil
from tqdm import tqdm
import argparse

# List of SNR scenarios and machine parts
snr_levels = ['-6_dB', '0_dB', '6_dB']
machine_parts = ['fan', 'pump', 'slider', 'valve']
ids = ['id_00', 'id_02', 'id_04', 'id_06']
states = ['abnormal', 'normal']

# Function to count files in the dataset
def count_files_in_dataset(original_dataset_path):
    file_count = 0
    for snr in snr_levels:
        for part in machine_parts:
            part_path = os.path.join(original_dataset_path, f'{snr}_{part}', part)
            for id_folder in ids:
                for state in states:
                    state_path = os.path.join(part_path, id_folder, state)
                    file_count += len(os.listdir(state_path))
    return file_count

# Function to copy and reorganize the dataset with progress bar
def copy_and_reorganize_dataset(original_dataset_path, new_dataset_path):
    total_files = count_files_in_dataset(original_dataset_path)
    with tqdm(total=total_files, desc="Copying files") as pbar:
        for snr in snr_levels:
            for part in machine_parts:
                original_part_path = os.path.join(original_dataset_path, f'{snr}_{part}', part)
                new_part_path = os.path.join(new_dataset_path, snr, part)

                for id_folder in ids:
                    for state in states:
                        original_state_path = os.path.join(original_part_path, id_folder, state)
                        new_state_path = os.path.join(new_part_path, id_folder, state)

                        # Create directories if they do not exist
                        os.makedirs(new_state_path, exist_ok=True)

                        # Copy files
                        for file_name in os.listdir(original_state_path):
                            full_file_name = os.path.join(original_state_path, file_name)
                            if os.path.isfile(full_file_name):
                                shutil.copy(full_file_name, new_state_path)
                                pbar.update(1)

# Main function to parse arguments and execute the copying process
def main():
    parser = argparse.ArgumentParser(description="Copy and reorganize MIMII dataset.")
    parser.add_argument('--input', required=True, help="Path to the original dataset")
    parser.add_argument('--output', required=True, help="Path to the new reorganized dataset")
    args = parser.parse_args()

    original_dataset_path = args.input
    new_dataset_path = args.output

    copy_and_reorganize_dataset(original_dataset_path, new_dataset_path)

if __name__ == "__main__":
    main()
