# =======================================================
# Author:   Tito Spadini
# Date:     2024-05-21
# Project:  SER - Machine Faults
# File:     mafaulda_audio_extraction.py
# Version:  1.0
# =======================================================


# ==================================================
# -------------------- Imports  --------------------
# ==================================================

import os

import pandas as pd
import numpy as np
import soundfile as sf

from tqdm import tqdm


# ====================================================
# -------------------- Functions  --------------------
# ====================================================

# ----- Convert CSV to WAV -----
def convert_csv_to_wav(src_dir, dst_dir):
    """Converts all CSV files in the source directory to WAV files in the destination directory.

    Args:
        src_dir (str): The source directory containing the CSV files.
        dst_dir (str): The destination directory to save the converted WAV files.

    Returns:
        None
    """
    # list to store all CSV file paths
    csv_files = []

    # Walk through all directories and files in the source
    for root, dirs, files in os.walk(src_dir):

        # sort directories and files in alphabetical order
        dirs.sort()
        files.sort()

        for file in files:
            if file.endswith('.csv'):
                # Construir o caminho completo do arquivo CSV e adicionar à lista
                csv_files.append(os.path.join(root, file))

    # Configuring the progress bar
    with tqdm(total=len(csv_files), desc="Converting MAFAULDA's CSV to WAV files ...") as pbar:
        for csv_path in csv_files:
            # read the CSV file
            df = pd.read_csv(csv_path)

            # extract the audio signal from the CSV file
            audio_signal = df.iloc[:, -1].values

            # sample frequency
            sample_rate = 50000

            # create the full path to the destination WAV file
            relative_path = os.path.relpath(os.path.dirname(csv_path), src_dir)
            wav_dir = os.path.join(dst_dir, relative_path)
            os.makedirs(wav_dir, exist_ok=True)
            wav_path = os.path.join(wav_dir, os.path.basename(csv_path).replace('.csv', '.wav'))

            # write the audio signal to the WAV file
            sf.write(wav_path, audio_signal, sample_rate, subtype='PCM_24')

            # update the progress bar
            pbar.update(1)
            pbar.set_postfix(file=os.path.basename(csv_path))


# ====================================================
# -------------------- Execution  --------------------
# ====================================================

# source directory (where the CSV files are located)
src_dir = '/home/tito/datasets/MAFAULDA'

# destination directory (where the WAV files will be saved)
dst_dir = '/home/tito/ser-machine-faults/data/XXX'

# call the function to convert CSV files to WAV files
convert_csv_to_wav(src_dir, dst_dir)
