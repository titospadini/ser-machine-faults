# =======================================================
# Author:   Tito Spadini
# Date:     2024-05-21
# Project:  SER - Machine Faults
# File:     mafaulda_audio_conversion.py
# Version:  1.0
# =======================================================


# ==================================================
# -------------------- Imports  --------------------
# ==================================================

import os

import argparse
import numpy as np
import soundfile as sf

from scipy.signal import resample
from tqdm import tqdm


# =========================================================
# -------------------- Side Functions  --------------------
# =========================================================

# ----- Normalization -----
def normalize_audio(audio, target_dbfs=-6):
    """Normalizes the audio to the target dBFS level.

    Args:
        audio (np.ndarray): The audio signal.
        target_dbfs (int): The target dBFS level. Defaults to -6.

    Returns:
        np.ndarray: The normalized audio signal.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_dbfs / 20)
    return audio * (target_rms / rms)


# ----- Silence Removal -----
def remove_silence(audio, sample_rate=24000, silence_threshold_dbfs=-48):
    """Removes silence from the audio signal.

    Args:
        audio (np.ndarray): The audio signal.
        sample_rate (int, optional): The sampling rate. Defaults to 24000.
        silence_threshold_dbfs (int, optional): The silence threshold in dBFS. Defaults to -48.

    Returns:
        np.ndarray: The audio signal without silence.
    """
    threshold = 10 ** (silence_threshold_dbfs / 20)
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]
    if non_silent_indices.size == 0:
        return audio  # No silence detected
    start_index = non_silent_indices[0]
    end_index = non_silent_indices[-1] + 1
    return audio[start_index:end_index]


# ----- Processing WAV Files -----
def process_wav_files(src_dir, dst_dir, frequency=24000, bits=16, norm_dbfs=-6, silence_dbfs=-48):
    """Processes all WAV files in the source directory and saves them in the destination directory.

    Args:
        src_dir (str): The source directory containing the WAV files.
        dst_dir (str): The destination directory to save the processed WAV files.
        frequency (int, optional): The target sampling rate. Defaults to 24000 Hz.
        bits (int, optional): The target bit depth. Defaults to 16.
        norm_dbfs (int, optional): The target dBFS for normalization. Defaults to -6.
        silence_dbfs (int, optional): The threshold dBFS for silence removal. Defaults to -48.

    Returns:
        None
    """
    # Collect all WAV files
    wav_files = []
    for root, dirs, files in os.walk(src_dir):

        # sort directories and files in alphabetical order
        dirs.sort()
        files.sort()

        # Add all WAV files to the list
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    # Progress bar setup
    with tqdm(total=len(wav_files), desc="Processing WAV files") as pbar:
        for wav_path in wav_files:
            audio, sr = sf.read(wav_path)

            # Resample audio if necessary
            if sr != frequency:
                num_samples = int(len(audio) * frequency / sr)
                audio = resample(audio, num_samples)

            # Normalize audio if requested
            if norm_dbfs is not None:
                audio = normalize_audio(audio, norm_dbfs)

            # Remove silence if requested
            if silence_dbfs is not None:
                audio = remove_silence(audio, frequency, silence_dbfs)

            # Construct destination path
            relative_path = os.path.relpath(os.path.dirname(wav_path), src_dir)
            wav_dir = os.path.join(dst_dir, relative_path)
            os.makedirs(wav_dir, exist_ok=True)
            wav_dst_path = os.path.join(wav_dir, os.path.basename(wav_path))

            # Save processed audio
            sf.write(wav_dst_path, audio, frequency, subtype=f'PCM_{bits}')
            pbar.update(1)
            pbar.set_postfix(file=os.path.basename(wav_path))


# ========================================================
# -------------------- Main Function  --------------------
# ========================================================
def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process WAV files with resampling, normalization, and silence removal.")

    # Add arguments
    parser.add_argument('src_dir', type=str, help='Source directory containing the original WAV files')
    parser.add_argument('dst_dir', type=str, help='Destination directory to save processed WAV files')
    parser.add_argument('-f', '--frequency', type=int, default=24000, help='Target sampling rate (default: 24000 Hz)')
    parser.add_argument('-b', '--bits', type=int, default=16, choices=[8, 16, 24, 32], help='Target bit depth (default: 16-bit)')
    parser.add_argument('-n', '--norm_dbfs', type=float, default=-6, nargs='?', const=None, help='Target dBFS for normalization (default: -6 dBFS)')
    parser.add_argument('-s', '--silence_dbfs', type=float, default=-48, nargs='?', const=None, help='Threshold dBFS for silence removal (default: -48 dBFS)')

    # Parse arguments
    args = parser.parse_args()

    # Process WAV files
    process_wav_files(args.src_dir, args.dst_dir, args.frequency, args.bits, args.norm_dbfs, args.silence_dbfs)


# =========================================================
# -------------------- Main Execution  --------------------
# =========================================================
if __name__ == "__main__":
    main()
