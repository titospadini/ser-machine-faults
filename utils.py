import numpy as np
import soundfile as sf

def read_audio(path):
    """Reads an audio file and returns the signal and its sampling frequency.

    Args:
        path (str): The path to the audio file.

    Returns:
        tuple: The audio signal and its sampling frequency.
    """
    data, sampling_frequency = sf.read(path)
    return data, sampling_frequency


def stereo_to_mono(signal, method="sum"):
    """Converts a stereo audio signal to a mono signal.

    Args:
        signal (np.ndarray): The audio signal.
        method (str, optional): The method to convert the signal. Defaults to "sum".

    Returns:
        np.ndarray: The mono audio signal.
    """
    if method == "average":
        signal = np.mean(signal, axis=1)
    elif method == "median":
        signal = np.median(signal, axis=1)
    elif method == "max":
        signal = np.max(signal, axis=1)
    elif method == "min":
        signal = np.min(signal, axis=1)
    elif method == "sum":
        signal = np.sum(signal, axis=1)
    else:
        raise ValueError(f"Invalid method: {method}.")

    return signal


def normalize(signal, dbfs=-6):
    """Normalizes an audio signal to a given dBFS level.

    Args:
        signal (np.ndarray): The audio signal.
        dbfs (int, optional): The target dBFS level. Defaults to -6 dBFS.

    Returns:
        np.ndarray: The normalized audio signal.
    """
    return np.power(10, dbfs / 20) * signal