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


def resample(signal, input_sampling_frequency, output_sampling_frequency):
    """
    Resamples a signal from input_fs to output_fs without using LibROSA.

    Args:
        signal (np.ndarray): The input signal.

        input_fs (float): The input sampling frequency.

        output_fs (float): The output sampling frequency.

    Returns:
        np.ndarray: The resampled signal.
    """
    # Calculate the resampling factor
    resampling_factor = output_sampling_frequency / input_sampling_frequency

    # Calculate the number of samples in the input signal
    num_samples_in = len(signal)

    # Calculate the number of samples in the output signal
    num_samples_out = int(np.ceil(num_samples_in * resampling_factor))

    # Create an array of indices for interpolation
    indices = (np.arange(num_samples_out) / resampling_factor)

    # Interpolate the signal using linear interpolation
    resampled_signal = np.interp(indices, np.arange(num_samples_in), signal)

    return resampled_signal