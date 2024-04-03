import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn import metrics

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


def remove_silence(signal, threshold_dbfs=-40):
    """Removes silence from an audio signal.

    Args:
        signal (np.ndarray): The audio signal.
        threshold_dbfs (int, optional): The threshold in dBFS. Defaults to -40.

    Returns:
        np.ndarray: The audio signal without silence.
    """
    dbfs = 20 * np.log10(np.maximum(np.abs(signal), 1e-10))

    return signal[dbfs > threshold_dbfs]


def segment_signal(signal, sampling_frequency, segment_duration=200, overlap_duration=10):
    """Segments an audio signal into segments of a given duration.

    Args:
        signal (np.ndarray): The audio signal.
        sampling_frequency (int): The sampling frequency of the audio signal.
        segment_duration (int, optional): The duration of the segments. Defaults to 200 ms.
        overlap_duration (int, optional): The duration of the overlap. Defaults to 10 ms.

    Returns:
        np.ndarray: An array of the audio segments.
    """
    segment_duration_samples = int(segment_duration * sampling_frequency / 1000)
    overlap_duration_samples = int(overlap_duration * sampling_frequency / 1000)

    segments = []

    for i in range(0, len(signal) - segment_duration_samples, segment_duration_samples - overlap_duration_samples):
        segments.append(signal[i : i + segment_duration_samples])

    return np.array(segments)


def audio_to_df(path, mixing_method="sum", normalize_dbfs=-6, new_sampling_frequency=16000, silence_threshold_dbfs=-48, segment_duration=500, overlap_duration=10):
    """Processes an audio file through the pipeline of normalization, resampling, silence removal, and segmentation saves its segments as a Pandas DataFrame.

    Args:
        path (str): path to the audio file.
        mixing_method (str, optional): method to mix the stereo signals to mono. Defaults to "sum".
        normalize_dbfs (int, optional): dBFS level to normalize the signal. Defaults to -6.
        new_sampling_frequency (int, optional): new sampling frequency to resample the signal. Defaults to 16000.
        silence_threshold_dbfs (int, optional): dBFS level to remove silence. Defaults to -48.
        segment_duration (int, optional): duration of the segments. Defaults to 500.
        overlap_duration (int, optional): overlap duration. Defaults to 10.

    Returns:
        pd.DataFrame: The audio segments as a Pandas DataFrame.
    """
    data, sampling_frequency = read_audio(path)

    if data.ndim > 1:
        data = stereo_to_mono(data, method=mixing_method)

    if normalize_dbfs is not None:
        data = normalize(data, dbfs=normalize_dbfs)

    if sampling_frequency != new_sampling_frequency:
        data = resample(data, input_sampling_frequency=sampling_frequency, output_sampling_frequency=new_sampling_frequency)

    if silence_threshold_dbfs is not None:
        data = remove_silence(data, threshold_dbfs=silence_threshold_dbfs)

    segments = segment_signal(data, sampling_frequency=new_sampling_frequency, segment_duration=segment_duration, overlap_duration=overlap_duration)

    df = pd.DataFrame(segments)

    return df


def plot_confusion_matrix(y_true, y_pred, labels, print_colorbar=False, print_numbers=True, print_boxes=False):
    """Plots a confusion matrix.

    Args:
        y_true (np.ndarray): vector of true labels.
        y_pred (np.ndarray): vector of predicted labels.
        labels (list): list of labels.
        print_colorbar (bool, optional): prints the colorbar. Defaults to False.
        print_numbers (bool, optional): prints the numbers. Defaults to True.
        print_boxes (bool, optional): prints boxes behind the numbers. Defaults to False.
    """
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)

    if print_numbers:
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, str(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3') if print_boxes else None)

    if print_colorbar:
        fig.colorbar(cax)

    # Define os ticks antes de definir os rótulos
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()