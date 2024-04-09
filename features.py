import numpy as np

from scipy import stats
from librosa.util import frame

def mean(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the mean along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The mean along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.mean(framed_signal, axis=-2, keepdims=True)


def median(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the median along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The median along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.median(framed_signal, axis=-2, keepdims=True)


def variance(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the variance along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The variance along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.var(framed_signal, axis=-2, keepdims=True)


def standard_deviation(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the standard deviation along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The standard deviation along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.std(framed_signal, axis=-2, keepdims=True)


def skewness(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the skewness along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The skewness along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return stats.skew(framed_signal, axis=-2, keepdims=True)


def kurtosis(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the kurtosis along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The kurtosis along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return stats.kurtosis(framed_signal, axis=-2, keepdims=True)


def energy(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the energy along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The energy along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.sum(framed_signal**2, axis=-2, keepdims=True)


def power(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the power along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The power along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.sum(framed_signal**2, axis=-2, keepdims=True) / framed_signal.shape[-2]


def feature_extractor(signal, features, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Extract features from a signal.

    Args:
        signal (np.ndarray): The signal.
        features (list): The list of features to extract.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Raises:
        ValueError: Invalid feature.

    Returns:
        np.ndarray: The extracted features.
    """
    feature_lst = []

    for feature in features:
        if feature == "mean":
            feature_lst.append(mean(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "median":
            feature_lst.append(median(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "variance":
            feature_lst.append(variance(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "standard_deviation":
            feature_lst.append(standard_deviation(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "skewness":
            feature_lst.append(skewness(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "kurtosis":
            feature_lst.append(kurtosis(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "energy":
            feature_lst.append(energy(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        elif feature == "power":
            feature_lst.append(power(signal, frame_length=frame_length, hop_length=hop_length, center=True).flatten())
        else:
            raise ValueError(f"Invalid feature: {feature}.")

    return np.array(feature_lst)