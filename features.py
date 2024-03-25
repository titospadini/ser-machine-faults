import numpy as np

def mean(signal):
    """Calculates the mean of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The mean of the audio signal.
    """
    return np.mean(signal)


def variance(signal):
    """Calculates the variance of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The variance of the audio signal.
    """
    return np.var(signal)


def standard_deviation(signal):
    """Calculates the standard deviation of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The standard deviation of the audio signal.
    """
    return np.std(signal)


def median(signal):
    """Calculates the median of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The median of the audio signal.
    """
    return np.median(signal)


def skewness(signal):
    """Calculates the skewness of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The skewness of the audio signal.
    """
    return (1 / len(signal)) * np.sum(((signal - mean(signal)) / standard_deviation(signal)) ** 3)


def kurtosis(signal):
    """Calculates the kurtosis of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The kurtosis of the audio signal.
    """
    return (1 / len(signal)) * np.sum(((signal - mean(signal)) / standard_deviation(signal)) ** 4) - 3


def root_mean_square(signal):
    """Calculates the root mean square of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The root mean square of the audio signal.
    """
    return np.sqrt(np.mean(np.square(signal)))


def zero_crossings(signal):
    """Calculates the number of zero crossings in an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        int: The number of zero crossings in the audio signal.
    """
    return np.sum(np.diff(np.signbit(signal)))


def zero_crossing_rate(signal):
    """Calculates the zero crossing rate of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        float: The zero crossing rate of the audio signal.
    """
    return zero_crossings(signal) / len(signal)


def spectrum(signal):
    """Calculates the spectrum of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        np.ndarray: The spectrum of the audio signal.
    """
    return np.fft.fft(signal)


def amplitude_spectrum(signal):
    """Calculates the amplitude spectrum of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        np.ndarray: The amplitude spectrum of the audio signal.
    """
    return np.abs(spectrum(signal))


def power_spectrum(signal):
    """Calculates the power spectrum of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

    Returns:
        np.ndarray: The power spectrum of the audio signal.
    """
    return amplitude_spectrum(signal) ** 2


def spectrum_frequencies(signal, sampling_frequency):
    """Calculates the spectrum frequencies of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.
        sampling_frequency (int): The sampling frequency of the audio signal.

    Returns:
        np.ndarray: The spectrum frequencies of the audio signal.
    """
    return np.fft.fftfreq(len(signal), 1 / sampling_frequency)


def spectral_centroid(signal, sampling_frequency):
    """Calculates the spectral centroid of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.
        sampling_frequency (int): The sampling frequency of the audio signal.

    Returns:
        float: The spectral centroid of the audio signal.
    """
    return np.sum(spectrum_frequencies(signal, sampling_frequency) * amplitude_spectrum(signal) / np.sum(amplitude_spectrum(signal)))


def spectral_bandwidth(signal, sampling_frequency):
    """Calculates the spectral bandwidth of an audio signal.

    Args:
        signal (np.ndarray): The audio signal.
        sampling_frequency (int): The sampling frequency of the audio signal.

    Returns:
        float: The spectral bandwidth of the audio signal.
    """
    return np.sqrt(np.sum(power_spectrum(signal) * (spectrum_frequencies(signal, sampling_frequency) - spectral_centroid(signal, sampling_frequency)) ** 2) / np.sum(power_spectrum(signal)))


def feature_extractor(signal, features=["root_mean_square", "zero_crossing_rate"]):
    """Extracts features from an audio signal.

    Args:
        signal (np.ndarray): The audio signal.

        features (list, optional): The list of features to extract. Defaults
        to ["root_mean_square", "zero_crossing_rate"].

    Returns:
        np.ndarray: Extracted features.
    """
    features_values = []
    for feature in features:
        if feature == "root_mean_square":
            features_values.append(root_mean_square(signal))
        elif feature == "zero_crossing_rate":
            features_values.append(zero_crossing_rate(signal))

    return np.array(features_values)