import numpy as np

def mean(signal):
    """Calculates the mean of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The mean of the signal.
    """
    return np.mean(signal)


def variance(signal):
    """Calculates the variance of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The variance of the signal.
    """
    return np.var(signal)


def standard_deviation(signal):
    """Calculates the standard deviation of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The standard deviation of the signal.
    """
    return np.std(signal)


def median(signal):
    """Calculates the median of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The median of the signal.
    """
    return np.median(signal)


def skewness(signal):
    """Calculates the skewness of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The skewness of the signal.
    """
    return (1 / len(signal)) * np.sum(((signal - mean(signal)) / standard_deviation(signal)) ** 3)


def kurtosis(signal):
    """Calculates the kurtosis of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The kurtosis of the signal.
    """
    return (1 / len(signal)) * np.sum(((signal - mean(signal)) / standard_deviation(signal)) ** 4) - 3


def energy(signal):
    """Calculates the energy of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The energy of the signal.
    """
    return np.sum(np.square(signal))


def root_mean_square(signal):
    """Calculates the root mean square of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The root mean square of the signal.
    """
    return np.sqrt(np.mean(np.square(signal)))


def zero_crossings(signal):
    """Calculates the number of zero crossings in a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        int: The number of zero crossings in the signal.
    """
    return np.sum(np.diff(np.signbit(signal)))


def zero_crossing_rate(signal):
    """Calculates the zero crossing rate of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The zero crossing rate of the signal.
    """
    return zero_crossings(signal) / len(signal)


def signal_peaks(signal, n_peaks=1):
    """Calculates the peaks of a signal.

    Args:
        signal (np.ndarray): The signal.
        n_peaks (int, optional): The number of peaks. Defaults to 1.

    Returns:
        np.ndarray: The peaks of the signal.
    """
    signal = np.asarray(signal)

    if n_peaks > signal.shape[0]:
        raise ValueError("The number of peaks cannot be greater than the number of samples.")

    return np.flip(signal[np.argpartition(signal, -n_peaks)[-n_peaks:]])


def peak_to_peak(signal):
    """Calculates the peak-to-peak amplitude of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        float: The peak-to-peak amplitude of the signal.
    """
    return np.max(signal) - np.min(signal)


def spectrum(signal):
    """Calculates the spectrum of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        np.ndarray: The spectrum of the signal.
    """
    return np.fft.fft(signal)


def amplitude_spectrum(signal):
    """Calculates the amplitude spectrum of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        np.ndarray: The amplitude spectrum of the signal.
    """
    return np.abs(spectrum(signal))


def power_spectrum(signal):
    """Calculates the power spectrum of a signal.

    Args:
        signal (np.ndarray): The signal.

    Returns:
        np.ndarray: The power spectrum of the signal.
    """
    return amplitude_spectrum(signal) ** 2


def spectrum_frequencies(signal, sampling_frequency):
    """Calculates the spectrum frequencies of a signal.

    Args:
        signal (np.ndarray): The signal.
        sampling_frequency (int): The sampling frequency of the signal.

    Returns:
        np.ndarray: The spectrum frequencies of the signal.
    """
    return np.fft.fftfreq(len(signal), 1 / sampling_frequency)


def spectral_centroid(signal, sampling_frequency):
    """Calculates the spectral centroid of a signal.

    Args:
        signal (np.ndarray): The signal.
        sampling_frequency (int): The sampling frequency of the signal.

    Returns:
        float: The spectral centroid of the signal.
    """
    return np.sum(spectrum_frequencies(signal, sampling_frequency) * amplitude_spectrum(signal) / np.sum(amplitude_spectrum(signal)))


def spectral_bandwidth(signal, sampling_frequency):
    """Calculates the spectral bandwidth of a signal.

    Args:
        signal (np.ndarray): The signal.
        sampling_frequency (int): The sampling frequency of the signal.

    Returns:
        float: The spectral bandwidth of the signal.
    """
    return np.sqrt(np.sum(power_spectrum(signal) * (spectrum_frequencies(signal, sampling_frequency) - spectral_centroid(signal, sampling_frequency)) ** 2) / np.sum(power_spectrum(signal)))


def feature_extractor(signal, features=["root_mean_square", "zero_crossing_rate"]):
    """Extracts features from a signal.

    Args:
        signal (np.ndarray): The signal.

        features (list, optional): The list of features to extract. Defaults
        to ["root_mean_square", "zero_crossing_rate"].

    Returns:
        np.ndarray: Extracted features.
    """
    features_values = []
    for feature in features:
        if feature == "mean":
            features_values.append(mean(signal))
        elif feature == "median":
            features_values.append(median(signal))
        elif feature == "variance":
            features_values.append(variance(signal))
        elif feature == "standard_deviation":
            features_values.append(standard_deviation(signal))
        elif feature == "skewness":
            features_values.append(skewness(signal))
        elif feature == "kurtosis":
            features_values.append(kurtosis(signal))
        elif feature == "root_mean_square":
            features_values.append(root_mean_square(signal))
        elif feature == "zero_crossing_rate":
            features_values.append(zero_crossing_rate(signal))
        elif feature == "peak_to_peak":
            features_values.append(peak_to_peak(signal))

    return np.array(features_values)


def get_features(signals, features=["root_mean_square", "zero_crossing_rate"]):
    """Extracts features from all input signals. Each row in the input shall be treated as a different signal.

    Args:
        signals (np.ndarray): The signals.

        features (list, optional): The list of features to extract. Defaults
        to ["root_mean_square", "zero_crossing_rate"].

    Returns:
        np.ndarray: Extracted features.
    """
    features_lst = []
    for signal in signals:
        features_lst.append(feature_extractor(signal, features))
    return np.array(features_lst)