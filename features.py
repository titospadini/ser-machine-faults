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