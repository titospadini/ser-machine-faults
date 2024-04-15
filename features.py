import numpy as np
from scipy import stats
import librosa

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

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

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.sum(framed_signal**2, axis=-2, keepdims=True) / framed_signal.shape[-2]


def min(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the min along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The min along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.min(framed_signal, axis=-2, keepdims=True)


def max(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the max along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The max along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.max(framed_signal, axis=-2, keepdims=True)


def peak_to_peak(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the peak to peak along the last axis.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults Whoever.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The peak to peak along the last axis.
    """
    signal = np.asarray(signal)

    if center:
        padding = [(0, 0) for _ in range(signal.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        signal = np.pad(signal, padding, mode=pad_mode)

    framed_signal = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

    return np.max(framed_signal, axis=-2, keepdims=True) - np.min(framed_signal, axis=-2, keepdims=True)


def root_mean_square(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the root mean square along the last axis.

    This is simply a wrapper of librosa.feature.rmse.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The root mean square along the last axis.
    """
    return librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def zero_crossing_rate(signal, frame_length=2048, hop_length=512, center=True):
    """ Compute the zero crossing rate along the last axis.

    This is simply a wrapper of librosa.feature.zero_crossing_rate.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The zero crossing rate along the last axis.
    """
    return librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length, center=center)


def spectral_centroid(signal, sampling_frequency=16000, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the spectral centroid along the last axis.

    This is simply a wrapper of librosa.feature.spectral_centroid.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The spectral centroid along the last axis.
    """
    return librosa.feature.spectral_centroid(y=signal, sr=sampling_frequency, n_fft=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def spectral_bandwidth(signal, sampling_frequency=16000, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the spectral bandwidth along the last axis.

    This is simply a wrapper of librosa.feature.spectral_bandwidth.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The spectral bandwidth along the last axis.
    """
    return librosa.feature.spectral_bandwidth(y=signal, sr=sampling_frequency, n_fft=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def spectral_rolloff(signal, sampling_frequency=16000, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the spectral rolloff along the last axis.

    This is simply a wrapper of librosa.feature.spectral_rolloff.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The spectral rolloff along the last axis.
    """
    return librosa.feature.spectral_rolloff(y=signal, sr=sampling_frequency, n_fft=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def spectral_flatness(signal, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the spectral flatness along the last axis.

    This is simply a wrapper of librosa.feature.spectral_flatness.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The spectral flatness along the last axis.
    """
    return librosa.feature.spectral_flatness(y=signal, n_fft=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def spectral_contrast(signal, sampling_frequency=16000, n_contrast_bands=5, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the spectral contrast along the last axis.

    This is simply a wrapper of librosa.feature.spectral_contrast.

    Args:
        signal (np.ndarray): The signal.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The spectral contrast along the last axis.
    """
    return librosa.feature.spectral_contrast(y=signal, sr=sampling_frequency, n_fft=frame_length, n_bands=n_contrast_bands, hop_length=hop_length, center=center, pad_mode=pad_mode)


def tonnetz(signal, sampling_frequency=16000, chroma=None, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the tonal centroid along the last axis.

    This is simply a wrapper of librosa.feature.tonnetz.

    Args:
        signal (np.ndarray): The signal.
        sampling_frequency (int, optional): The sampling frequency. Defaults to 16000.
        chroma (np.ndarray, optional): The chroma matrix. Defaults to None.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The tonal centroid along the last axis.
    """
    return librosa.feature.tonnetz(y=signal, sr=sampling_frequency, chroma=chroma, n_fft=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def mfcc(signal, sampling_frequency=16000, n_mfcc=20, dct_type=2, norm="ortho", lifter=0, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the mel-frequency cepstral coefficients along the last axis.

    This is simply a wrapper of librosa.feature.mfcc.

    Args:
        signal (np.ndarray): The signal.
        sampling_frequency (int, optional): The sampling frequency. Defaults to 16000.
        n_mfcc (int, optional): The number of mel-frequency cepstral coefficients. Defaults to 20.
        dct_type (int, optional): The type of DCT. Defaults to 2.
        norm (str, optional): The normalization. Defaults to "ortho".
        lifter (int, optional): The lifter. Defaults to 0.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The mel-frequency cepstral coefficients along the last axis.
    """
    return librosa.feature.mfcc(y=signal, sr=sampling_frequency, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, n_fft=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)


def delta_mfcc(signal, width=9, order=1, axis=-1, mode="interp", sampling_frequency=16000, n_mfcc=20, dct_type=2, norm="ortho", lifter=0, frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Compute the delta mel-frequency cepstral coefficients along the last axis.

    This is simply a wrapper of librosa.feature.delta.

    Args:
        signal (np.ndarray): The signal.
        width (int, optional): The width. Defaults to 9.
        order (int, optional): The order. Defaults to 1.
        axis (int, optional): The axis. Defaults to -1.
        mode (str, optional): The mode. Defaults to "interp".
        sampling_frequency (int, optional): The sampling frequency. Defaults to 16000.
        n_mfcc (int, optional): The number of mel-frequency cepstral coefficients. Defaults to 20.
        dct_type (int, optional): The type of DCT. Defaults to 2.
        norm (str, optional): The normalization. Defaults to "ortho".
        lifter (int, optional): The lifter. Defaults to 0.
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Returns:
        np.ndarray: The delta mel-frequency cepstral coefficients along the last axis.
    """
    mfcc = mfcc(signal, sampling_frequency=sampling_frequency, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
    return librosa.feature.delta(mfcc, width=width, order=order, axis=axis, mode=mode)


def feature_extractor(signal, features, sampling_frequency=16000, n_contrast_bands=5, chroma=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0, width=9, axis=-1, mode="interp", frame_length=2048, hop_length=512, center=True, pad_mode="constant"):
    """ Extract features from a signal.

    Args:
        signal (np.ndarray): The signal.
        features (list): The list of features to extract.
        sampling_frequency (int, optional): The sampling frequency. Defaults to 16000.
        n_contrast_bands (int, optional): The number of contrast bands for spectral contrast. Defaults to 5.
        chroma (np.ndarray, optional): The chroma matrix for Tonnetz feature. Defaults to None.
        n_mfcc (int, optional): The number of mel-frequency cepstral coefficients. Defaults to 20.
        dct_type (int, optional): The type of DCT for MFCC. Defaults to 2.
        norm (str, optional): The normalization for MFCC. Defaults to "ortho".
        lifter (int, optional): The lifter for MFCC. Defaults to 0.
        width (int, optional): The width for delta MFCC. Defaults to 9.
        axis (int, optional): The axis for delta MFCC. Defaults to -1.
        mode (str, optional): The mode for delta MFCC. Defaults to "interp".
        frame_length (int, optional): The frame length. Defaults to 2048.
        hop_length (int, optional): The hop length. Defaults to 512.
        center (bool, optional): Pad the signal by half the frame length. Defaults to True.
        pad_mode (str, optional): The padding mode. Defaults to "constant".

    Raises:
        ValueError: Invalid feature.

    Returns:
        np.ndarray: The extracted features.
    """
    feature_lst     = []
    component_lst   = []

    for feature in features:
        if feature == "mean":
            feature_values = mean(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "median":
            feature_values = median(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "variance":
            feature_values = variance(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "standard_deviation":
            feature_values = standard_deviation(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "skewness":
            feature_values = skewness(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "kurtosis":
            feature_values = kurtosis(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "energy":
            feature_values = energy(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "power":
            feature_values = power(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "min":
            feature_values = min(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "max":
            feature_values = max(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "peak_to_peak":
            feature_values = peak_to_peak(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "root_mean_square":
            feature_values = root_mean_square(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "zero_crossing_rate":
            feature_values = zero_crossing_rate(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "spectral_centroid":
            feature_values = spectral_centroid(signal=signal, sampling_frequency=sampling_frequency, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "spectral_bandwidth":
            feature_values = spectral_bandwidth(signal=signal, sampling_frequency=sampling_frequency, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "spectral_rolloff":
            feature_values = spectral_rolloff(signal=signal, sampling_frequency=sampling_frequency, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "spectral_flatness":
            feature_values = spectral_flatness(signal=signal, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "spectral_contrast":
            feature_values = spectral_contrast(signal=signal, sampling_frequency=sampling_frequency, n_contrast_bands=n_contrast_bands, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "tonnetz":
            feature_values = tonnetz(signal=signal, sampling_frequency=sampling_frequency, chroma=chroma, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "mfcc":
            feature_values = mfcc(signal=signal, sampling_frequency=sampling_frequency, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "delta_mfcc":
            feature_values = delta_mfcc(signal=signal, sampling_frequency=sampling_frequency, n_mfcc=n_mfcc, width=width, axis=axis, mode=mode, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        elif feature == "delta_delta_mfcc":
            feature_values = delta_mfcc(signal=signal, sampling_frequency=sampling_frequency, n_mfcc=n_mfcc, width=width, axis=axis, mode=mode, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
            feature_components = feature_values.shape[0]
            component_lst.append(feature_components)
            for i in range(feature_components):
                feature_lst.append(feature_values[i])
        else:
            raise ValueError(f"Invalid feature: {feature}.")

    return np.array(feature_lst), component_lst