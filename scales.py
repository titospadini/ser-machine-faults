import numpy as np

def hertz_to_mel(frequency, method="slaney"):
    """Converts a frequency from Hertz to Mel scale.

    Args:
        frequency (float): The frequency in Hertz.
        method (str, optional): The method to convert the frequency.
            Valid methods are:
                - "slaney" (Default)
                - "oshaughnessy"
                - "fant"
                - "norman"
                - "htk"

    Raises:
        ValueError: Invalid method.

    Returns:
        float: The frequency in Mel scale.
    """
    if method == "slaney":
        if frequency < 1000.0:
            return 3.0 * frequency / 200.0
        else:
            return 15.0 + 27.0 * np.log(frequency / 1000.0) / np.log(6.4)
    elif method == "oshaughnessy":
        return 2595.0 * np.log(1 + frequency / 700.0)
    elif method == "fant":
        return 1000.0 / np.log(2) * np.log(1 + frequency / 1000.0)
    elif method == "norman":
        return 2410.0 * np.log10(1 + frequency / 625.0)
    elif method == "htk":
        return 1127.0 * np.log(1 + frequency / 700.0)
    else:
        raise ValueError(f"Invalid method: {method}.")


def mel_to_hertz(frequency, method="slaney"):
    """Converts a frequency from Mel scale to Hertz.

    Args:
        frequency (float): The frequency in Mel scale.
        method (str, optional): The method to convert the frequency.
            Valid methods are:
                - "slaney" (Default)
                - "oshaughnessy"
                - "fant"
                - "norman"
                - "htk"

    Raises:
        ValueError: Invalid method.

    Returns:
        float: The frequency in Hertz.
    """
    if method == "slaney":
        if frequency < 1000.0:
            return 200.0 * frequency / 3.0
        else:
            return 1000.0 * np.exp(np.log(6.4) / 27.0 * (frequency - 15.0))
    elif method == "oshaughnessy":
        return 700 * (np.exp(frequency / 2595.0) - 1.0)
    elif method == "fant":
        return 1000.0 * (np.exp(frequency / (1000.0 / np.log(2))) - 1)
    elif method == "norman":
        return 625.0 * (10 ** (frequency / 2410.0) - 1)
    elif method == "htk":
        return 700.0 * (np.exp(frequency / 1127.0) - 1)
    else:
        raise ValueError(f"Invalid method: {method}.")


def hertz_to_bark(frequency, method="wang"):
    """Converts a frequency from Hertz to Bark scale.

    Args:
        frequency (float): The frequency in Hertz.
        method (str, optional): The method to convert the frequency.
            Valid methods are:
                - "wang" (Default)
                - "traunmuller"
                - "zwicker"

    Returns:
        float: The frequency in Bark scale.
    """
    if method == "wang":
        return 6.0 * np.arcsinh(frequency / 600.0)
    elif method == "traunmuller":
        return 26.81 * frequency / (1960.0 + frequency) - 0.53
    elif method == "zwicker":
        return 13.0 * np.arctan(0.00076 * frequency) + 3.5 * np.arctan((frequency / 7500.0) ** 2)
    else:
        raise ValueError(f"Invalid method: {method}.")


def bark_to_hertz(frequency, method="wang"):
    """Converts a frequency from Bark scale to Hertz.

    Args:
        frequency (float): The frequency in Bark scale.
        method (str, optional): The method to convert the frequency.
            Valid methods are:
                - "wang" (Default)
                - "traunmuller"
                - "zwicker"

    Returns:
        float: The frequency in Hertz.
    """
    if method == "wang":
        return 600.0 * np.sinh(frequency / 6.0)
    elif method == "traunmuller":
        return (1960.0 + frequency) * (frequency + 0.53) / 26.81
    elif method == "zwicker":
        return np.tan((frequency - 3.5 * np.arctan((frequency / 7500.0) ** 2)) / 13.0) / 0.00076
    else:
        raise ValueError(f"Invalid method: {method}.")


def hertz_to_erb(frequency, method="linear"):
    """Converts a frequency from Hertz to Equivalent Rectangular Bandwidth (ERB).

    Args:
        frequency (float): The frequency in Hertz.
        method (str, optional): The method to convert the frequency.
            Valid methods are:
                - "linear" (Default)
                - "polynomial"

    Returns:
        float: The frequency in ERB.
    """
    if method == "linear":
        return 24.7 * (0.00437 * frequency + 1)
    elif method == "polynomial":
        return 6.23 * frequency ** 2 + 93.39 * frequency + 28.52
    else:
        raise ValueError(f"Invalid method: {method}.")


def erb_to_hertz(frequency, method="linear"):
    """Converts a frequency from Equivalent Rectangular Bandwidth (ERB) to Hertz.

    Args:
        frequency (float): The frequency in ERB.
        method (str, optional): The method to convert the frequency.
            Valid methods are:
                - "linear" (Default)
                - "polynomial"

    Returns:
        float: The frequency in Hertz.
    """
    if method == "linear":
        return (frequency / 24.7 - 1) / 0.00437
    elif method == "polynomial":
        return 1000 * ((0.312 - (np.exp((frequency - 43.0) / 11.17)) * 14.675) / (np.exp((frequency - 43.0) / 11.17) - 1.0))
    else:
        raise ValueError(f"Invalid method: {method}.")