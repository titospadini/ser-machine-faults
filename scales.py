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
            return 15.0 + 27.0 * np.log10(frequency / 1000.0) / np.log10(6.4)
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