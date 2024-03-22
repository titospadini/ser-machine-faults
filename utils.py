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