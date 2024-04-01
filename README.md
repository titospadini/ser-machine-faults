# SER for Machine Faults Classification
Sound Event Recognition (SER) for machine faults classification.

At this moment, this project is still in development.

This project uses Python 3.12.2 with the libraries*:
- numpy
- soundfile
- matplotlib
- scikit-learn
- pandas
- jupyterlab

\* check the [requirements.txt](https://github.com/titospadini/ser-machine-faults/blob/main/requirements.txt) file for more details.


## Small and simple dataset

This part of the project is in [simple_classification.ipynb](https://github.com/titospadini/ser-machine-faults/blob/main/simple_classification.ipynb).

For initial testing purposes, this project was using audio files from the following repository: https://github.com/RashadShubita/Fault-Detection-using-TinyML.


### Audio files and classes
There are 5 audio files, each one representing a different class:
- "M1_OFF_S1.flac": 0 ("Off")
- "M1_H_S1.flac":   1 ("Health")
- "M1_F1_S1.flac":  2 ("Bearing Fault")
- "M1_F2_S1.flac":  3 ("Fan Fault")
- "M1_F3_S1.flac":  4 ("Gear Fault")


### Audio specifications
These audio files have the following specifications:
- Sampling Frequency:   48000 Hz
- Bit Depth:            24 bits
- Encoding:             FLAC
- Channels:             2 (Stereo)
- Duration:             ~ 1 minute per file (~ 5 minutes and 8 seconds in total)

For now, this project is using the following specifications**:
- Sampling Frequency:   16000 Hz
- Bit Depth:            16 bits
- Channels:             1 (Mono)

\** not mentioned specifications here are assumed to be the same as the ones in the audio files.


### Audio preparation
To achieve the desired audio specifications, the audio files were prepared using the following steps in Python with SoundFile and Numpy libraries plus some custom-made functions (see [utils.py](https://github.com/titospadini/ser-machine-faults/blob/main/utils.py) for more details):
1. read the audio file;
2. mixing down the channels from stereo to mono;
3. normalize the signal to -6 dBFS;
4. resample the signal from 48000 Hz to 16000 Hz;
5. remove silence intervals considering a threshold of -48 dBFS;
5. segment the signal into 500 ms segments with 100 ms overlap.


### Feature extraction
Then, for each segment the following features were extracted (see [features.py](https://github.com/titospadini/ser-machine-faults/blob/main/features.py) for more details):
- mean
- median
- variance
- standard deviation
- skewness
- kurtosis
- root mean square (RMS)
- zero crossing rate (ZCR)


### Machine learning
Then, these features were concatenated into a feature matrix X, and each segment's class number was appended to the class vector y.

From the feature matrix X and the class vector y, the machine learning steps were performed using Scikit-Learn library.


#### Data splitting
The data was split into a training set (80%) and a test set (20%), with random state set to 42.


#### Model training
For now, the selected classifier is a simple Decision Tree, and the cross-validation method is a stratified 5-fold.


#### Model evaluation
The model achieved a training mean accuracy of 99.22% and a test mean accuracy of 99%.