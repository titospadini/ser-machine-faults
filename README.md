# SER for Machine Faults Classification
Sound Event Recognition (SER) for machine faults classification.

This project is under development on Ubuntu 24.04 LTS running via WSL on a Windows 11 Pro desktop.

## Installation and usage
This project uses Miniconda and Python 3.12.3 interpreter with the following libraries*:
- numpy
- soundfile
- matplotlib
- scikit-learn
- pandas
- jupyterlab

\* check the [requirements.txt](https://github.com/titospadini/ser-machine-faults/blob/main/requirements.txt) file for more details.

It is recommended to create a virtual environment with the following command:
`conda create -n machine-faults python=3.12.3`

Then, to use the virtual environment, activate it with:
`conda activate machine-faults`

To install the dependencies (if not already installed), run:
`pip install -r requirements.txt`

To use the Jupyter notebooks, run:
`jupyter notebook` or `jupyter lab`

When done, deactivate the environment with:
`conda deactivate`


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


## MaFaulDa
This part of the project is in [mafaulda.ipynb](https://github.com/titospadini/ser-machine-faults/blob/main/mafaulda.ipynb).

The [Machinery Fault Database (MaFaulDa) dataset](http://www02.smt.ufrj.br/~offshore/mfs/) was created by Felipe M. L. Ribeiro (2016).

Despite MaFaulDa has 8 channels of data, the only data used here comes from the 8th channel, which is the microphone (Shure SM81) mono signal.

### Classes
There are 6 folders, each one representing a different main class:
- "normal":                     0
- "horizontal-misalignment":    1
- "vertical-misalignment":      2
- "imbalance":                  3
- "overhang":                   4
- "underhang":                  5

### Scenarios
The "normal" class does not have different types of scenarios.

The "horizontal-misalignment" has different scenarios, but only variations of the length of the misalignment: 0.5 mm, 1.0 mm, 1.5 mm, 2.0 mm.

The "vertical-misalignment" also has variations of misalignment: 0.51 mm, 0.63 mm, 1.27 mm, 1.40 mm, 1.78 mm, 1.90 mm.

The "imbalance" has variations of weight: 6 g, 10 g, 15 g, 20 g, 25 g, 30 g, 35 g.

The "overhang" is a bit different. It has three types of scenarios, which are:

- "ball_fault": 0 g, 6 g, 20 g, 35 g.
- "cage_fault": 0 g, 6 g, 20 g, 35 g.
- "outer_race": 0 g, 6 g, 20 g, 35 g.

The "underhang" also has three types of scenarios, which are:

- "ball_fault": 0 g, 6 g, 20 g, 35 g.
- "cage_fault": 0 g, 6 g, 20 g, 35 g.
- "outer_race": 0 g, 6 g, 20 g, 35 g.

So, there are 6 classes, but, if you count every single different possible fault scenario as a different class, there will be 42 classes.


### Measurements
Each measurement has a duration of 5 seconds. The number of measurements per class are:

- Normal: 49 measurements
- Horizontal misalignment: 197 measurements
- Vertical misalignment: 301 measurements
- Imbalance: 333 measurements
- Overhang bearing: 188 + 188 + 137 (total of 513 measurements)
- Underhang bearing: 188 + 184 + 186 (total of 558 measurements)


### Audio specifications
These audio files have the following specifications:
- Sampling Frequency:   50000 Hz
- Bit Depth:            24 bits
- Encoding:             Raw
- Channels:             1 (only 1 of the 8 signals is an audio signal)
- Duration:             ~ 5 seconds per file (~ 2 hours and 42 minutes in total)

For now, this project is using the following specifications**:
- Sampling Frequency:   16000 Hz
- Bit Depth:            16 bits
- Channels:             1 (Mono)

\** not mentioned specifications here are assumed to be the same as the ones in the audio files.