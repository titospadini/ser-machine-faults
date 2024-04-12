import os

import numpy as np
import pandas as pd

from features import feature_extractor
from utils import *

# ===========================
# ----- hyperparameters -----
# ===========================
BASE_DIR = "/home/tito/datasets/MAFAULDA"

ORIGINAL_SAMPLING_FREQUENCY = 50000 # 50 kHz
NEW_SAMPLING_FREQUENCY      = 16000 # 16 kHz
NORMALIZE_DBFS              = -6    # -6 dBFS
SILENCE_THRESHOLD_DBFS      = -48   # -48 dBFS
SEGMENT_DURATION            = 500   # 500 ms
OVERLAP_DURATION            = 100   # 100 ms

FRAME_LENGTH                = int(SEGMENT_DURATION / 1000 * NEW_SAMPLING_FREQUENCY)
HOP_LENGTH                  = int((SEGMENT_DURATION - OVERLAP_DURATION) / 1000 * NEW_SAMPLING_FREQUENCY)


# ====================
# ----- Features -----
# ====================
FEATURES_LST = [
                "mean",
                "median",
                "variance",
                "standard_deviation",
                "skewness",
                "kurtosis",
                "energy",
                "power",
                "min",
                "max",
                "peak_to_peak",
                "root_mean_square",
                "zero_crossing_rate",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "spectral_flatness",
                "spectral_contrast",
                ]



# ======================================================
# -------------------- Directories  --------------------
# ======================================================
# base directories for each class
DIR_NORMAL                  = f"{BASE_DIR}/normal"
DIR_HORIZONTAL_MISALIGNMENT = f"{BASE_DIR}/horizontal-misalignment"
DIR_VERTICAL_MISALIGNMENT   = f"{BASE_DIR}/vertical-misalignment"
DIR_IMBALANCE               = f"{BASE_DIR}/imbalance"
DIR_OVERHANG                = f"{BASE_DIR}/overhang"
DIR_UNDERHANG               = f"{BASE_DIR}/underhang"


# subdirectories for HORIZONTAL_MISALIGNMENT
DIR_HORIZONTAL_MISALIGNMENT_05 = f"{DIR_HORIZONTAL_MISALIGNMENT}/0.5mm"
DIR_HORIZONTAL_MISALIGNMENT_10 = f"{DIR_HORIZONTAL_MISALIGNMENT}/1.0mm"
DIR_HORIZONTAL_MISALIGNMENT_15 = f"{DIR_HORIZONTAL_MISALIGNMENT}/1.5mm"
DIR_HORIZONTAL_MISALIGNMENT_20 = f"{DIR_HORIZONTAL_MISALIGNMENT}/2.0mm"

DIR_HORIZONTAL_MISALIGNMENT_LST = [
    DIR_HORIZONTAL_MISALIGNMENT_05,
    DIR_HORIZONTAL_MISALIGNMENT_10,
    DIR_HORIZONTAL_MISALIGNMENT_15,
    DIR_HORIZONTAL_MISALIGNMENT_20
]


# subdirectories for VERTICAL_MISALIGNMENT
DIR_VERTICAL_MISALIGNMENT_051 = f"{DIR_VERTICAL_MISALIGNMENT}/0.51mm"
DIR_VERTICAL_MISALIGNMENT_063 = f"{DIR_VERTICAL_MISALIGNMENT}/0.63mm"
DIR_VERTICAL_MISALIGNMENT_127 = f"{DIR_VERTICAL_MISALIGNMENT}/1.27mm"
DIR_VERTICAL_MISALIGNMENT_140 = f"{DIR_VERTICAL_MISALIGNMENT}/1.40mm"
DIR_VERTICAL_MISALIGNMENT_178 = f"{DIR_VERTICAL_MISALIGNMENT}/1.78mm"
DIR_VERTICAL_MISALIGNMENT_190 = f"{DIR_VERTICAL_MISALIGNMENT}/1.90mm"

DIR_VERTICAL_MISALIGNMENT_LST = [
    DIR_VERTICAL_MISALIGNMENT_051,
    DIR_VERTICAL_MISALIGNMENT_063,
    DIR_VERTICAL_MISALIGNMENT_127,
    DIR_VERTICAL_MISALIGNMENT_140,
    DIR_VERTICAL_MISALIGNMENT_178,
    DIR_VERTICAL_MISALIGNMENT_190
]


# subdirectories for IMBALANCE
DIR_IMBALANCE_06 = f"{DIR_IMBALANCE}/6g"
DIR_IMBALANCE_10 = f"{DIR_IMBALANCE}/10g"
DIR_IMBALANCE_15 = f"{DIR_IMBALANCE}/15g"
DIR_IMBALANCE_20 = f"{DIR_IMBALANCE}/20g"
DIR_IMBALANCE_25 = f"{DIR_IMBALANCE}/25g"
DIR_IMBALANCE_30 = f"{DIR_IMBALANCE}/30g"
DIR_IMBALANCE_35 = f"{DIR_IMBALANCE}/35g"

DIR_IMBALANCE_LST = [
    DIR_IMBALANCE_06,
    DIR_IMBALANCE_10,
    DIR_IMBALANCE_15,
    DIR_IMBALANCE_20,
    DIR_IMBALANCE_25,
    DIR_IMBALANCE_30,
    DIR_IMBALANCE_35
]


# subdirectories for OVERHANG
DIR_OVERHANG_BALL = f"{DIR_OVERHANG}/ball_fault"
DIR_OVERHANG_BALL_00 = f"{DIR_OVERHANG_BALL}/0g"
DIR_OVERHANG_BALL_06 = f"{DIR_OVERHANG_BALL}/6g"
DIR_OVERHANG_BALL_20 = f"{DIR_OVERHANG_BALL}/20g"
DIR_OVERHANG_BALL_35 = f"{DIR_OVERHANG_BALL}/35g"

DIR_OVERHANG_CAGE = f"{DIR_OVERHANG}/cage_fault"
DIR_OVERHANG_CAGE_00 = f"{DIR_OVERHANG_CAGE}/0g"
DIR_OVERHANG_CAGE_06 = f"{DIR_OVERHANG_CAGE}/6g"
DIR_OVERHANG_CAGE_20 = f"{DIR_OVERHANG_CAGE}/20g"
DIR_OVERHANG_CAGE_35 = f"{DIR_OVERHANG_CAGE}/35g"

DIR_OVERHANG_OUTER = f"{DIR_OVERHANG}/outer_race"
DIR_OVERHANG_OUTER_00 = f"{DIR_OVERHANG_OUTER}/0g"
DIR_OVERHANG_OUTER_06 = f"{DIR_OVERHANG_OUTER}/6g"
DIR_OVERHANG_OUTER_20 = f"{DIR_OVERHANG_OUTER}/20g"
DIR_OVERHANG_OUTER_35 = f"{DIR_OVERHANG_OUTER}/35g"

DIR_OVERHANG_LST = [
    DIR_OVERHANG_BALL_00,
    DIR_OVERHANG_BALL_06,
    DIR_OVERHANG_BALL_20,
    DIR_OVERHANG_BALL_35,
    DIR_OVERHANG_CAGE_00,
    DIR_OVERHANG_CAGE_06,
    DIR_OVERHANG_CAGE_20,
    DIR_OVERHANG_CAGE_35,
    DIR_OVERHANG_OUTER_00,
    DIR_OVERHANG_OUTER_06,
    DIR_OVERHANG_OUTER_20,
    DIR_OVERHANG_OUTER_35
]


# subdirectories for UNDERGANG
DIR_UNDERHANG_BALL = f"{DIR_UNDERHANG}/ball_fault"
DIR_UNDERHANG_BALL_00 = f"{DIR_UNDERHANG_BALL}/0g"
DIR_UNDERHANG_BALL_06 = f"{DIR_UNDERHANG_BALL}/6g"
DIR_UNDERHANG_BALL_20 = f"{DIR_UNDERHANG_BALL}/20g"
DIR_UNDERHANG_BALL_35 = f"{DIR_UNDERHANG_BALL}/35g"

DIR_UNDERHANG_CAGE = f"{DIR_UNDERHANG}/cage_fault"
DIR_UNDERHANG_CAGE_00 = f"{DIR_UNDERHANG_CAGE}/0g"
DIR_UNDERHANG_CAGE_06 = f"{DIR_UNDERHANG_CAGE}/6g"
DIR_UNDERHANG_CAGE_20 = f"{DIR_UNDERHANG_CAGE}/20g"
DIR_UNDERHANG_CAGE_35 = f"{DIR_UNDERHANG_CAGE}/35g"

DIR_UNDERHANG_OUTER = f"{DIR_UNDERHANG}/outer_race"
DIR_UNDERHANG_OUTER_00 = f"{DIR_UNDERHANG_OUTER}/0g"
DIR_UNDERHANG_OUTER_06 = f"{DIR_UNDERHANG_OUTER}/6g"
DIR_UNDERHANG_OUTER_20 = f"{DIR_UNDERHANG_OUTER}/20g"
DIR_UNDERHANG_OUTER_35 = f"{DIR_UNDERHANG_OUTER}/35g"

DIR_UNDERGANG_LST = [
    DIR_UNDERHANG_BALL_00,
    DIR_UNDERHANG_BALL_06,
    DIR_UNDERHANG_BALL_20,
    DIR_UNDERHANG_BALL_35,
    DIR_UNDERHANG_CAGE_00,
    DIR_UNDERHANG_CAGE_06,
    DIR_UNDERHANG_CAGE_20,
    DIR_UNDERHANG_CAGE_35,
    DIR_UNDERHANG_OUTER_00,
    DIR_UNDERHANG_OUTER_06,
    DIR_UNDERHANG_OUTER_20,
    DIR_UNDERHANG_OUTER_35
]


DIR_ALL = [
    DIR_NORMAL,                         # 0     -   Normal
    DIR_HORIZONTAL_MISALIGNMENT_05,     # 1     -   Horizontal Misalignment
    DIR_HORIZONTAL_MISALIGNMENT_10,     # 2     -   Horizontal Misalignment
    DIR_HORIZONTAL_MISALIGNMENT_15,     # 3     -   Horizontal Misalignment
    DIR_HORIZONTAL_MISALIGNMENT_20,     # 4     -   Horizontal Misalignment
    DIR_VERTICAL_MISALIGNMENT_051,      # 5     -   Vertical Misalignment
    DIR_VERTICAL_MISALIGNMENT_063,      # 6     -   Vertical Misalignment
    DIR_VERTICAL_MISALIGNMENT_127,      # 7     -   Vertical Misalignment
    DIR_VERTICAL_MISALIGNMENT_140,      # 8     -   Vertical Misalignment
    DIR_VERTICAL_MISALIGNMENT_178,      # 9     -   Vertical Misalignment
    DIR_VERTICAL_MISALIGNMENT_190,      # 10    -   Vertical Misalignment
    DIR_IMBALANCE_06,                   # 11    -   Imbalance
    DIR_IMBALANCE_10,                   # 12    -   Imbalance
    DIR_IMBALANCE_15,                   # 13    -   Imbalance
    DIR_IMBALANCE_20,                   # 14    -   Imbalance
    DIR_IMBALANCE_25,                   # 15    -   Imbalance
    DIR_IMBALANCE_30,                   # 16    -   Imbalance
    DIR_IMBALANCE_35,                   # 17    -   Imbalance
    DIR_OVERHANG_BALL_00,               # 18    -   Overhang
    DIR_OVERHANG_BALL_06,               # 19    -   Overhang
    DIR_OVERHANG_BALL_20,               # 20    -   Overhang
    DIR_OVERHANG_BALL_35,               # 21    -   Overhang
    DIR_OVERHANG_CAGE_00,               # 22    -   Overhang
    DIR_OVERHANG_CAGE_06,               # 23    -   Overhang
    DIR_OVERHANG_CAGE_20,               # 24    -   Overhang
    DIR_OVERHANG_CAGE_35,               # 25    -   Overhang
    DIR_OVERHANG_OUTER_00,              # 26    -   Overhang
    DIR_OVERHANG_OUTER_06,              # 27    -   Overhang
    DIR_OVERHANG_OUTER_20,              # 28    -   Overhang
    DIR_OVERHANG_OUTER_35,              # 29    -   Overhang
    DIR_UNDERHANG_BALL_00,              # 30    -   Underhang
    DIR_UNDERHANG_BALL_06,              # 31    -   Underhang
    DIR_UNDERHANG_BALL_20,              # 32    -   Underhang
    DIR_UNDERHANG_BALL_35,              # 33    -   Underhang
    DIR_UNDERHANG_CAGE_00,              # 34    -   Underhang
    DIR_UNDERHANG_CAGE_06,              # 35    -   Underhang
    DIR_UNDERHANG_CAGE_20,              # 36    -   Underhang
    DIR_UNDERHANG_CAGE_35,              # 37    -   Underhang
    DIR_UNDERHANG_OUTER_00,             # 38    -   Underhang
    DIR_UNDERHANG_OUTER_06,             # 39    -   Underhang
    DIR_UNDERHANG_OUTER_20,             # 40    -   Underhang
    DIR_UNDERHANG_OUTER_35              # 41    -   Underhang
]


# ======================================================
# -------------------- Classes -------------------------
# ======================================================
# classes of machine faults in the dataset
CLASS_NORMAL                    = 0
CLASS_HORIZONTAL_MISALIGNMENT   = 1
CLASS_VERTICAL_MISALIGNMENT     = 2
CLASS_IMBALANCE                 = 3
CLASS_OVERHANG                  = 4
CLASS_UNDERHANG                 = 5

CLASSES = [
    CLASS_NORMAL,
    CLASS_HORIZONTAL_MISALIGNMENT,
    CLASS_VERTICAL_MISALIGNMENT,
    CLASS_IMBALANCE,
    CLASS_OVERHANG,
    CLASS_UNDERHANG
    ]

CLASS_ID_NORMAL                     = [CLASS_NORMAL]
CLASS_ID_HORIZONTAL_MISALIGNMENT    = [CLASS_HORIZONTAL_MISALIGNMENT] * len(DIR_HORIZONTAL_MISALIGNMENT_LST)
CLASS_ID_VERTICAL_MISALIGNMENT      = [CLASS_VERTICAL_MISALIGNMENT] * len(DIR_VERTICAL_MISALIGNMENT_LST)
CLASS_ID_IMBALANCE                  = [CLASS_IMBALANCE] * len(DIR_IMBALANCE_LST)
CLASS_ID_OVERHANG                   = [CLASS_OVERHANG] * len(DIR_OVERHANG_LST)
CLASS_ID_UNDERHANG                  = [CLASS_UNDERHANG] * len(DIR_UNDERGANG_LST)

CLASS_IDX = CLASS_ID_NORMAL                     + \
            CLASS_ID_HORIZONTAL_MISALIGNMENT    + \
            CLASS_ID_VERTICAL_MISALIGNMENT      + \
            CLASS_ID_IMBALANCE                  + \
            CLASS_ID_OVERHANG                   + \
            CLASS_ID_UNDERHANG


# =====================================================================
# -------------------- Feature Extraction Process  --------------------
# =====================================================================
X = []
y = []
for idx, DIR in enumerate(DIR_ALL):
    print(f"reading {DIR} ... [class {CLASS_IDX[idx]}]")
    for file in os.listdir(DIR):
        file_counter = 0
        if file.endswith(".csv"):
            # read microphone signal from sensors measurments in CSV file
            audio = pd.read_csv(f"{DIR}/{file}", header=None, usecols=[7]).values.flatten()

            # normalize audio data
            audio = normalize(audio, dbfs=NORMALIZE_DBFS)

            # resample audio data
            audio = resample(audio, input_sampling_frequency=ORIGINAL_SAMPLING_FREQUENCY, output_sampling_frequency=NEW_SAMPLING_FREQUENCY)

            # remove silence
            audio = remove_silence(audio, threshold_dbfs=SILENCE_THRESHOLD_DBFS)

            # feature extraction (with segmentation)
            features, n_components_lst = feature_extractor(audio, features=FEATURES_LST, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False, pad_mode="constant")

            # expand feature list (if n_components > 1 for some feature, only for the first file)
            if file_counter == 0:
                if sum(n_components_lst) != len(FEATURES_LST):
                    NEW_FEATURES_LST = expand_feature_lst(FEATURES_LST, n_components_lst)

            # stack features in feature matrix X
            if len(X) == 0:
                X = features
            else:
                X = np.hstack((X, features))

            # stack classes numbers in class vector y
            y = np.hstack((y, CLASS_IDX[idx] * np.ones((features.shape[1])))).astype(int)

            file_counter += 1

FEATURES_LST = NEW_FEATURES_LST
X = X.astype(np.float32).T


# ==========================================================
# -------------------- Exporting Files  --------------------
# ==========================================================
# save feature matrix, class vector and features list in .npy files
df = pd.DataFrame(X, columns=FEATURES_LST)
df["class"] = y
df["label"] = df["class"].map({ 0: "normal",
                                1: "horizontal_misalignment",
                                2: "vertical_misalignment",
                                3: "imbalance",
                                4: "overhang",
                                5: "underhang"}
                                )
df.to_csv("data.csv", index=False)