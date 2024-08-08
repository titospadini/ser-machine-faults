import os
import argparse

import pandas as pd

from utils import *

# ======================================================
# -------------------- Parameters  --------------------
# ======================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    required=True,
    default="/home/tito/datasets/MAFAULDA",
    help="directory containing the audio files"
    )

parser.add_argument(
    "-o",
    "--output_dir",
    type=str, required=False,
    default="data/MAFAULDA",
    help="output directory"
    )

parser.add_argument(
    "-n",
    "--normalize_dbfs",
    type=int,
    required=False,
    default=-6,
    help="dBFS level to normalize the signal"
    )

parser.add_argument(
    "-f",
    "--frequency",
    type=int,
    required=True,
    default=48000,
    help="new sampling frequency to resample the signal"
    )

parser.add_argument(
    "-c",
    "--codec",
    type=str,
    required=True,
    default="pcm_24",
    help="audio codec"
    )

args = parser.parse_args()

INPUT_DIR           = args.input_dir
OUTPUT_DIR          = args.output_dir
SAMPLING_FREQUENCY  = int(args.frequency)
NORMALIZE_DBFS      = int(args.normalize_dbfs) if args.normalize_dbfs is not None else None
CODEC               = args.codec


# ======================================================
# -------------------- Directories  --------------------
# ======================================================

# base directories for each class
DIR_NORMAL                  = f"{INPUT_DIR}/normal"
DIR_HORIZONTAL_MISALIGNMENT = f"{INPUT_DIR}/horizontal-misalignment"
DIR_VERTICAL_MISALIGNMENT   = f"{INPUT_DIR}/vertical-misalignment"
DIR_IMBALANCE               = f"{INPUT_DIR}/imbalance"
DIR_OVERHANG                = f"{INPUT_DIR}/overhang"
DIR_UNDERHANG               = f"{INPUT_DIR}/underhang"


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
    DIR_NORMAL,
    DIR_HORIZONTAL_MISALIGNMENT_05,
    DIR_HORIZONTAL_MISALIGNMENT_10,
    DIR_HORIZONTAL_MISALIGNMENT_15,
    DIR_HORIZONTAL_MISALIGNMENT_20,
    DIR_VERTICAL_MISALIGNMENT_051,
    DIR_VERTICAL_MISALIGNMENT_063,
    DIR_VERTICAL_MISALIGNMENT_127,
    DIR_VERTICAL_MISALIGNMENT_140,
    DIR_VERTICAL_MISALIGNMENT_178,
    DIR_VERTICAL_MISALIGNMENT_190,
    DIR_IMBALANCE_06,
    DIR_IMBALANCE_10,
    DIR_IMBALANCE_15,
    DIR_IMBALANCE_20,
    DIR_IMBALANCE_25,
    DIR_IMBALANCE_30,
    DIR_IMBALANCE_35,
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
    DIR_OVERHANG_OUTER_35,
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


# =====================================================
# -------------------- Processing  --------------------
# =====================================================
for DIR in DIR_ALL:
    print(f"reading {DIR} ...")
    for file in sorted(os.listdir(DIR)):
        file_counter = 0
        if file.endswith(".csv"):
            print(f"{DIR}/{file} ...")

            # read microphone signal from sensors measurments in CSV file
            audio = pd.read_csv(f"{DIR}/{file}", header=None, usecols=[7]).values.flatten()

            # normalize audio data
            if NORMALIZE_DBFS is not None:
                audio = normalize(audio, dbfs=NORMALIZE_DBFS)

            # write audio signal in WAV file
            sf.write(
                file=f"{DIR.replace("datasets", "ser-machine-faults/data")}/{file.replace('.csv', '.wav')}",
                data=audio,
                samplerate=SAMPLING_FREQUENCY,
                format="wav",
                subtype=CODEC
                )