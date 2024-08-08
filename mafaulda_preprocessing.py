import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn import model_selection
from sklearn import preprocessing
from typing import Tuple

# ========================
# ----- data reading -----
# ========================

def data_reading(
        csv_file: str,
        features_columns: list
        ) -> pd.DataFrame:

    basic_columns = [
        "filename",
        "class",
        "label"
    ]

    selected_columns = []
    selected_columns.extend(features_columns)
    selected_columns.extend(basic_columns)

    df = pd.read_csv(csv_file, usecols=selected_columns)

    df["source"] = "original"

    return df



# =========================
# ----- data cleaning -----
# =========================

def data_cleaning(
        df: pd.DataFrame,
        verbose=False
        ) -> pd.DataFrame:

    # count the number of NaN values in each column
    total_nan = df.isna().sum().sum()

    # count the number of rows with at least one NaN value
    rows_nan = df.isna().any(axis=1).sum()

    # count the number of columns with at least one NaN value
    columns_nan = df.isna().any().sum()

    # create a copy of dataframe without columns with at least NaN
    df_without_nan = df.dropna(axis=1)

    if verbose:
        print("NaN values in DataFrame:", total_nan)
        print("Rows with at least one NaN value:", rows_nan)
        print("Columns with at least one NaN value:", columns_nan)
    
    return df_without_nan



# ============================
# ----- data preparation -----
# ============================

def data_preparation(
        df: pd.DataFrame,
        columns_to_drop: list[str],
        sampling_strategy: dict | str,
        scaling_method: str = 'RobustScaler',
        random_state: int = 42
        ) -> tuple:

    # extract the features and the class
    features = df.drop(columns=columns_to_drop)
    class_column = df["class"]

    # encode class column if necessary
    label_encoder = preprocessing.LabelEncoder()
    class_column_encoded = label_encoder.fit_transform(class_column)

    # set X and y values
    X = features.values
    y = class_column_encoded.copy()

    
    # Data Scaling
    if scaling_method == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    elif scaling_method == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
    elif scaling_method == 'MinMaxScaler':
        scaler = preprocessing.MinMaxScaler()
    elif scaling_method == 'MaxAbsScaler':
        scaler = preprocessing.MaxAbsScaler()
    elif scaling_method == 'L1':
        scaler = preprocessing.Normalizer(norm='l1')
    elif scaling_method == 'L2':
        scaler = preprocessing.Normalizer(norm='l2')
    elif scaling_method == 'max':
        scaler = preprocessing.Normalizer(norm='max')
    elif scaling_method == 'yeo-johnson':
        scaler = preprocessing.PowerTransformer(method="yeo-johnson")
    elif scaling_method == 'uniform':
        scaler = preprocessing.QuantileTransformer(output_distribution="uniform", random_state=random_state)
    elif scaling_method == 'normal':
        scaler = preprocessing.QuantileTransformer(output_distribution="normal", random_state=random_state)
    else:
        TypeError("These are the available scaling methods: 'RobustScaler', 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'L1', 'L2', 'max', 'yeo-johnson', 'uniform', 'normal'.")

    X = scaler.fit_transform(X)


    # data augmentation
    X_resampled, y_resampled = ADASYN(
        sampling_strategy=sampling_strategy,
        random_state=random_state
        ).fit_resample(X, y)
    
    n_generated = X_resampled.shape[0] - X.shape[0]

    new_df = pd.DataFrame(X_resampled[-n_generated:], columns=features.columns)
    new_df['class'] = label_encoder.inverse_transform(y_resampled[-n_generated:])
    new_df['source'] = 'generated'

    new_df.loc[new_df['class'] == 0, 'label'] = 'normal'
    new_df.loc[new_df['class'] == 1, 'label'] = 'horizontal_misalignment'
    new_df.loc[new_df['class'] == 2, 'label'] = 'vertical_misalignment'
    new_df.loc[new_df['class'] == 3, 'label'] = 'imbalance'
    new_df.loc[new_df['class'] == 4, 'label'] = 'overhang'
    new_df.loc[new_df['class'] == 5, 'label'] = 'underhang'

    original_filenames = df['filename'].values
    new_filenames = [f'generated_{i}.wav' for i in range(len(original_filenames), len(original_filenames) + len(new_df))]
    new_df['filename'] = new_filenames

    df_resampled = pd.concat([df, new_df], ignore_index=True)

    return df_resampled



# ==========================
# ----- data splitting -----
# ==========================

def data_splitting(
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
        ) -> Tuple[np.array, np.array, np.array, np.array]:
    
    unique_files = df['filename'].unique()

    train_files, test_files = model_selection.train_test_split(unique_files, test_size=test_size, random_state=random_state)

    train_df = df[df['filename'].isin(train_files)]
    test_df = df[df['filename'].isin(test_files)]

    feature_lst = [column for column in df.columns if column not in ['filename', 'class', 'label', 'source']]
    labels = list(df['label'].unique())

    X_train = train_df[feature_lst].values
    y_train = train_df['class'].values
    
    X_test = test_df[feature_lst].values
    y_test = test_df['class'].values

    return X_train, y_train, X_test, y_test
