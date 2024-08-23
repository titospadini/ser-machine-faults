import argparse
import gc
import joblib
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cuml.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from cuml.svm import LinearSVC, SVC

from sklearn import metrics

from mafaulda_metrics import *
from mafaulda_naive_bayes import *
from mafaulda_preprocessing import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
parser.add_argument('-i', '--input_csv', type=str, default='data/mafaulda_main_classes_24khz_500ms_20ms_40-mfcc.csv',  help='Path to the dataset CSV file.')
parser.add_argument('-m', '--model', type=str, default='svm_rbf', choices=['nb_bernoulli', 'nb_gaussian', 'nb_multinomial', 'svm_linear', 'svm_poly', 'svm_rbf', 'svm_sigmoid'], help='Model to train.')
parser.add_argument('-f', '--features', type=str, default='mfcc', choices=['mfcc'], help='Features to use.')
parser.add_argument('-S', '--scaling_method', type=str, default='RobustScaler', choices=['StandardScaler', 'RobustScaler', 'MinMaxScaler'], help='Scaling method to use.')
parser.add_argument('-t', '--test_size', type=float, default=0.2, help='Size of the test set.')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed.')
parser.add_argument('-a', '--augment_samples', type=int, default=6000, help='Number of samples to augment.')
parser.add_argument('-c', '--c_param', type=float, default=1.0, help='C parameter for SVM.')
parser.add_argument('-g', '--gamma_param', type=str, default='auto', choices=['auto', 'scale'], help='Gamma parameter for SVM.')
parser.add_argument('-d', '--degree_param', type=float, default=3, help='Degree parameter for SVM.')
args = parser.parse_args()

# Configuration and constants
INPUT_CSV       = args.input_csv
MODEL           = args.model
FEATURES        = args.features
SCALING_METHOD  = args.scaling_method
TEST_SIZE       = args.test_size
SEED            = args.seed
AUGMENT_SAMPLES = args.augment_samples
C_PARAM         = args.c_param
GAMMA_PARAM     = args.gamma_param
DEGREE_PARAM    = args.degree_param

if FEATURES == 'mfcc':
    FEATURES_LIST = ['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8']

# Load the dataset
logging.info('Loading the dataset...')
df = data_reading(csv_file=INPUT_CSV, features_columns=FEATURES_LIST)

# Data preparation
logging.info('Preparing the data...')
sampling_strategy = {i: AUGMENT_SAMPLES for i in range(6)}
df = data_preparation(df=data_cleaning(df), columns_to_drop=['filename', 'class', 'label', 'source'], sampling_strategy=sampling_strategy, scaling_method=SCALING_METHOD, random_state=SEED)
X_train, y_train, X_test, y_test = data_splitting(df, test_size=TEST_SIZE, random_state=SEED)

# Model-specific name additions
model_name = MODEL
if 'svm' in MODEL:
    model_name += f'_C{C_PARAM}'
    if MODEL == 'svm_poly':
        model_name += f'_degree{DEGREE_PARAM}'
elif 'nb' in MODEL:
    if MODEL == 'nb_bernoulli':
        model_name += '_Bernoulli'
    elif MODEL == 'nb_gaussian':
        model_name += '_Gaussian'
    elif MODEL == 'nb_multinomial':
        model_name += '_Multinomial'

# Model training
logging.info('Training the model...')
skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
accuracies = []
for train_index, test_index, in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

    if MODEL == 'nb_bernoulli':
        model = BernoulliNB()
    elif MODEL == 'nb_gaussian':
        model = GaussianNB()
    elif MODEL == 'nb_multinomial':
        model = MultinomialNB()
    elif MODEL == 'svm_linear':
        model = LinearSVC(C=C_PARAM)
    elif MODEL == 'svm_poly':
        model = SVC(C=C_PARAM, kernel='poly', degree=DEGREE_PARAM, gamma=GAMMA_PARAM)
    elif MODEL == 'svm_rbf':
        model = SVC(C=C_PARAM, kernel='rbf', gamma=GAMMA_PARAM)
    elif MODEL == 'svm_sigmoid':
        model = SVC(C=C_PARAM, kernel='sigmoid', gamma=GAMMA_PARAM)
    else:
        raise ValueError('Invalid model.')

    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)
    accuracies.append(metrics.accuracy_score(y_val_fold, y_pred))

logging.info(f'Validation Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')

# Model evaluation
logging.info('Evaluating the model...')
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='macro')
recall = metrics.recall_score(y_test, y_pred, average='macro')
f1 = metrics.f1_score(y_test, y_pred, average='macro')
fbeta = metrics.fbeta_score(y_test, y_pred, average='macro', beta=2)

logging.info(f'Accuracy: {accuracy:.4f}')
logging.info(f'Precision: {precision:.4f}')
logging.info(f'Recall: {recall:.4f}')
logging.info(f'F1: {f1:.4f}')
logging.info(f'F-Beta: {fbeta:.4f}')

# Generate and save confusion matrix figure
logging.info('Generating confusion matrix...')
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGn", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')

# Save the confusion matrix figure as PDF
os.makedirs('figures', exist_ok=True)
confusion_matrix_filename = f'figures/confusion_matrix_{model_name}_{FEATURES}_{fbeta:.4f}.pdf'
plt.savefig(confusion_matrix_filename, format='pdf', dpi=300, bbox_inches='tight')
logging.info(f'Confusion matrix saved as {confusion_matrix_filename}')
plt.close()

# Save the metrics to a CSV file
logging.info('Saving the metrics...')
os.makedirs('metrics', exist_ok=True)
metrics_filename = f'metrics/{model_name}_{FEATURES}_{fbeta:.4f}.csv'
metrics_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1': [f1],
    'FBeta': [fbeta]
})
metrics_df.to_csv(metrics_filename, index=False)
logging.info(f'Metrics saved as {metrics_filename}')

# Save the model
logging.info('Saving the model...')
os.makedirs('models', exist_ok=True)
model_filename = f'models/{model_name}_{FEATURES}_{fbeta:.4f}.pkl'
joblib.dump(model, model_filename)
logging.info(f'Model saved as {model_filename}')
print()

# Clean up
del model
gc.collect()
