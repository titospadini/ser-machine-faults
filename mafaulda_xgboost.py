import numpy as np

from sklearn import metrics
from sklearn import model_selection

from sklearn.impute import SimpleImputer

from xgboost import DMatrix, train


# ========================
# ----- data reading -----
# ========================
def _xgboost_train(
        X_train,
        y_train,
        params,
        n_rounds=100,
        skf_splits=5,
        print_fold_metrics=True,
        print_full_metrics=True,
        return_metrics=True,
        save_model=True,
        model_filename='models/xgboost_model.json',
        random_state=42):
    # Lists to store accuracy of each fold and predictions
    accuracy_list   = []
    precision_list  = []
    recall_list     = []
    f1_list         = []
    fbeta_list      = []

    y_test_all = []
    y_pred_all = []

    # defining the cross-validation strategy
    skf = model_selection.StratifiedKFold(
        n_splits=skf_splits,
        shuffle=True,
        random_state=random_state
        )

    # Perform Stratified K-Fold Cross Validation
    fold_number = 1    

    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        # Convert the datasets into DMatrix
        dtrain = DMatrix(X_train_fold, label=y_train_fold)
        dtest = DMatrix(X_test_fold, label=y_test_fold)
        
        # Train the model
        evals = [(dtest, 'eval'), (dtrain, 'train')]
        bst = train(params, dtrain, num_boost_round=n_rounds, evals=evals, early_stopping_rounds=10, verbose_eval=False)
        
        # Make predictions
        y_pred_proba = bst.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics (accuracy, precision, recall, f1-score, fbeta, and AUC)
        accuracy    = metrics.accuracy_score(y_test_fold, y_pred)
        precision   = metrics.precision_score(y_test_fold, y_pred, average='macro')
        recall      = metrics.recall_score(y_test_fold, y_pred, average='macro')
        f1          = metrics.f1_score(y_test_fold, y_pred, average='macro')
        fbeta       = metrics.fbeta_score(y_test_fold, y_pred, beta=2.0, average='macro')

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        fbeta_list.append(fbeta)

        # Store predictions for confusion matrix
        y_test_all.extend(y_test_fold)
        y_pred_all.extend(y_pred)
        
        if print_fold_metrics:
            print(f'Fold {fold_number} --> accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, fbeta: {fbeta:.4f}')
        
        fold_number += 1
    
    if save_model:
        bst.save_model(model_filename)
        print()
        print(f'Trained model saved as {model_filename}')
    
    if print_full_metrics:
        # Calculate the mean accuracy
        mean_accuracy   = np.mean(accuracy_list)
        mean_precision  = np.mean(precision_list)
        mean_recall     = np.mean(recall_list)
        mean_f1         = np.mean(f1_list)
        mean_fbeta      = np.mean(fbeta_list)

        print()
        print(f"Accuracy:       {mean_accuracy:.4f} ± {np.std(accuracy_list):.4f}")
        print(f"Precision:      {mean_precision:.4f} ± {np.std(precision_list):.4f}")
        print(f"Recall:         {mean_recall:.4f} ± {np.std(recall_list):.4f}")
        print(f"F-1 Score:      {mean_f1:.4f} ± {np.std(f1_list):.4f}")
        print(f"F-Beta Score:   {mean_fbeta:.4f} ± {np.std(fbeta_list):.4f}")

    if return_metrics:
        return bst, y_test_all, y_pred_all, accuracy_list, precision_list, recall_list, f1_list, fbeta_list
    else:
        return bst


def xgboost_train(
        X_train,
        y_train,
        n_rounds=10,
        skf_splits=5,
        print_fold_metrics=True,
        print_full_metrics=True,
        return_metrics=True,
        save_model=True,
        model_filename='models/xgboost_model.json',
        random_state=42,
        **params
        ):
    
    params = {
        'max_depth':                5,
        'learning_rate':            0.1,
        'subsample':                0.7,
        'colsample_bytree':         0.8,
        'colsample_bylevel':        0.8,
        'objective':                'multi:softprob',
        'eval_metric':              'mlogloss',
        'random_state':             random_state,
        'tree_method':              'hist',
        'device':                   'cuda',
        'num_class':                len(np.unique(y_train))
    }

    model,          \
    y_valid_all,    \
    y_pred_all,     \
    accuracy_list,  \
    precision_list, \
    recall_list,    \
    f1_list,        \
    fbeta_list = _xgboost_train(
        X_train             = X_train,
        y_train             = y_train,
        params              = params,
        n_rounds            = n_rounds,
        skf_splits          = skf_splits,
        print_fold_metrics  = print_fold_metrics,
        print_full_metrics  = print_full_metrics,
        return_metrics      = return_metrics,
        save_model          = save_model,
        model_filename      = model_filename,
        random_state        = random_state
    )

    if return_metrics:
        return model, y_valid_all, y_pred_all, accuracy_list, precision_list, recall_list, f1_list, fbeta_list
    else:
        return model


def xgboost_predict(
        model,
        dtest
    ):
    pred_probs = model.predict(dtest)
    y_pred = np.argmax(pred_probs, axis=1)
    
    return y_pred


def xgboost_train_sweeping(
        X_train,
        y_train,
        min_rounds=100,
        max_rounds=2000,
        step_rounds=100,
        max_depth=5,
        print_fold_metrics=False,
        print_full_metrics=False,
        verbose=True,
        **params
    ):

    full_accuracy_list  = []
    full_precision_list = []
    full_recall_list    = []
    full_f1_list        = []
    full_fbeta_list     = []
    full_y_valid_all    = []
    full_y_pred_all     = []

    for n_estimators in range(min_rounds, max_rounds + step_rounds, step_rounds):
        if verbose:
            print(f'Number of estimators: {n_estimators:5} ...')

            params = {
                'max_depth': max_depth
            }

            model,          \
            y_valid_all,    \
            y_pred_all,     \
            accuracy_list,  \
            precision_list, \
            recall_list,    \
            f1_list,        \
            fbeta_list = xgboost_train(
                X_train             = X_train,
                y_train             = y_train,
                n_rounds            = n_estimators,
                print_fold_metrics  = print_fold_metrics,
                print_full_metrics  = print_full_metrics,
                return_metrics      = True,
                save_model          = False,
                params              = params
            )

            full_accuracy_list.append(accuracy_list)
            full_precision_list.append(precision_list)
            full_recall_list.append(recall_list)
            full_f1_list.append(f1_list)
            full_fbeta_list.append(fbeta_list)
            full_y_valid_all.extend(y_valid_all)
            full_y_pred_all.extend(y_pred_all)

    return  full_accuracy_list,     \
            full_precision_list,    \
            full_recall_list,       \
            full_f1_list,           \
            full_fbeta_list,        \
            full_y_valid_all,       \
            full_y_pred_all
