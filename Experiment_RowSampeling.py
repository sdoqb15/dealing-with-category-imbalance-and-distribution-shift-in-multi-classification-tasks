# Basic
import csv
import numpy as np
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
import catboost as cb

import utils
from Config import CONFIG


def experiment_row_random(X, y, min_threash, max_threash):
    """
    Performs grid search to find optimal random sampling strategies for imbalanced data.

    Parameters:
      X: Feature matrix
      y: Target labels
      min_threash, max_threash: The experiment boundary of sampling value

    Returns:
      cur_min: Minimum weighted log loss
      best_feature: The feature(sampling strategy) that achieves best weighted log loss
      results: Dictionary of grid search results
    """
    cls_num = len(Counter(y['label']))  # The correct value of number of classes
    cur_min = np.inf                    # The minimum weighted log loss
    best_feature = None                 # The best sampling param
    results = []                        # The loss results
    OVERALL_MIN_THRESH = CONFIG.EXPERIMENT_OVER_THRESH[0]
    OVERALL_MAX_THRESH = CONFIG.EXPERIMENT_OVER_THRESH[1]

    # Iterate under and over sampling parameter
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for under_thresh in range(min_threash, max_threash + 1, CONFIG.EXPERIMENT_STEP):
        for over_thresh in range(OVERALL_MIN_THRESH, OVERALL_MAX_THRESH + 1, CONFIG.EXPERIMENT_STEP):
            k = 1
            cv_loss = []
            fail_flag = False
            print(f'=== Row Sampling Under threshold: {under_thresh} Over threshold:{over_thresh} ===')
            for train_idx, val_idx in sgkf.split(X, y):
                print(f'=== Fold {k} ===')
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create sampling pipeline
                rus_strategy = {k: under_thresh for k, v in Counter(y_train['label']).items() if v > under_thresh}
                ros_strategy = {k: over_thresh for k, v in Counter(y_train['label']).items() if v < over_thresh}
                sampler = Pipeline([
                    ('UnderSampler', RandomUnderSampler(sampling_strategy=rus_strategy, random_state=CONFIG.SEED)),
                    ('OverSampler', RandomOverSampler(sampling_strategy=ros_strategy, random_state=CONFIG.SEED))
                ])

                # Prevent the experiment was stopped due to extreme sampling params
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                except ValueError as e:
                    error_message = str(e)
                    print(f'Caught {error_message} at {under_thresh}, {over_thresh}')
                    fail_flag = True
                    break

                # Label vanished, which means the sampling is extreme
                cur_cls_num = len(Counter(y_resampled['label']))
                if cls_num != cur_cls_num:
                    print(f'Class vanished on parameter{cls_num} {cur_cls_num}: {under_thresh} {over_thresh}')
                    fail_flag = True
                    break

                # Create model
                if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
                    model = xgb.XGBClassifier(**CONFIG.xgb_basic_params)
                elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
                    model = LGBMClassifier(**CONFIG.lgb_basic_params)
                elif CONFIG.EXPERIMENT_MODEL_SETTING == 2:
                    model = cb.CatBoostClassifier(**CONFIG.cb_basic_params)

                # Prevent the experiment was stopped due to extreme sampling params
                try:
                    model.fit(X_resampled, y_resampled['label'].to_numpy())
                except LightGBMError as e:
                    msg = str(e)
                    print(f'Caught LGBMError on {under_thresh} {over_thresh}:\n{msg}')
                    fail_flag = True
                    break
                except Exception as e:
                    msg = str(e)
                    print(f'Caught Exception on {under_thresh} {over_thresh}:\n{msg}')
                    fail_flag = True
                    break

                # Get prediction on validation set
                y_pred_proba = model.predict_proba(X_val)
                y_pred_label = np.argmax(y_pred_proba, axis=1)
                y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val['label'].to_numpy()]

                # Evaluate performance
                loss = utils.weighted_log_loss(y_val_onehot, y_pred_proba)
                acc = accuracy_score(y_val, y_pred_label)
                bal_acc = balanced_accuracy_score(y_val, y_pred_label)
                macro_f1 = f1_score(y_val, y_pred_label, average='macro')
                weighted_f1 = f1_score(y_val, y_pred_label, average='weighted')
                print(f'Weighted Log Loss: {loss}')
                print(f"Accuracy: {acc:.4f}")
                print(f"Balanced Accuracy: {bal_acc:.4f}")
                print(f"Macro F1-score: {macro_f1:.4f}")
                print(f"Weighted F1-score: {weighted_f1:.4f}")
                k += 1
                cv_loss.append(loss)
            avg_loss = np.inf
            if not fail_flag:
                avg_loss = np.mean(cv_loss)
            if fail_flag:
                print(f'=== ({under_thresh},{over_thresh}) failed ===')
            if avg_loss < cur_min:
                cur_min = avg_loss
                best_feature = (under_thresh, over_thresh)

            print(
                f"Experiment on {(under_thresh, over_thresh)} is finished with loss {avg_loss}. Current best weight log loss: {cur_min}. Current best strateg: {best_feature}.")
            print("=" * 20)
            results.append((under_thresh, over_thresh, avg_loss))

    return cur_min, best_feature, results


def experiment_row_smoteenn(X, y, min_threash, max_threash, min_neighbours, max_neighbours):
    """
    Performs grid search to find optimal smoteenn sampling strategies for imbalanced data.

    Parameters:
      X: Feature matrix
      y: Target labels
      min_threash, max_threash: The experiment boundary of sampling value
      min_neighbours, max_neighbours: The experiment boundary of param n_neighbors of ENN

    Returns:
      cur_min: Minimum weighted log loss
      best_feature: The feature(sampling strategy) that achieves best weighted log loss
      results: Dictionary of grid search results
    """
    cls_num = len(Counter(y['label']))  # The correct value of number of classes
    cur_min = np.inf                    # The minimum weighted log loss
    best_feature = None                 # The best sampling param
    results = []                        # The loss results
    OVERALL_MIN_THRESH = CONFIG.EXPERIMENT_OVER_THRESH[0]
    OVERALL_MAX_THRESH = CONFIG.EXPERIMENT_OVER_THRESH[1]

    # Iterate under and over sampling parameter
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for under_thresh in range(min_threash, max_threash+1, CONFIG.EXPERIMENT_STEP):
        for over_thresh in range(OVERALL_MIN_THRESH, OVERALL_MAX_THRESH+1, CONFIG.EXPERIMENT_STEP):
            for nn in range(min_neighbours, max_neighbours+1):
                k = 1
                cv_loss = []
                fail_flag = False
                print(f'Row Sampling {under_thresh} {over_thresh} {nn}')
                # K-Fold validation to test models' performance
                for train_idx, val_idx in sgkf.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Create sampling pipeline
                    us_strategy = {k: under_thresh for k, v in Counter(y_train['label']).items() if v > under_thresh}
                    os_strategy = {k: over_thresh for k, v in Counter(y_train['label']).items() if v < over_thresh}
                    sampler = Pipeline([
                        ('OverSampler', SMOTE(sampling_strategy=os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                        ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                        ('ENN', EditedNearestNeighbours(n_neighbors=nn))
                    ])

                    # Prevent the experiment was stopped due to extreme sampling params
                    try:
                        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                    except ValueError as e:
                        error_message = str(e)
                        print(f'Caught {error_message} at {under_thresh}, {over_thresh}')
                        fail_flag = True
                        break

                    # Label vanished, which means the sampling is extreme
                    cur_cls_num = len(Counter(y_resampled['label']))
                    if cls_num != cur_cls_num:
                        print(f'Class vanished on parameter{cls_num} {cur_cls_num}: {under_thresh} {over_thresh}')
                        fail_flag = True
                        break

                    # Create model
                    if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
                        model = xgb.XGBClassifier(**CONFIG.xgb_basic_params)
                    elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
                        model = LGBMClassifier(**CONFIG.lgb_basic_params)
                    elif CONFIG.EXPERIMENT_MODEL_SETTING == 2:
                        model = cb.CatBoostClassifier(**CONFIG.cb_basic_params)

                    # Prevent the experiment was stopped due to extreme sampling params
                    try:
                        model.fit(X_resampled, y_resampled['label'].to_numpy())
                    except LightGBMError as e:
                        msg = str(e)
                        print(f'Caught LGBMError on {under_thresh} {over_thresh} {nn}:\n{msg}')
                        fail_flag = True
                        break
                    except Exception as e:
                        msg = str(e)
                        print(f'Caught Exception on {under_thresh} {over_thresh} {nn}:\n{msg}')
                        fail_flag = True
                        break

                    # Get prediction on validation set
                    y_pred_proba = model.predict_proba(X_val)
                    y_pred_label = np.argmax(y_pred_proba, axis=1)
                    y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val["label"].to_numpy()]

                    # Evaluate performance
                    loss = utils.weighted_log_loss(y_val_onehot, y_pred_proba)
                    acc = accuracy_score(y_val, y_pred_label)
                    bal_acc = balanced_accuracy_score(y_val, y_pred_label)
                    macro_f1 = f1_score(y_val, y_pred_label, average='macro')
                    weighted_f1 = f1_score(y_val, y_pred_label, average='weighted')
                    print(f'Weighted Log Loss: {loss}')
                    print(f"Accuracy: {acc:.4f}")
                    print(f"Balanced Accuracy: {bal_acc:.4f}")
                    print(f"Macro F1-score: {macro_f1:.4f}")
                    print(f"Weighted F1-score: {weighted_f1:.4f}")
                    cv_loss.append(loss)
                    k += 1
                avg_loss = np.inf
                if not fail_flag:
                    avg_loss = np.mean(cv_loss)
                if fail_flag:
                    print(f'=== ({under_thresh},{over_thresh},{nn}) failed ===')
                if avg_loss < cur_min:
                    cur_min = avg_loss
                    best_feature = (under_thresh, over_thresh, nn)
                print(f"Experiment on {(under_thresh, over_thresh, nn)} is finished with loss {avg_loss}. Current best weight log loss: {cur_min}. Current best strateg: {best_feature}.")
                print("=" * 20)

                results.append((under_thresh, over_thresh, nn, avg_loss))
    return cur_min, best_feature, results


def experiment_row_adatom(X, y, min_threash, max_threash):
    """
    Performs grid search to find optimal ADASYN and TomekLinks sampling strategies for imbalanced data.

    Parameters:
      X: Feature matrix
      y: Target labels
      min_threash, max_threash: Boundary of sampling parameters

    Returns:
      cur_min: Minimum weighted log loss
      best_feature: The feature(sampling strategy) that achieves best weighted log loss
      results: Dictionary of grid search results
    """
    cls_num = len(Counter(y['label']))  # The correct value of number of classes
    cur_min = np.inf                    # The min weighted log loss
    best_feature = None                 # The best parameter
    results = []                        # Loss results
    OVERALL_MIN_THRESH = CONFIG.EXPERIMENT_OVER_THRESH[0]
    OVERALL_MAX_THRESH = CONFIG.EXPERIMENT_OVER_THRESH[1]
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)

    # Iterate under and over sampling parameter
    for under_thresh in range(min_threash, max_threash+1, CONFIG.EXPERIMENT_STEP):
        for over_thresh in range(OVERALL_MIN_THRESH, OVERALL_MAX_THRESH+1, CONFIG.EXPERIMENT_STEP):
            k = 1
            cv_loss = []
            fail_flag = False
            # K-Fold validation to test models' performance
            print(f'Row Sampling {under_thresh} {over_thresh}')
            for train_idx, val_idx in sgkf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create sampling pipeline
                us_strategy = {k: under_thresh for k, v in Counter(y_train['label']).items() if v > under_thresh}
                os_strategy = {k: over_thresh for k, v in Counter(y_train['label']).items() if v < over_thresh}
                sampler = Pipeline([
                    ('OverSampler', ADASYN(sampling_strategy=os_strategy, random_state=CONFIG.SEED, n_neighbors=3)),
                    ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                    ('TOMEK', TomekLinks())
                ])

                # Prevent the experiment was stopped due to extreme sampling params
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                except ValueError as e:
                    error_message = str(e)
                    print(f'Caught {error_message} at {under_thresh}, {over_thresh}')
                    fail_flag = True
                    break

                # Label vanished, which means the sampling is extreme
                cur_cls_num = len(Counter(y_resampled['label']))
                if cls_num != cur_cls_num:
                    print(f'Class vanished on parameter{cls_num} {cur_cls_num}: {under_thresh} {over_thresh}')
                    fail_flag = True
                    break

                # Create model
                if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
                    model = xgb.XGBClassifier(**CONFIG.xgb_basic_params)
                elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
                    model = LGBMClassifier(**CONFIG.lgb_basic_params)
                elif CONFIG.EXPERIMENT_MODEL_SETTING == 2:
                    model = cb.CatBoostClassifier(**CONFIG.cb_basic_params)

                # Prevent the experiment was stopped due to extreme sampling params
                try:
                    model.fit(X_resampled, y_resampled["label"].to_numpy())
                except LightGBMError as e:
                    msg = str(e)
                    print(f'Caught LGBMError on {under_thresh} {over_thresh}:\n{msg}')
                    fail_flag = True
                    break
                except Exception as e:
                    msg = str(e)
                    print(f'Caught Exception on {under_thresh} {over_thresh}:\n{msg}')
                    fail_flag = True
                    break

                # Get prediction on validation set
                y_pred_proba = model.predict_proba(X_val)
                y_pred_label = np.argmax(y_pred_proba, axis=1)
                y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val["label"].to_numpy()]

                # Evaluate performance
                loss = utils.weighted_log_loss(y_val_onehot, y_pred_proba)
                acc = accuracy_score(y_val, y_pred_label)
                bal_acc = balanced_accuracy_score(y_val, y_pred_label)
                macro_f1 = f1_score(y_val, y_pred_label, average='macro')
                weighted_f1 = f1_score(y_val, y_pred_label, average='weighted')
                print(f'Weighted Log Loss: {loss}')
                print(f"Accuracy: {acc:.4f}")
                print(f"Balanced Accuracy: {bal_acc:.4f}")
                print(f"Macro F1-score: {macro_f1:.4f}")
                print(f"Weighted F1-score: {weighted_f1:.4f}")
                cv_loss.append(loss)
                k += 1
            avg_loss = np.inf
            if not fail_flag:
                avg_loss = np.mean(cv_loss)
            if fail_flag:
                print(f'=== ({under_thresh},{over_thresh}) failed ===')
            if avg_loss < cur_min:
                cur_min = avg_loss
                best_feature = (under_thresh, over_thresh)

            print(f"Experiment on {(under_thresh, over_thresh)} is finished with loss {avg_loss}. Current best weight log loss: {cur_min}. Current best strateg: {best_feature}.")
            print("=" * 20)

            results.append((under_thresh, over_thresh, avg_loss))
    return cur_min, best_feature, results


def non_sampling(X, y):
    """
    A comparison reference to evaluate the models performance without row sampling

    Parameters:
      X: Feature matrix
      y: Target labels

    Returns:
        cv_loss: List of cross validation losses
    """
    k = 1
    cv_loss = []    # Cross validation losses
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for train_idx, val_idx in sgkf.split(X, y):
        print(f'=== Fold {k} ===')
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Creating model variable
        if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
            model = xgb.XGBClassifier(**CONFIG.xgb_basic_params)
        elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
            model = LGBMClassifier(**CONFIG.lgb_basic_params)
        elif CONFIG.EXPERIMENT_MODEL_SETTING == 2:
            model = cb.CatBoostClassifier(**CONFIG.cb_basic_params)

        model.fit(X_train, y_train['label'].to_numpy())

        # Get prediction on validation set
        y_pred_proba = model.predict_proba(X_val)
        y_pred_label = np.argmax(y_pred_proba, axis=1)
        y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val["label"].to_numpy()]

        # Evaluate result
        loss = utils.weighted_log_loss(y_val_onehot, y_pred_proba)
        acc = accuracy_score(y_val, y_pred_label)
        bal_acc = balanced_accuracy_score(y_val, y_pred_label)
        macro_f1 = f1_score(y_val, y_pred_label, average='macro')
        weighted_f1 = f1_score(y_val, y_pred_label, average='weighted')
        print(f'Weighted Log Loss: {loss}')
        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(f"Weighted F1-score: {weighted_f1:.4f}")
        cv_loss.append(loss)
        k += 1
    print(np.mean(cv_loss))
    return cv_loss


if __name__ == '__main__':
    results = []
    best_score = None
    best_ratio = None
    method_name = ""
    model_name = ""
    thresh_start = 0
    thresh_end = 0
    # Read dataset
    X, y = utils.read_data(CONFIG.TRAINING_FEATURE_PATH, CONFIG.TRAINING_LABEL_PATH)
    if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
        model_name = "XGBoost"
    elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
        model_name = "LGBM"
    elif CONFIG.EXPERIMENT_MODEL_SETTING == 2:
        model_name = "CatBoost"

    # Define method name string
    if CONFIG.EXPERIMENT_ROW_SAMPLING_STRATEGY == 0:
        method_name = "Random"
    elif CONFIG.EXPERIMENT_ROW_SAMPLING_STRATEGY == 1:
        method_name = "SmoteENN"
    elif CONFIG.EXPERIMENT_ROW_SAMPLING_STRATEGY == 2:
        method_name = "ADATOM"

    if CONFIG.EXPERIMENT_UNDER_THRESH_IDX > 2:
        raise ValueError(f'====== {CONFIG.EXPERIMENT_UNDER_THRESH_IDX} Value ERROR ======')

    # Read parameter boundary from CONFIG
    thresh_start = CONFIG.EXPERIMENT_UNDER_THRESH[CONFIG.EXPERIMENT_UNDER_THRESH_IDX][0]
    thresh_end = CONFIG.EXPERIMENT_UNDER_THRESH[CONFIG.EXPERIMENT_UNDER_THRESH_IDX][1]

    # Apply experiment according to CONFIG setting
    print(f'=== {method_name} on {model_name} ===')
    if CONFIG.EXPERIMENT_ROW_SAMPLING_STRATEGY == 0:
        best_score, best_param, results = experiment_row_random(X, y, thresh_start, thresh_end)
    elif CONFIG.EXPERIMENT_ROW_SAMPLING_STRATEGY == 1:
        best_score, best_param, results = experiment_row_smoteenn(X, y, thresh_start, thresh_end, 3, 3)
    elif CONFIG.EXPERIMENT_ROW_SAMPLING_STRATEGY == 2:
        best_score, best_param, results = experiment_row_adatom(X, y, thresh_start, thresh_end)
    print(f'Best Score {best_score}, Best Param {best_param}')

    # Save result list to disk
    param_range = str(thresh_start) + '_' + str(thresh_end)
    file_name = method_name + '_' + model_name + '_' + param_range + '.csv'
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(results)