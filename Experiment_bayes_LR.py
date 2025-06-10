import utils
from Config import CONFIG

import ast
import optuna
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours

import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError

# global variables
meta_val = []
meta_train = []
meta_val_y = []
meta_train_y = []
param_dict = {}


def read_smart_csv(path):
    def smart_eval(cell):
        if pd.isna(cell):
            return cell
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return cell
    preview = pd.read_csv(path, nrows=1)
    converters = {col: smart_eval for col in preview.columns}
    return pd.read_csv(path, converters=converters)


def save_to_csv(np_data, file_path):
    for fold in range(len(np_data)):
        np_arr = np_data[fold]
        np_arr_df = pd.DataFrame(np_arr)
        np_arr_df.to_csv(f'{file_path}_{fold}.csv')


def LR_objective(trial):
    """
    Objective function for optimizing hyperparameters of an classifier using Optuna.
    This function defines the hyperparameters to be optimized,
        train an classifier on the training dataset,
        and evaluates its performance on the test dataset.
    The metric of the model is returned as the objective value to be maximized or minimized by Optuna.

    Parameters:
        trial (optuna.Trial): An Optuna trial object that is used to suggest hyperparameter values.

    Returns:
        Metric (float): The accuracy of the classifier on the test dataset,
                        which is used as the objective value for optimization.
    """
    params = {
        'penalty': 'l2',
        'class_weight': 'balanced',
        'C': trial.suggest_float('C', 1e-4, 100, log=True),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'sag', 'saga']),
        'max_iter': trial.suggest_int('max_iter', 100, 5000),
        'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
        'multi_class': trial.suggest_categorical('multi_class', ['multinomial', 'ovr']),
        'warm_start': trial.suggest_categorical('warm_start', [True, False])
    }

    total_loss = []
    global meta_val, meta_val_y, meta_train, meta_train_y, param_dict
    for i in range(CONFIG.KFOLD):
        # Read train and test dataset from global variables
        LR_val_X = meta_val[i]
        LR_val_y = meta_val_y[i]
        LR_train_X = meta_train[i]
        LR_train_y = meta_train_y[i]

        # Create and Train LR model
        model = LogisticRegression(**params)
        model.fit(LR_train_X, LR_train_y)

        # Generate prediction on validation and evaluate
        LR_pred_proba = model.predict_proba(LR_val_X)
        LR_val_y_onehot = np.eye(CONFIG.NUM_CLASS)[LR_val_y['label'].to_numpy()]
        score = utils.weighted_log_loss(LR_val_y_onehot, LR_pred_proba)
        total_loss.append(score)

    print(f'LR bayes trial {trial.number}: {np.mean(total_loss):.4f}')
    return np.mean(total_loss)


def generate_dataset(X, y):
    """
    Generates Dataset for meta learning experiment

    This function performs a two-level cross-validation: an outer custom stratified split
        to partition the data into training and validation folds, and an inner StratifiedKFold
        to train three pipelines (SMOTE oversampling, random undersampling, EditedNearestNeighbours,
        and a classifier) for XGBoost, LightGBM, and CatBoost. It aggregates out-of-fold
        probability predictions into meta-training features and averages predictions on the outer
        validation set into meta-validation features. All trained models are saved to disk.

    Parameters:
        X (pd.DataFrame): Feature matrix where each row is a sample.
                          Columns should include all features for model training.
        y (pd.DataFrame): DataFrame containing the target labels in a column named "label".

    Returns:
        None
    """
    global meta_val, meta_val_y, meta_train, meta_train_y, param_dict
    skf0 = utils.custom_stratified_split(X, y, rare_label=16, k=5)
    # Outter cross validation (cross validation of meta learner)
    for outer_fold, (train_idx, val_idx) in enumerate(skf0):
        print(f'====== Current outter Fold: {outer_fold} ======')
        outter_X_train, outter_X_val = X.iloc[train_idx], X.iloc[val_idx]
        outter_y_train, outter_y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_models = []
        lgb_models = []
        cb_models = []

        inner_meta_train = []
        inner_meta_train_y = []
        inner_meta_val = None
        # Inner cross validation (cross validation of base learners)
        inner_sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_sgkf.split(outter_X_train, outter_y_train)):
            print(f'====== Current inner Fold: {inner_fold} ======')
            inner_X_train, inner_X_val = outter_X_train.iloc[inner_train_idx], outter_X_train.iloc[inner_val_idx]
            inner_y_train, inner_y_val = outter_y_train.iloc[inner_train_idx], outter_y_train.iloc[inner_val_idx]

            # Apply column sampeling
            xgb_X_resampled_train = inner_X_train.iloc[:, CONFIG.COL_XGBOOST]
            lgbm_X_resampled_train = inner_X_train.iloc[:, CONFIG.COL_LGBM]
            cb_X_resampled_train = inner_X_train.iloc[:, CONFIG.COL_CATBOOST]

            xgb_X_resampled_val = inner_X_val.iloc[:, CONFIG.COL_XGBOOST]
            lgb_X_resampled_val = inner_X_val.iloc[:, CONFIG.COL_LGBM]
            cb_X_resampled_val = inner_X_val.iloc[:, CONFIG.COL_CATBOOST]

            # Apply row sampeling
            xgb_us_strategy = {k: 1000 for k, v in Counter(inner_y_train['label']).items() if v > 1000}
            xgb_os_strategy = {k: 400 for k, v in Counter(inner_y_train['label']).items() if v < 400}

            lgbm_us_strategy = {k: 1400 for k, v in Counter(inner_y_train['label']).items() if v > 1400}
            lgbm_os_strategy = {k: 500 for k, v in Counter(inner_y_train['label']).items() if v < 500}

            cb_us_strategy = {k: 1000 for k, v in Counter(inner_y_train['label']).items() if v > 1000}
            cb_os_strategy = {k: 400 for k, v in Counter(inner_y_train['label']).items() if v < 400}

            xgb_model = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=xgb_os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=xgb_us_strategy, random_state=CONFIG.SEED)),
                ('ENN', EditedNearestNeighbours(n_neighbors=3)),
                ('classifier', xgb.XGBClassifier(**(param_dict["XGBoost"]["bayes"]), **(CONFIG.xgb_basic_params))),
            ])

            lgbm_model = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=lgbm_os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=lgbm_us_strategy, random_state=CONFIG.SEED)),
                ('ENN', EditedNearestNeighbours(n_neighbors=3)),
                ('classifier', lgb.LGBMClassifier(**(param_dict["LGBM"]["bayes"] | CONFIG.lgb_basic_params))),
            ])

            cb_model = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=cb_os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=cb_us_strategy, random_state=CONFIG.SEED)),
                ('ENN', EditedNearestNeighbours(n_neighbors=3)),
                ('classifier', cb.CatBoostClassifier(**(CONFIG.cb_basic_params), **(param_dict["CatBoost"]["bayes"]))),
            ])

            # Fit base learners
            xgb_model.fit(xgb_X_resampled_train, inner_y_train['label'].to_numpy())
            lgbm_model.fit(lgbm_X_resampled_train, inner_y_train['label'].to_numpy())
            cb_model.fit(cb_X_resampled_train, inner_y_train['label'].to_numpy())

            # Predict validation set and store result
            inner_y_pred_xgb_val = xgb_model.predict_proba(xgb_X_resampled_val)
            inner_y_pred_lgb_val = lgbm_model.predict_proba(lgb_X_resampled_val)
            inner_y_pred_cb_val = cb_model.predict_proba(cb_X_resampled_val)

            outter_xgb_X_resampled_val = outter_X_val.iloc[:, CONFIG.COL_XGBOOST]
            outter_lgb_X_resampled_val = outter_X_val.iloc[:, CONFIG.COL_LGBM]
            outter_cb_X_resampled_val = outter_X_val.iloc[:, CONFIG.COL_CATBOOST]

            outter_y_pred_xgb_val = xgb_model.predict_proba(outter_xgb_X_resampled_val)
            outter_y_pred_lgb_val = lgbm_model.predict_proba(outter_lgb_X_resampled_val)
            outter_y_pred_cb_val = cb_model.predict_proba(outter_cb_X_resampled_val)

            inner_meta_train_temp = [inner_y_pred_xgb_val, inner_y_pred_lgb_val, inner_y_pred_cb_val]
            inner_meta_train_temp_np = np.hstack(inner_meta_train_temp)
            inner_meta_train.append(inner_meta_train_temp_np)
            inner_meta_train_y.append(inner_y_val["label"].to_numpy())

            inner_meta_val_temp = [outter_y_pred_xgb_val, outter_y_pred_lgb_val, outter_y_pred_cb_val]
            inner_meta_val_temp_np = np.hstack(inner_meta_val_temp)
            if inner_meta_val is None:
                inner_meta_val = inner_meta_val_temp_np
            else:
                inner_meta_val += inner_meta_val_temp_np

            # Store fitted models
            joblib.dump(xgb_model, f"xgb_model_{outer_fold}_fold{inner_fold}.pkl")
            joblib.dump(lgb_models, f"lgbm_model_{outer_fold}_fold{inner_fold}.pkl")
            joblib.dump(cb_models, f"cb_model_{outer_fold}_fold{inner_fold}.pkl")

            xgb_models.append(xgb_model)
            lgb_models.append(lgbm_model)
            cb_models.append(cb_model)
        inner_meta_train_np = np.vstack(inner_meta_train)
        inner_meta_train_y_np = np.hstack(inner_meta_train_y)
        meta_train.append(inner_meta_train_np)
        meta_train_y.append(inner_meta_train_y_np)
        meta_val.append(inner_meta_val / CONFIG.KFOLD)
        meta_val_y.append(outter_y_val)


if __name__ == '__main__':
    # global meta_val, meta_val_y, meta_train, meta_train_y, param_dict
    # Read bayes tuned parameters for each model to dictionary
    df = read_smart_csv(CONFIG.BEST_PARAMETER_PATH)
    for idx, row in df.iterrows():
        param_dict[row['model']] = row.drop(labels='model').to_dict()

    # Read dataset and generate meta dataset
    X, y = utils.read_data(CONFIG.TRAINING_FEATURE_PATH, CONFIG.TRAINING_LABEL_PATH)
    generate_dataset(X, y)

    # Save meta dataset
    save_to_csv(meta_train, 'meta_train')
    save_to_csv(meta_train_y, 'meta_train_y')
    save_to_csv(meta_val, 'meta_val')
    save_to_csv(meta_val_y, 'meta_val_y')

    # Start bayes tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(LR_objective, n_trials=100)

    print("Best Param:", study.best_params)
    print("Best Loss:", study.best_value)
