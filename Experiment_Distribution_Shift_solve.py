import utils
from Config import CONFIG

import joblib
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression

param_dict = {}


def ensemble_base(X_with_weight, y, X_test, X_test_2, y_test_2, X_test_2_full, method_flag=0, save_flag=False):
    """
    Generate stacking meta-features from XGBoost, LightGBM, and CatBoost base models,
    and test performance of ensemble model.

    Parameters:
        X_with_weight (Pandas.DataFrame): X of train dataset with sample weights
        y (Pandas.DataFrame): y of train dataset
        X_test (Pandas.DataFrame): X of test dataset 1
        X_test_2 (Pandas.DataFrame): X of the first 202 samples in test dataset 2
        y_test_2 (Pandas.DataFrame): y of the first 202 samples test dataset 2
        X_test_2_full (Pandas.DataFrame): X of the last 1818 samples in test dataset 2
        method_flag (int): (0 / 1) for (do not / do) deal with distribution shift
        save_flag (bool): (True / False) for (save / not save) .npy
    Returns:
        None (models, CSVs, and metrics are saved/printed).
    """
    meta_train = []
    meta_train_y = []

    meta_val_xgb = []
    meta_val_lgbm = []
    meta_val_cb = []

    meta_test_xgb = []
    meta_test_lgbm = []
    meta_test_cb = []

    meta_test_full_xgb = []
    meta_test_full_lgbm = []
    meta_test_full_cb = []

    xgb_models = []
    lgbm_models = []
    cb_models = []

    train_counts = Counter(y['label'].to_numpy())
    total_train = sum(train_counts.values())
    p_train_y = np.array([train_counts.get(i, 0) / total_train for i in range(CONFIG.NUM_CLASS)])

    test_counts = Counter(y_test_2['label'].to_numpy())
    for i in range(CONFIG.NUM_CLASS):
        if i not in test_counts:
            test_counts[i] = 1.0
    total_test = sum(test_counts.values())
    p_test_y = np.array([test_counts.get(i, 0) / total_test for i in range(CONFIG.NUM_CLASS)])

    p_weight = p_test_y / p_train_y

    skf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_with_weight, y), 1):
        print(f'======= Current Fold: {fold} =======')
        X_train, X_val = X_with_weight.iloc[train_idx], X_with_weight.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        meta_train_temp = []
        meta_train_y.append(y_val["label"].to_numpy())

        # Create xgb model and sampler
        xgb_model = xgb.XGBClassifier(**(param_dict["XGBoost"]["bayes"]), **(CONFIG.xgb_basic_params))

        xgb_us_strategy = {k: param_dict["XGBoost"]["row_sampling"][0] for k, v in Counter(y_train['label']).items() if
                           v > param_dict["XGBoost"]["row_sampling"][0]}
        xgb_os_strategy = {k: param_dict["XGBoost"]["row_sampling"][1] for k, v in Counter(y_train['label']).items() if
                           v < param_dict["XGBoost"]["row_sampling"][1]}
        xgb_sampler = Pipeline([
            ('OverSampler', SMOTE(sampling_strategy=xgb_os_strategy, random_state=CONFIG.SEED, k_neighbors=3)),
            ('UnderSampler', RandomUnderSampler(sampling_strategy=xgb_us_strategy, random_state=CONFIG.SEED)),
            ('ENN', EditedNearestNeighbours(n_neighbors=3))
        ])

        # Apply row and column sampling
        xgb_X_resampled, xgb_y_resampled = xgb_sampler.fit_resample(X_train, y_train)
        xgb_X_weights = xgb_X_resampled['weight'].to_numpy()
        xgb_X_resampled = xgb_X_resampled.iloc[:, param_dict["XGBoost"]["col_sampling"]]

        # Fit model
        if method_flag == 0:
            xgb_model.fit(xgb_X_resampled, xgb_y_resampled['label'].to_numpy())
        elif method_flag == 1:
            # Apply sampel weight [covariate shift]
            xgb_model.fit(xgb_X_resampled, xgb_y_resampled['label'].to_numpy(), sample_weight=xgb_X_weights)

        xgb_models.append(xgb_model)
        # joblib.dump(xgb_model, f"xgb_model_fold{fold}.pkl")

        y_pred_xgb = xgb_model.predict_proba(X_val.iloc[:, param_dict["XGBoost"]["col_sampling"]])
        y_val_pred_xgb = xgb_model.predict_proba(X_test.iloc[:, param_dict["XGBoost"]["col_sampling"]])
        y_test_pred_xgb = xgb_model.predict_proba(X_test_2.iloc[:, param_dict["XGBoost"]["col_sampling"]])
        y_test_pred_full_xgb = xgb_model.predict_proba(X_test_2_full.iloc[:, param_dict["XGBoost"]["col_sampling"]])

        # Label weight [Label Shift]
        if method_flag == 1:
            y_test_pred_xgb = p_weight * y_test_pred_xgb
            y_test_pred_xgb_sum = y_test_pred_xgb.sum(axis=1, keepdims=True)
            y_test_pred_xgb_sum = np.maximum(y_test_pred_xgb_sum, 1e-8)
            y_test_pred_xgb = y_test_pred_xgb / y_test_pred_xgb_sum

            y_val_pred_xgb = p_weight * y_val_pred_xgb
            y_val_pred_xgb_sum = y_val_pred_xgb.sum(axis=1, keepdims=True)
            y_val_pred_xgb_sum = np.maximum(y_val_pred_xgb_sum, 1e-8)
            y_val_pred_xgb = y_val_pred_xgb / y_val_pred_xgb_sum

            y_test_pred_full_xgb = p_weight * y_test_pred_full_xgb
            y_test_pred_full_xgb_sum = y_test_pred_full_xgb.sum(axis=1, keepdims=True)
            y_test_pred_full_xgb_sum = np.maximum(y_test_pred_full_xgb_sum, 1e-8)
            y_test_pred_full_xgb = y_test_pred_full_xgb / y_test_pred_full_xgb_sum

        meta_train_temp.append(y_pred_xgb)
        meta_val_xgb.append(y_val_pred_xgb)
        meta_test_xgb.append(y_test_pred_xgb)
        meta_test_full_xgb.append(y_test_pred_full_xgb)

        # Create LGBM model and sampler
        lgbm_model = lgb.LGBMClassifier(**(param_dict["LGBM"]["bayes"] | CONFIG.lgb_basic_params))

        lgbm_us_strategy = {k: param_dict["LGBM"]["row_sampling"][0] for k, v in Counter(y_train['label']).items() if
                            v > param_dict["LGBM"]["row_sampling"][0]}
        lgbm_os_strategy = {k: param_dict["LGBM"]["row_sampling"][1] for k, v in Counter(y_train['label']).items() if
                            v < param_dict["LGBM"]["row_sampling"][1]}
        lgbm_sampler = Pipeline([
            ('OverSampler', SMOTE(sampling_strategy=lgbm_os_strategy, random_state=CONFIG.SEED, k_neighbors=3)),
            ('UnderSampler', RandomUnderSampler(sampling_strategy=lgbm_us_strategy, random_state=CONFIG.SEED)),
            ('ENN', EditedNearestNeighbours(n_neighbors=3))
        ])

        # Apply row and column sampling
        lgbm_X_resampled, lgbm_y_resampled = lgbm_sampler.fit_resample(X_train, y_train)
        lgbm_X_weights = lgbm_X_resampled['weight'].to_numpy()
        lgbm_X_resampled = lgbm_X_resampled.iloc[:, param_dict["LGBM"]["col_sampling"]]

        # Fit model
        if method_flag == 0:
            lgbm_model.fit(lgbm_X_resampled, lgbm_y_resampled['label'].to_numpy())
        if method_flag == 1:
            # Apply sample weight [covariate shift]
            lgbm_model.fit(lgbm_X_resampled, lgbm_y_resampled['label'].to_numpy(), sample_weight=lgbm_X_weights)

        lgbm_models.append(lgbm_model)
        # joblib.dump(lgbm_model, f"lgbm_model_fold{fold}.pkl")

        y_pred_lgbm = lgbm_model.predict_proba(X_val.iloc[:, param_dict["LGBM"]["col_sampling"]])
        y_val_pred_lgbm = lgbm_model.predict_proba(X_test.iloc[:, param_dict["LGBM"]["col_sampling"]])
        y_test_pred_lgbm = lgbm_model.predict_proba(X_test_2.iloc[:, param_dict["LGBM"]["col_sampling"]])
        y_test_pred_full_lgbm = lgbm_model.predict_proba(X_test_2_full.iloc[:, param_dict["LGBM"]["col_sampling"]])

        # Label weight [label shift]
        if method_flag == 1:
            y_test_pred_lgbm = p_weight * y_test_pred_lgbm
            y_test_pred_lgbm_sum = y_test_pred_lgbm.sum(axis=1, keepdims=True)
            y_test_pred_lgbm_sum = np.maximum(y_test_pred_lgbm_sum, 1e-8)
            y_test_pred_lgbm = y_test_pred_lgbm / y_test_pred_lgbm_sum

            y_val_pred_lgbm = p_weight * y_val_pred_lgbm
            y_val_pred_lgbm_sum = y_val_pred_lgbm.sum(axis=1, keepdims=True)
            y_val_pred_lgbm_sum = np.maximum(y_val_pred_lgbm_sum, 1e-8)
            y_val_pred_lgbm = y_val_pred_lgbm / y_val_pred_lgbm_sum

            y_test_pred_full_lgbm = p_weight * y_test_pred_full_lgbm
            y_test_pred_full_lgbm_sum = y_test_pred_full_lgbm.sum(axis=1, keepdims=True)
            y_test_pred_full_lgbm_sum = np.maximum(y_test_pred_full_lgbm_sum, 1e-8)
            y_test_pred_full_lgbm = y_test_pred_full_lgbm / y_test_pred_full_lgbm_sum

        meta_val_lgbm.append(y_val_pred_lgbm)
        meta_test_lgbm.append(y_test_pred_lgbm)
        meta_train_temp.append(y_pred_lgbm)
        meta_test_full_lgbm.append(y_test_pred_full_lgbm)

        # Create cb model and sampler
        cb_model = cb.CatBoostClassifier(**(param_dict["CatBoost"]["bayes"]), **(CONFIG.cb_basic_params))

        cb_us_strategy = {k: param_dict["CatBoost"]["row_sampling"][0] for k, v in Counter(y_train['label']).items() if
                          v > param_dict["CatBoost"]["row_sampling"][0]}
        cb_os_strategy = {k: param_dict["CatBoost"]["row_sampling"][1] for k, v in Counter(y_train['label']).items() if
                          v < param_dict["CatBoost"]["row_sampling"][1]}
        cb_sampler = Pipeline([
            ('OverSampler', SMOTE(sampling_strategy=cb_os_strategy, random_state=CONFIG.SEED, k_neighbors=3)),
            ('UnderSampler', RandomUnderSampler(sampling_strategy=cb_us_strategy, random_state=CONFIG.SEED)),
            ('ENN', EditedNearestNeighbours(n_neighbors=3))
        ])

        # Apply row and column sampling
        cb_X_resampled, cb_y_resampled = cb_sampler.fit_resample(X_train, y_train)
        cb_X_weights = cb_X_resampled['weight'].to_numpy()
        cb_X_resampled = cb_X_resampled.iloc[:, param_dict["CatBoost"]["col_sampling"]]

        # Fit model
        if method_flag == 0:
            cb_model.fit(cb_X_resampled, cb_y_resampled['label'].to_numpy())
        if method_flag == 1:
            # Apply sample weight [covariate shift]
            cb_model.fit(cb_X_resampled, cb_y_resampled['label'].to_numpy(), sample_weight=cb_X_weights)

        cb_models.append(cb_model)
        # joblib.dump(cb_model, f"cb_model_fold{fold}.pkl")

        y_pred_cb = cb_model.predict_proba(X_val.iloc[:, param_dict["CatBoost"]["col_sampling"]])
        y_val_pred_cb = cb_model.predict_proba(X_test.iloc[:, param_dict["CatBoost"]["col_sampling"]])
        y_test_pred_cb = cb_model.predict_proba(X_test_2.iloc[:, param_dict["CatBoost"]["col_sampling"]])
        y_test_pred_full_cb = cb_model.predict_proba(X_test_2_full.iloc[:, param_dict["CatBoost"]["col_sampling"]])

        # Label weight [label shift]
        if method_flag == 1:
            y_test_pred_cb = p_weight * y_test_pred_cb
            y_test_pred_cb_sum = y_test_pred_cb.sum(axis=1, keepdims=True)
            y_test_pred_cb_sum = np.maximum(y_test_pred_cb_sum, 1e-8)
            y_test_pred_cb = y_test_pred_cb / y_test_pred_cb_sum

            y_val_pred_cb = p_weight * y_val_pred_cb
            y_val_pred_cb_sum = y_val_pred_cb.sum(axis=1, keepdims=True)
            y_val_pred_cb_sum = np.maximum(y_val_pred_cb_sum, 1e-8)
            y_val_pred_cb = y_val_pred_cb / y_val_pred_cb_sum

            y_test_pred_full_cb = p_weight * y_test_pred_full_cb
            y_test_pred_full_cb_sum = y_test_pred_full_cb.sum(axis=1, keepdims=True)
            y_test_pred_full_cb_sum = np.maximum(y_test_pred_full_cb_sum, 1e-8)
            y_test_pred_full_cb = y_test_pred_full_cb / y_test_pred_full_cb_sum

        meta_val_cb.append(y_val_pred_cb)
        meta_test_cb.append(y_test_pred_cb)
        meta_train_temp.append(y_pred_cb)
        meta_test_full_cb.append(y_test_pred_full_cb)

        meta_train_temp_np = np.hstack(meta_train_temp)
        meta_train.append(meta_train_temp_np)
    # Concatenate train / test dataset for LR model
    meta_train_np = np.vstack(meta_train)
    meta_train_y_np = np.hstack(meta_train_y)

    meta_val_xgb_mean = np.mean(np.stack(meta_val_xgb, axis=0), axis=0)
    meta_val_lgbm_mean = np.mean(np.stack(meta_val_lgbm, axis=0), axis=0)
    meta_val_cb_mean = np.mean(np.stack(meta_val_cb, axis=0), axis=0)
    meta_val_np = np.hstack([meta_val_xgb_mean, meta_val_lgbm_mean, meta_val_cb_mean])

    meta_test_full_xgb_mean = np.mean(np.stack(meta_test_full_xgb, axis=0), axis=0)
    meta_test_full_lgbm_mean = np.mean(np.stack(meta_test_full_lgbm, axis=0), axis=0)
    meta_test_full_cb_mean = np.mean(np.stack(meta_test_full_cb, axis=0), axis=0)
    meta_test_full_np = np.hstack([meta_test_full_xgb_mean, meta_test_full_lgbm_mean, meta_test_full_cb_mean])

    meta_test_xgb_mean = np.mean(np.stack(meta_test_xgb, axis=0), axis=0)
    meta_test_lgbm_mean = np.mean(np.stack(meta_test_lgbm, axis=0), axis=0)
    meta_test_cb_mean = np.mean(np.stack(meta_test_cb, axis=0), axis=0)
    meta_test_np = np.hstack([meta_test_xgb_mean, meta_test_lgbm_mean, meta_test_cb_mean])

    # Create LogisticRegression model as meta model
    meta_model = LogisticRegression(
        penalty='l2',
        C=0.8958736258514433,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=2813,
        multi_class='multinomial',
        tol=1.5803457205743786e-05,
        warm_start=False
    )

    # Fit metamodel
    meta_model.fit(meta_train_np, meta_train_y_np)
    # joblib.dump(meta_model, "meta_model.pkl")

    # Predict and save final results
    y_val_pred_proba = meta_model.predict_proba(meta_val_np)
    df_val_pred = pd.DataFrame(y_val_pred_proba)
    df_val_pred.to_csv(f"y_pred_proba_{method_flag}.csv", index=False)

    y_test_onehot = np.eye(CONFIG.NUM_CLASS)[y_test_2['label'].to_numpy()]
    y_test_pred_label = meta_model.predict(meta_test_np)
    y_test_pred_proba = meta_model.predict_proba(meta_test_np)
    y_test_full_pred_proba = meta_model.predict_proba(meta_test_full_np)
    loss = utils.weighted_log_loss(y_test_onehot, y_test_pred_proba)

    if save_flag:
        np.save('preds_2.npy', y_test_full_pred_proba)

    acc = accuracy_score(y_test_2, y_test_pred_label)
    bal_acc = balanced_accuracy_score(y_test_2, y_test_pred_label)
    macro_f1 = f1_score(y_test_2, y_test_pred_label, average='macro')
    weighted_f1 = f1_score(y_test_2, y_test_pred_label, average='weighted')
    print(f'weighted log loss on method {method_flag}: {loss}')
    # print(f"Weighted Log Loss: {loss}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print(f'Pred proba {y_test_pred_proba}')


if __name__ == '__main__':
    df = utils.read_smart_csv(CONFIG.BEST_PARAMETER_PATH)

    param_dict = {}

    for idx, row in df.iterrows():
        param_dict[row['model']] = row.drop(labels='model').to_dict()

    # Read datasets from (private) kaggle input
    X, y = utils.read_data(CONFIG.TRAINING_FEATURE_PATH, CONFIG.TRAINING_LABEL_PATH)
    X_test, _ = utils.read_data(CONFIG.TEST_FEATURE_PATH, None)
    X_test_2_full, y_test_2 = utils.read_data(CONFIG.TEST_SHIFTED_PATH, CONFIG.TEST_SHIFTED_LABEL_PATH)

    # Split the first 202 and the last 1818 samples from test 2
    X_test_2 = X_test_2_full.head(202)
    X_test_2_full = X_test_2_full.tail(len(X_test_2_full) - 202).copy()

    # Read weight table from (private) kaggle (private) input
    weight_table, _ = utils.read_data(CONFIG.WEIGHT_TABLE_PATH)
    if weight_table.shape[1] > 1:
        redun_col = weight_table.columns[0]
        weight_table = weight_table.drop(redun_col, axis=1)

    X_with_weight = pd.concat([X, X_test_2], axis=0, ignore_index=True)
    X_with_weight = pd.concat([X_with_weight, weight_table], axis=1)
    y = pd.concat([y, y_test_2], axis=0, ignore_index=True)

    ensemble_base(X_with_weight, y, X_test, X_test_2, y_test_2, X_test_2_full, 1, True)
