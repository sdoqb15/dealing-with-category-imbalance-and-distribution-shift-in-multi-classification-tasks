import utils
from Config import CONFIG

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.pipeline import Pipeline

if __name__ == '__main__':
    # Read datasets and parameters
    train, train_y = utils.read_data(CONFIG.TRAINING_FEATURE_PATH, CONFIG.TRAINING_LABEL_PATH)
    test_1, _ = utils.read_data(CONFIG.TEST_1_PATH)
    test_2, test_2_y = utils.read_data(CONFIG.TEST_2_PATH, CONFIG.TEST_2_LABEL_PATH)

    df = utils.read_smart_csv(CONFIG.BEST_PARAMETER_PATH)
    param_dict = {}
    for idx, row in df.iterrows():
        param_dict[row['model']] = row.drop(labels='model').to_dict()

    # Detect distribution shift of X on train dataset and test_1
    result_train_test_1, target_train_test_1 = utils.ks_statistical(train, test_1)
    target_train_test_1_sorted = sorted(target_train_test_1, key=lambda x: x['p_value'], reverse=True)
    utils.bar_plot(target_train_test_1_sorted)

    # Detect distribution shift of X on train dataset and test_2
    result_train_test_2, target_train_test_2 = utils.ks_statistical(train, test_2)
    target_train_test_2_sorted = sorted(target_train_test_2, key=lambda x: x['p_value'], reverse=True)
    utils.bar_plot(target_train_test_2_sorted)

    # Detect whether P(y|X) remains same or not.
    test_2_reduced = test_2.head(202)
    base_xgb = []
    base_lgbm = []
    base_cb = []
    for fold in range(1, CONFIG.KFOLD+1):
        ori_xgb_model = joblib.load(f'./models/ori_xgb_model_fold{fold}.pkl')
        ori_lgbm_model = joblib.load(f'./models/ori_lgbm_model_fold{fold}.pkl')
        ori_cb_model = joblib.load(f'./models/ori_cb_model_fold{fold}.pkl')

        y_pred_proba_xgb = ori_xgb_model.predict_proba(test_2_reduced.iloc[:, param_dict["XGBoost"]["col_sampling"]])
        y_pred_proba_lgbm = ori_lgbm_model.predict_proba(test_2_reduced.iloc[:, param_dict["LGBM"]["col_sampling"]])
        y_pred_proba_cb = ori_cb_model.predict_proba(test_2_reduced.iloc[:, param_dict["CatBoost"]["col_sampling"]])

        base_xgb.append(y_pred_proba_xgb)
        base_lgbm.append(y_pred_proba_lgbm)
        base_cb.append(y_pred_proba_cb)

    base_xgb_mean = np.mean(np.stack(base_xgb, axis=0), axis=0)
    base_lgbm_mean = np.mean(np.stack(base_lgbm, axis=0), axis=0)
    base_cb_mean = np.mean(np.stack(base_cb, axis=0), axis=0)

    base_output = np.hstack([base_xgb_mean, base_lgbm_mean, base_cb_mean])

    ori_meta_model = joblib.load(f'models/ori_meta_model.pkl')

    y_test_2_onehot = np.eye(CONFIG.NUM_CLASS)[test_2_y["label"].to_numpy()]
    y_pred_test_2 = ori_meta_model.predict(base_output)
    y_pred_proba_test_2 = ori_meta_model.predict_proba(base_output)

    loss = utils.weighted_log_loss(y_test_2_onehot, y_pred_proba_test_2)
    acc = accuracy_score(test_2_y, y_pred_test_2)
    bal_acc = balanced_accuracy_score(test_2_y, y_pred_test_2)
    macro_f1 = f1_score(test_2_y, y_pred_test_2, average='macro')
    weighted_f1 = f1_score(test_2_y, y_pred_test_2, average='weighted')
    print(f"Weighted Log Loss: {loss}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")

    # Detect Label Shift
    result_train_test_2_y, target_train_test_2_y = utils.ks_statistical(train_y, test_2_y)
    print(target_train_test_2_y)

    full_train = pd.concat([train, train_y], axis=1)
    full_test_2 = pd.concat([test_2.head(202), test_2_y], axis=1)
    labels = full_test_2['label'].unique()
    res = []
    for label in labels:
        cur_train = full_train[full_train['label'] == label]
        cur_test_2 = full_test_2[full_test_2['label'] == label]
        cur_train = cur_train.drop('label', axis = 1)
        cur_test_2 = cur_test_2.drop('label', axis = 1)
        result, target = utils.ks_statistical(cur_train, cur_test_2)
        res.append((label, target))
    for p in res:
        p_vals = []
        for i in p[1]:
            p_vals.append(i['p_value'])
        print(f'{p[0]} : {len(p[1])} {np.mean(p_vals)}')

    # ========= Experiment of using LR to predict a row data is from training set or testset =========
    # Preparing dataset
    train_df = train.copy(deep=True)
    test_2_df = test_2.copy(deep=True)
    train_df['label'] = 0
    test_2_df['label'] = 1
    tot_df = pd.concat([train_df, test_2_df], axis=0, ignore_index=True)
    count_0 = tot_df['label'].value_counts().get(0, 0)
    count_1 = tot_df['label'].value_counts().get(1, 0)
    X = tot_df.copy(deep=True)
    y = tot_df.copy(deep=True).loc[:, 'label']
    X = X.drop('label', axis=1)
    y = pd.DataFrame(y, columns=['label'])

    auc = []
    # Cross validation on dataset and record roc-auc
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for train_idx, val_idx in sgkf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sampler = RandomUnderSampler(sampling_strategy=1, random_state=CONFIG.SEED)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_resampled, y_resampled['label'].to_numpy())
        y_pred = model.predict(X_val)
        cur_auc = roc_auc_score(y_val['label'].to_numpy(), y_pred)
        auc.append(cur_auc)
    print(auc)
    print(np.mean(auc))

    # ========= Experiment of using GBDT to predict a row data is from training set or testset =========
    # Cross validation on dataset
    fold = 1
    xgb_auc = []
    lgbm_auc = []
    cb_auc = []
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for train_idx, val_idx in sgkf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sampler = RandomUnderSampler(sampling_strategy=1, random_state=CONFIG.SEED)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=CONFIG.SEED,
        )

        # LightGBM 二分类
        lgbm_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            random_state=CONFIG.SEED,
        )

        # CatBoost 二分类
        cb_model = cb.CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_state=CONFIG.SEED,
            verbose=0
        )

        xgb_model.fit(X_resampled, y_resampled)
        lgbm_model.fit(X_resampled, y_resampled)
        cb_model.fit(X_resampled, y_resampled)

        xgb_pred = xgb_model.predict_proba(X_val)
        lgbm_pred = lgbm_model.predict_proba(X_val)
        cb_pred = cb_model.predict_proba(X_val)
        y_val_onehot = np.eye(2)[y_val["label"].to_numpy()]
        cur_xgb_accuracy = roc_auc_score(y_val_onehot, xgb_pred)
        cur_lgbm_accuracy = roc_auc_score(y_val_onehot, lgbm_pred)
        cur_cb_accuracy = roc_auc_score(y_val_onehot, cb_pred)
        xgb_auc.append(cur_xgb_accuracy)
        lgbm_auc.append(cur_lgbm_accuracy)
        cb_auc.append(cur_cb_accuracy)
    print(xgb_auc)
    print(lgbm_auc)
    print(cb_auc)

    # Use statistical test to get the best learner
    utils.statistical_test(np.array([auc, xgb_auc]).T)
    utils.statistical_test(np.array([auc, lgbm_auc]).T)
    utils.statistical_test(np.array([auc, cb_auc]).T)

    # Use LR model to generate weight table of P(test) / P(train) for retraining gbdt models
    LR_models = []
    weight_table = [0.0] * train.shape[0]
    y_test_2_bin = pd.DataFrame({'label': [1] * 2020})
    sgkf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train, train_y)):
        print(f'=== Fold: {fold} ===')
        inner_X_train, inner_X_val = train.iloc[train_idx], train.iloc[val_idx]
        inner_y_train, inner_y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=CONFIG.SEED
        )
        X_train_concat = pd.concat([inner_X_train, test_2], axis=0, ignore_index=True)
        y_train_concat = pd.concat([inner_y_train, y_test_2_bin], axis=0, ignore_index=True)
        model.fit(X_train_concat, y_train_concat)
        y_pred_proba = model.predict_proba(inner_X_val)
        pred_weight = []
        for i in y_pred_proba:
            pred_weight.append(i[1] / i[0])
        for i in range(len(val_idx)):
            weight_table[val_idx[i]] = pred_weight[i]
        joblib.dump(model, f"LR_weight_model_{fold}.pkl")
        LR_models.append(model)
    weight_table_df = pd.DataFrame(weight_table, columns=['weight'])
    weight_table_df.to_csv('weight_table.csv')
