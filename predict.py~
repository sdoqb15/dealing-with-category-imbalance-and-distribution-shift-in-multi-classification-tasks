import joblib
import numpy as np
import pandas as pd

import utils
from Config import CONFIG

if __name__ == '__main__':
    X_test_path = 'x-ydatasets/X_train.csv'
    y_test_path = 'x-ydatasets/y_train.csv'
    X_test, y_test = utils.read_data(X_test_path, y_test_path)

    df = utils.read_smart_csv(CONFIG.BEST_PARAMETER_PATH)
    param_dict = {}
    for idx, row in df.iterrows():
        param_dict[row['model']] = row.drop(labels='model').to_dict()

    base_xgb = []
    base_lgbm = []
    base_cb = []
    for fold in range(1, CONFIG.KFOLD+1):
        xgb_model = joblib.load(f'./models/xgb_model_fold{fold}.pkl')
        lgbm_model = joblib.load(f'./models/lgbm_model_fold{fold}.pkl')
        cb_model = joblib.load(f'./models/cb_model_fold{fold}.pkl')

        y_pred_proba_xgb = xgb_model.predict_proba(X_test.iloc[:, param_dict["XGBoost"]["col_sampling"]])
        y_pred_proba_lgbm = lgbm_model.predict_proba(X_test.iloc[:, param_dict["LGBM"]["col_sampling"]])
        y_pred_proba_cb = cb_model.predict_proba(X_test.iloc[:, param_dict["CatBoost"]["col_sampling"]])

        base_xgb.append(y_pred_proba_xgb)
        base_lgbm.append(y_pred_proba_lgbm)
        base_cb.append(y_pred_proba_cb)

    base_xgb_mean = np.mean(np.stack(base_xgb, axis=0), axis=0)
    base_lgbm_mean = np.mean(np.stack(base_lgbm, axis=0), axis=0)
    base_cb_mean = np.mean(np.stack(base_cb, axis=0), axis=0)

    base_oup = np.hstack([base_xgb_mean, base_lgbm_mean, base_cb_mean])

    meta_model = joblib.load(f'models/meta_model.pkl')

    meta_pred = meta_model.predict(base_oup)
    meta_pred_proba = meta_model.predict_proba(base_oup)

    y_test_onehot = np.eye(CONFIG.NUM_CLASS)[y_test["label"].to_numpy()]
    print(utils.weighted_log_loss(meta_pred_proba, y_test_onehot))
