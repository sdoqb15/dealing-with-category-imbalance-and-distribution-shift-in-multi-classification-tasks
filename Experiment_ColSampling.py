import utils
from Config import CONFIG

import shap
import numpy as np
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours

import xgboost as xgb
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
import catboost as cb


def importance_threshold_grid_search(X, y, base_estimator='xgb'):
    """
    Evaluate model with feature selection based on feature importance percentage.

    Parameters:
      X: Feature matrix
      y: Target labels
      base_estimator: Model type to use ("xgb", "lgb", or "cb")

    Returns:
      Result: Object containing best score, best percentage, number of features, and all results
    """

    if base_estimator == 'xgb':
        model_class = xgb.XGBClassifier
        model_params = CONFIG.xgb_basic_params
    elif base_estimator == 'lgb':
        model_class = LGBMClassifier
        model_params = CONFIG.lgb_basic_params
    elif base_estimator == 'cb':
        model_class = cb.CatBoostClassifier
        model_params = CONFIG.cb_basic_params
    else:
        raise ValueError(f"Unsupported base estimator: {base_estimator}")

    importance_model = model_class(**model_params)
    importance_model.fit(X, y['label'].to_numpy())

    # Get feature importance
    if hasattr(importance_model, 'feature_importances_'):
        importances = importance_model.feature_importances_
    elif hasattr(importance_model, 'feature_importance'):
        importances = importance_model.feature_importance()
    else:
        raise ValueError("Model does not support feature importance")

    # Step 2: Sort features by importance
    indices = np.argsort(importances)[::-1]  # Indices in descending order

    # Use percentages instead of thresholds
    percentages = np.round(np.arange(0.5, 1.0 + 0.02, 0.02), 2).tolist()

    # Store results for different percentages
    percentage_results = {}

    # Perform k-fold cross-validation for each percentage
    for percentage in percentages:
        print(f"\nTesting top {percentage * 100:.0f}% features by importance")

        # Prepare k-fold cross-validation
        kf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
        fold_scores = []

        # Execute training and evaluation on each fold
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # Split into training and validation sets
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            us_strategy = {k: 1400 for k, v in Counter(y_train['label']).items() if v > 1400}
            os_strategy = {k: 500 for k, v in Counter(y_train['label']).items() if v < 500}
            sampler = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=os_strategy, random_state=CONFIG.SEED, k_neighbors=3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                ('ENN', EditedNearestNeighbours(n_neighbors=3))
            ])

            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

            # Calculate number of features to keep
            n_features = X.shape[1]
            n_features_to_keep = max(1, int(n_features * percentage))

            # Create feature mask
            feature_mask = np.zeros(n_features, dtype=bool)
            feature_mask[indices[:n_features_to_keep]] = True

            # Apply feature selection to training and validation sets
            X_train_selected = X_resampled.iloc[:, feature_mask]
            X_val_selected = X_val.iloc[:, feature_mask]

            # Step 3: Train new model on selected features
            model = model_class(**model_params)
            model.fit(X_train_selected, y_resampled['label'].to_numpy())

            # Step 4: Evaluate model performance on validation set
            y_pred_proba = model.predict_proba(X_val_selected)
            y_pred_label = np.argmax(y_pred_proba, axis=1)
            y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val["label"].to_numpy()]
            loss = utils.weighted_log_loss(y_val_onehot, y_pred_proba)
            score = loss  # Convert to score

            fold_scores.append(score)
            n_selected_features = int(X.shape[1] * percentage)
            print(f"===== Fold {fold_idx + 1}: Selected {n_selected_features} features, score: {score:.4f} =======")

            acc = accuracy_score(y_val, y_pred_label)
            bal_acc = balanced_accuracy_score(y_val, y_pred_label)
            macro_f1 = f1_score(y_val, y_pred_label, average='macro')
            weighted_f1 = f1_score(y_val, y_pred_label, average='weighted')
            print(f"Accuracy: {acc:.4f}")
            print(f"Balanced Accuracy: {bal_acc:.4f}")
            print(f"Macro F1-score: {macro_f1:.4f}")
            print(f"Weighted F1-score: {weighted_f1:.4f}")

            # Print feature importance information after first fold (only once)
            if fold_idx == 0:
                # Print top 10 most important features
                feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(n_features)]

        # Calculate average results
        mean_score = np.mean(fold_scores)

        print(f"Average results for top {percentage * 100:.0f}% features:")
        print(f"  Average cross-validation score [log loss]: {mean_score}")

        # Save results
        percentage_results[percentage] = mean_score

    # Find best percentage
    best_percentage = min(percentage_results, key=lambda p: percentage_results[p])
    best_result = percentage_results[best_percentage]

    print(f"\nBest results for feature importance percentage:")
    print(f"Best percentage: top {best_percentage * 100:.0f}%")
    print(f"Best score: {best_result}")
    print(f"Best Sample Index: {indices[:int(X.shape[1] * best_percentage)]}")

    with open(f'{base_estimator}_featureSelection_importanceBased_result.txt', 'w') as f:
        for key, value in percentage_results.items():
            f.write(f"{key}: {value}\n")


def feature_SHAP_grid_search(X, y, base_estimator='xgb'):
    """
    Evaluate model performance using SHAP with cross-validation.

    Parameters:
      X: Feature matrix
      y: Target labels
      base_estimator: Model type to use ("xgb", "lgb", or "cb")

    Returns:
      Result: Object containing best score, best percentage, number of features, and all results
    """
    if base_estimator == 'xgb':
        model_class = xgb.XGBClassifier
        model_params = CONFIG.xgb_basic_params
    elif base_estimator == 'lgb':
        model_class = LGBMClassifier
        model_params = CONFIG.lgb_basic_params
    elif base_estimator == 'cb':
        model_class = cb.CatBoostClassifier
        model_params = CONFIG.cb_basic_params
    else:
        raise ValueError(f"Unsupported base estimator: {base_estimator}")

    # Feature retention percentages
    feature_percentages = np.round(np.arange(0.1, 1.0 + 0.02, 0.02), 2).tolist()

    # Store results for different percentages
    percentage_results = {}

    full_model = model_class(**model_params)
    full_model.fit(X, y["label"].to_numpy())

    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X)

    # If multiclass, get average over classes
    if isinstance(shap_values, list):
        shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
    else:
        shap_values = np.abs(shap_values)

    shap_mean_importance = np.mean(shap_values, axis=0)
    sorted_indices = np.argsort(shap_mean_importance)[::-1]

    for percentage in feature_percentages:
        n_features_to_select = int(X.shape[1] * percentage)
        print(f"\nTesting top {percentage*100:.0f}% SHAP features ({n_features_to_select} features)")

        # Prepare k-fold cross-validation
        kf = StratifiedKFold(n_splits=CONFIG.KFOLD, shuffle=True, random_state=CONFIG.SEED)
        fold_scores = []
        fold_feature_indices = []  # Store selected feature indices for each fold

        # Execute training and evaluation on each fold
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            try:
                # Split into training and validation sets
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                if base_estimator in ["xgb", "cb"]:
                    us_strategy = {k: 1000 for k, v in Counter(y_train['label']).items() if v > 1000}
                    os_strategy = {k: 400 for k, v in Counter(y_train['label']).items() if v < 400}
                    sampler = Pipeline([
                        ('OverSampler', SMOTE(sampling_strategy=os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                        ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                        ('ENN', EditedNearestNeighbours(n_neighbors=3))
                    ])
                else:
                    pass

                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                selected_indices = sorted_indices[:n_features_to_select]

                X_train_selected, X_val_selected = X_resampled.iloc[:,selected_indices], X_val.iloc[:, selected_indices]

                # Train new model with selected features
                model = model_class(**model_params)
                model.fit(X_train_selected, y_resampled['label'].to_numpy())

                # Evaluate on validation set
                y_pred_proba = model.predict_proba(X_val_selected)
                y_pred_label = np.argmax(y_pred_proba, axis=1)
                y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val["label"].to_numpy()]
                score = utils.weighted_log_loss(y_val_onehot, y_pred_proba)

                fold_scores.append(score)
                print(f"======== Fold {fold_idx+1}: Selected {len(selected_indices)} features, score: {score:.4f} =========")

                acc = accuracy_score(y_val, y_pred_label)
                bal_acc = balanced_accuracy_score(y_val, y_pred_label)
                macro_f1 = f1_score(y_val, y_pred_label, average='macro')
                weighted_f1 = f1_score(y_val, y_pred_label, average='weighted')
                print(f"Accuracy: {acc:.4f}")
                print(f"Balanced Accuracy: {bal_acc:.4f}")
                print(f"Macro F1-score: {macro_f1:.4f}")
                print(f"Weighted F1-score: {weighted_f1:.4f}")
            except Exception as e:
                print(f"  Fold {fold_idx+1} error: {str(e)}")
                continue

        if not fold_scores:
            print(f"All folds failed for percentage {percentage}, skipping")
            continue

        # Calculate average results
        mean_score = np.mean(fold_scores)

        print(f"Average results for {percentage*100:.0f}% feature retention:")
        print(f"  Average cross-validation score [log loss]: {mean_score}")

        # Save results
        percentage_results[percentage] = mean_score

    if not percentage_results:
        raise ValueError("All SHAP attempts failed")

    # Find best percentage
    best_percentage = min(percentage_results, key=lambda p: percentage_results[p])
    best_result = percentage_results[best_percentage]

    print(f"\nBest results for SHAP:")
    print(f"Best retention percentage: {best_percentage*100:.0f}%")
    print(f"Best score: {best_result}")
    print(f"Best Sample Index: {sorted_indices[:int(X.shape[1] * best_percentage)]}")
    print('Best Features')

    for i in best_result:
        print(f'{i},')

    with open(f'{base_estimator}_featureSelection_SHAP_result.txt', 'w') as f:
        for key, value in percentage_results.items():
            f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    X, y = utils.read_data(CONFIG.TRAINING_FEATURE_PATH, CONFIG.TRAINING_LABEL_PATH)

    # Resolve experiment configurations in config to string
    model_names = ["xgb", "lgb", "cb"]
    model_full_names = ["XGBoost", "LightGBM", "CatBoost"]
    if CONFIG.EXPERIMENT_MODEL_SETTING not in [0, 1, 2]:
        raise ValueError(f"Invalid model selection: {CONFIG.EXPERIMENT_MODEL_SETTING}. Must be 0(XGBoost), 1(LGBM) or 2(CatBoost)")

    base_estimator = model_names[CONFIG.EXPERIMENT_MODEL_SETTING]
    model_full_name = model_full_names[CONFIG.EXPERIMENT_MODEL_SETTING]

    # Apply experiment
    if CONFIG.EXPERIMENT_FEATURE_SAMPLING_STRATEGY == 0:
        print(f"Executing feature selection based on GBDT feature importance, model: {base_estimator.upper()}")
        importance_threshold_grid_search(X, y, base_estimator=base_estimator)
    if CONFIG.EXPERIMENT_FEATURE_SAMPLING_STRATEGY == 1:
        print(f"Executing SHAP, model: {base_estimator.upper()}")
        feature_SHAP_grid_search(X, y, base_estimator=base_estimator)
