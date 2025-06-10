import ast
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import kstest                  # Kolmogorov-Smirnov Test
from scipy.stats import wilcoxon                 # Wilcoxon Signed-Rank Test
from scipy.stats import friedmanchisquare            # Friedman Test
from scipy.stats import ttest_rel                # Paired t-test
from scipy.stats import f_oneway                 # Analysis of Variance
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp

from Config import CONFIG


def set_seed(seed=42):
    """
    Ensure the reproducibility and controllability of the randomness involved in the program.

    Parameters:
        seed (int, optional): The random number seed used in this project.

    Returns:
        None
    """
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy


def weighted_log_loss(y_true, y_pred, **kwargs):
    """
    This function is from the project discription with some modification to protect dividing by 0. It
    will compute the weighted cross-entropy(log loss) given true labels and predicted probabilities

    Parameters:
      y_true: One-hot encoded true labels
      y_pred: Predicted probabilities

    Returns :
      loss: Weighted log loss
    """

    # Protect divided by 0
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Compute class frequencies
    class_counts = np.sum(y_true, axis=0)  # Sum over samples to get counts per class
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights /= np.sum(class_weights)  # Normalize weights to sum to 1

    # Compute weighted loss
    sample_weights = np.sum(y_true * class_weights, axis=1)  # Get weight for each sample
    loss = -1 * np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))

    return loss


def evaluate_function(X, y, params, n_splits = 5, random_state = None):
    """
    Evaluate the weighted cross-entropy loss of a machine learning model via Stratified K-Fold cross-validation. Here, we use
    StratifiedKFold in order to guarantee that the class distribution in each fold is similar to it in the overall dataset

    Parameters:
      X: Features of dataset
      y: Corresponding labels
      params: Parameters for a sklear compatible estimator
      n_splits: Number of folds for cross validation
      random_state: Random seed

    Return:
      mean_scores: The mean value of the scores across all folds.
      val_scores: List of scores for each fold
    """

    sgkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_scores = []

    if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
        X_resampled = X.iloc[:, CONFIG.COL_XGBOOST]
    elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
        X_resampled = X.iloc[:, CONFIG.COL_LGBM]
    else:
        X_resampled = X.iloc[:, CONFIG.COL_CATBOOST]


    for train_idx, val_idx in sgkf.split(X_resampled, y):
        X_train, X_val = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
            us_strategy = {k: 1000 for k, v in Counter(y_train['label']).items() if v > 1000}
            os_strategy = {k: 400 for k, v in Counter(y_train['label']).items() if v < 400}

            model = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                ('classifier', xgb.XGBClassifier(**params)),
            ])

        elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
            us_strategy = {k: 1400 for k, v in Counter(y_train['label']).items() if v > 1400}
            os_strategy = {k: 500 for k, v in Counter(y_train['label']).items() if v < 500}

            model = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                ('classifier', lgb.LGBMClassifier(**params)),
            ])

        else:
            us_strategy = {k: 1000 for k, v in Counter(y_train['label']).items() if v > 1000}
            os_strategy = {k: 400 for k, v in Counter(y_train['label']).items() if v < 400}

            model = Pipeline([
                ('OverSampler', SMOTE(sampling_strategy=os_strategy, random_state=CONFIG.SEED, k_neighbors = 3)),
                ('UnderSampler', RandomUnderSampler(sampling_strategy=us_strategy, random_state=CONFIG.SEED)),
                ('classifier', cb.CatBoostClassifier(**params)),
            ])

        model.fit(X_train, y_train['label'].to_numpy())
        y_pred_proba = model.predict_proba(X_val)

        y_val_onehot = np.eye(CONFIG.NUM_CLASS)[y_val['label'].to_numpy()]

        score = weighted_log_loss(y_val_onehot, y_pred_proba)
        val_scores.append(score)

    mean_scores = np.mean(val_scores)

    return mean_scores, val_scores


def custom_stratified_split(X, y, rare_label=16, k=5, seed=42):
    """
    Creates a custom stratified k-fold split that ensures each fold’s validation set
    contains exactly one instance of a specified rare label, while stratifying on all
    other labels.

    This function first separates indices of the rare class from all other samples.
    It then applies StratifiedKFold to the “other” samples to generate k folds.
    For each fold, exactly one rare‐label index is assigned to the validation set
    (rotating through the rare indices), and the remaining rare indices join the
    training set. The other samples follow the standard stratified split.

    Parameters:
        X (pd.DataFrame):      Feature matrix with samples as rows.
        y (pd.DataFrame):      DataFrame containing a "label" column of class labels.
        rare_label (int, optional, default=16):
                               The label value treated as “rare” (one per fold in validation).
        k (int, optional, default=5):
                               Number of folds to generate.
        seed (int, optional, default=42):
                               Random seed for shuffling in StratifiedKFold.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of length k, where each element
        is a tuple (train_idx, val_idx). `train_idx` and `val_idx` are numpy arrays
        of integer indices for the training and validation splits, respectively.
    """
    rare_indices = np.where(y['label'] == rare_label)[0]

    other_indices = np.where(y['label'] != rare_label)[0]

    skf_other = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    other_folds = list(skf_other.split(X.iloc[other_indices], y.iloc[other_indices]['label']))

    folds = []
    for i in range(k):
        rare_val_idx = [rare_indices[i % 6]]
        rare_train_idx = [ix for ix in rare_indices if ix not in rare_val_idx]

        other_train_idx, other_val_idx = other_folds[i]
        train_idx = np.concatenate([rare_train_idx, other_indices[other_train_idx]])
        val_idx = np.concatenate([rare_val_idx, other_indices[other_val_idx]])

        folds.append((train_idx, val_idx))

    return folds


def box_plot(data, title):
    """Draw a box polt based on the given data"""

    plt.boxplot(data, labels=[f'Group {i + 1}' for i in range(data.shape[1])])
    plt.title(title)
    plt.ylabel('Scores')
    plt.show()


def statistical_test(scores):
    """
    This function will first determine whether or not the data follows a normal distribution and then conduct different
    non-parametric tests based on the number of groups. Scores are the we obtained using k-fold cross validation on n models. The j-th column
    represents the score of the j-th model and the i-th row represents the score at the i-th fold
    Parameters:
        scores: A p-k array containing p k-fold cross validation scores
    """

    # Get k and n
    k, n = scores.shape
    normality = True

    # Kolmogorov-Smirnov test
    print("Kolmogorov-Smirnov Test starts: ")
    for i in range(n):
        stat, p_value_normal = kstest(scores[:, i], cdf="norm")
        print(f"    Group {i + 1}: Kolmogorov-Smirnov test p-value = {p_value_normal:.4f}")

        if p_value_normal < 0.05:
            normality = False

    if not normality:
        # Wilcoxon Signed-Rank Test
        if n == 2:
            print("Wilcoxon Signed-Rank Test starts: ")
            stat, p_value_wilcoxon = wilcoxon(scores[:, 0], scores[:, 1])
            print(f"    Wilcoxon test p-value = {p_value_wilcoxon:.4f}")
            box_plot(scores, 'Wilcoxon Test Results')

        # Friedman Test
        elif n > 2:
            print("Friedman Test starts: ")
            stat, p_value_friedman = friedmanchisquare(*scores.T)
            print(f"    Friedman test p-value = {p_value_friedman:.4f}")
            box_plot(scores, 'Friedman Test Results')

            # Nemenyi Post-Hoc Test
            if p_value_friedman < 0.05:
                print("Reject the null hypothesis for Friedman test. Performing Nemenyi post-hoc test.")
                p_value_nemenyi = posthoc_nemenyi_friedman(scores)
                print(f"    Nemenyi post-hoc test p-values:\n{p_value_nemenyi}")
    else:
        # Paired t-test
        if n == 2:
            print("Paired t-test starts: ")
            stat, p_value_ttest = ttest_rel(scores[:, 0], scores[:, 1])
            print(f"    Paired t-test p-value = {p_value_ttest:.4f}")
            box_plot(scores, 'Paired t-test Results')

        # Analysis of Variance
        elif n > 2:
            print("ANOVA Test starts: ")
            stat, p_value_anova = f_oneway(*scores.T)
            print(f"    ANOVA test p-value = {p_value_anova:.4f}")
            box_plot(scores, 'ANOVA Test Results')

            # Tukey HSD Post-Hoc Test
            if p_value_anova < 0.05:
                print("Reject the null hypothesis for ANOVA. Performing Tukey HSD post-hoc test.")

                scores_flat = scores.ravel()
                group_labels = np.repeat(np.arange(n), k)

                p_value_tukey = pairwise_tukeyhsd(endog=scores_flat, groups=group_labels, alpha=0.05)
                print(f"    Tukey HSD post-hoc test p-values:\n{p_value_tukey}")


def ks_statistical(x, y):
    """
    Computes the Kolmogorov–Smirnov statistic for each feature between two datasets.
    This function iterates over every column in x and y, applies scipy.stats.ks_2samp
    to compare their distributions, and records the KS statistic and p-value for each feature.
    Features whose p-value falls below the configured threshold are flagged as significant.

    Parameters:
        x (pd.DataFrame): Reference dataset with feature columns.
        y (pd.DataFrame): Comparison dataset with the same feature columns as x.

    Returns:
        results (list of dict): All features with keys 'feature_name', 'ks_statistic', and 'p_value'.
        targets (list of dict): Subset of results where p_value < CONFIG.KS_THRESHOLD.
    """
    results = []
    targets = []
    for feature in x.columns:
        x_feat = x.loc[:, feature]
        y_feat = y.loc[:, feature]
        ks_stat, p_val = ks_2samp(x_feat, y_feat)
        row = {
            'feature_name': feature,
            'ks_statistic': ks_stat,
            'p_value': p_val
        }
        results.append(row)
        if p_val < CONFIG.KS_THRESHOLD:
            targets.append(row)
    return results, targets


def basic_model_evaluation(X, y):
    """
    Return the scores of k-fold cross validation on the given dataset. The basic model here is a XGBoost
    with fixed configuation

    Parameter:
      X: The features used for evaluation
      y: The corresponding labels

    Return:
      scores: Scores of k-fold validation
    """

    model = Pipeline([
        ('sampler_1', RandomOverSampler(sampling_strategy='not majority', random_state=CONFIG.SEED)),
        ('sampler_2', RandomUnderSampler(sampling_strategy='not majority', random_state=CONFIG.SEED)),
        ('classifier', xgb.XGBClassifier(**CONFIG.xgb_basic_params)),
    ])

    _, scores = evaluate_function(X, y, model, n_splits=CONFIG.KFOLD, random_state=CONFIG.SEED)
    return scores


def read_data(X_path, y_path=None):
    """
    Read data from X_path, if y_path is given, also read the labels

    Parameter:
        X_path: The path to read the features
        y_path: The path to read the labels

    Return:
        X: Features from the given path
        y: Corresponding labels from the given path
    """
    X = pd.read_csv(X_path, header=0)
    y = None
    if y_path:
        y = pd.read_csv(y_path, header=0)
    return X, y


def draw_bubble(scores):
    """Draw a bubble plot"""
    X = [ x[0] for x in scores ]
    Y = [ x[1] for x in scores ]

    s = np.array([(x[2]) for x in scores]).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 3000))
    s = scaler.fit_transform(s).flatten()

    c = [ x[2] for x in scores ]

    plt.figure(figsize = (16, 12))

    scatter = plt.scatter(X, Y, s = s, c = c, cmap = "coolwarm",
                          alpha = 0.6, edgecolors = "w", linewidth = 1.5)
    plt.colorbar(scatter, label="Bubble Size")
    plt.xlabel("sampling_ratio1")
    plt.ylabel("sampling_ratio2")
    plt.show()


def bar_plot(data):
    """Draw a bar plot"""
    features = [item['feature_name'] for item in data]
    p_values = [item['p_value'] for item in data]
    feature_indices = np.arange(1, len(data)+1)

    plt.figure(figsize=(15, 8))

    bar_width = 0.6
    bars = plt.bar(feature_indices, p_values, width=bar_width, color='skyblue')

    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (α=0.05)')

    for bar, p in zip(bars, p_values):
        if p < 0.05:
            bar.set_color('salmon')

    plt.xlabel('Feature Name')
    plt.ylabel('p-value')
    plt.title('p-values of KS Test for Each Feature')
    plt.xticks(rotation=90)
    plt.legend()

    plt.xlim(0.5, len(data)+0.5)

    plt.tight_layout()

    plt.show()


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