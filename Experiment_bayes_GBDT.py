import utils
from Config import CONFIG

import optuna
from optuna.samplers import TPESampler


def bayesian_tuning(objective, n_trials=100, direction='minimize'):
    """
    Runs an bayesian tuning to optimize a given objective function by Optuna.
    This function create an Optua study with the specified direction
        and optimize the given function over a specified number of trials.
    It returns the best trial with optimal hyperparameters and corresponding metric.

    Parameters:
        objective (function): The objective function that the Optuna will optimize.
                              It should accept only a trial object and return a single
                              value to be optimized.
        n_trials (int, optional, default = 100):The number of trials to run in the study.
        direction (str, optional, default = 'maximize'): The direction of optimization.
                                                         Should be 'minimize' or 'maximize'.

    Returns:
    optuna.study: The study object containing details of the optimization process,
                  such as the parameter combinations and corresponding evaluation metrics for each experiment.
    optuna.study.best_trial: The best trial found during the study, including
                             the optimal hyperparameters and the best metric value.

    """
    # Create the Pttuna study with the specified direction and random seed
    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=CONFIG.SEED))

    # Run the optimization process for the specified opjective and number of trials
    study.optimize(objective, n_trials=n_trials)

    # Get the best trial
    best_trial = study.best_trial

    return study, best_trial


def xgb_objective(trial):
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

    # Define the hyperparameters to be optimized.
    params = {
        'objective': 'multi:softprob',
        'num_class': CONFIG.NUM_CLASS,
        "tree_method": 'hist',
        "device": 'cuda',
        'random_state': CONFIG.SEED,  # Random seed
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Number of trees
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 2e-1, log=True),  # Contribution of each tree
        'max_depth': trial.suggest_int('max_depth', 4, 8),  # Maximum depth of each tree
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),  # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),  # L1 regularization
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),  # Fraction of samples for each tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),  # Fraction of features for each tree
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),  # Fraction of features for each node
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        # Fraction of features for each tree level
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
    }

    mean_val, _ = utils.evaluate_function(X, y, params, n_splits=5, random_state=None)
    return mean_val


def lgb_objective(trial):
    """Similar functionality to other ojective functions"""
    # Define the hyperparameters to be optimized.
    params = {
        'objective': 'multiclass',
        'num_class': CONFIG.NUM_CLASS,
        'device_type': 'gpu',
        'verbosity': -1,                                                                # Controls the logging level
        'num_iterations': trial.suggest_int('n_iter', 100, 1000),                       # Number of trees
        'boosting_type': 'gbdt',                                                        # The boosting method used.
        'random_state': CONFIG.SEED,                                                    # Random seed
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),            # L1 regularization
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),            # L2 regularization
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),    # Contribution of each tree
        'max_depth': trial.suggest_int('max_depth', 4, 8),                              # Maximum depth of each tree
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),                         # Number of leaves in each tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),          # Fraction of features for each tree
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),          # Fraction of features for each node
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),          # Fraction of samples for each tree
        'bagging_freq': trial.suggest_int('bagging_freq', 2, 7),                        # Frequency of bagging
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 150),              # Minimum number of samples required to form a leaf
    }

    mean_val, _ = utils.evaluate_function(X, y, params, n_splits=5, random_state=None)
    return mean_val


def cb_objective(trial):
    """Similar functionality to other ojective functions"""
    # Define the hyperparameters to be optimized.
    params = {
        'loss_function': 'MultiClass',
        'classes_count': CONFIG.NUM_CLASS,
        'task_type': 'GPU',
        'iterations': trial.suggest_int('iterations', 100, 1000),                       # Number of trees
        'verbose': False,                                                               # Controls the logging level
        'random_state': CONFIG.SEED,                                                    # Random seed
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 2e-1, log=True),    # Contribution of each tree
        'max_depth': trial.suggest_int('max_depth', 4, 8),                              # Maximum depth of each tree
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),        # L2 regularization
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),                        # Fraction of samples for each tree
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),        # Fraction of features for each tree
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),              # Minimum number of samples required to form a leaf
        'bootstrap_type': 'Bernoulli',  # Method of sampling used
    }

    mean_val, _ = utils.evaluate_function(X, y, params, n_splits=5, random_state=None)
    return mean_val


if __name__ == '__main__':
    X, y = utils.read_data(CONFIG.TRAINING_FEATURE_PATH, CONFIG.TRAINING_LABEL_PATH)

    # Apply bayes tuning according to configuration
    if CONFIG.EXPERIMENT_MODEL_SETTING == 0:
        print("======================Bayes for XGBoost Starts....======================")
        model_study, model_best_trial = bayesian_tuning(xgb_objective, direction = 'minimize')
    elif CONFIG.EXPERIMENT_MODEL_SETTING == 1:
        print("======================Bayes for LGBM Starts....======================")
        model_study, model_best_trial = bayesian_tuning(lgb_objective, direction = 'minimize')
    else:
        print("======================Bayes for CatBoost Starts....======================")
        model_study, model_best_trial = bayesian_tuning(cb_objective, direction = 'minimize')

    print(f"Best trial number: {model_best_trial.number}")
    print(f"Best trial value: {model_best_trial.value}")
    print(f"Best hyperparameters: {model_best_trial.params}")