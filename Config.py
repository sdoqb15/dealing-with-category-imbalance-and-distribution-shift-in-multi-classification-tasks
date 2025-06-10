from pathlib import Path


class CONFIG:
    """ Set all the configuations here for convenience """

    # ======== Experiment Configurations ========
    EXPERIMENT_MODEL_SETTING = 1  # 0---XGBoost, 1---LGBM, 2---CatBoost
    # ======== Row Sampeling Experiment Configurations ========
    EXPERIMENT_ROW_SAMPLING_STRATEGY = 1  # 0---Random, 1----SMOTEENN, 2---ADASYN & TomekLinks

    EXPERIMENT_UNDER_THRESH = [(1000, 1600), (1700, 2300), (2400, 3000)]
    EXPERIMENT_UNDER_THRESH_IDX = 2

    EXPERIMENT_OVER_THRESH = (100, 1500)
    EXPERIMENT_STEP = 100

    # ======== Feature Sampeling Experiment Configurations ========
    EXPERIMENT_FEATURE_SAMPLING_STRATEGY = 0  # 0----GBDT Importance, 1---SHAP
    # Result of feature sampeling
    COL_XGBOOST = [17, 270, 283, 48, 99, 90, 222, 88, 111, 299, 55, 230, 172, 274, 46, 71, 116, 182, 227, 214, 215, 122, 47, 205, 21, 238, 53, 79, 290, 87, 265, 279, 125, 131, 64, 263, 59, 112, 5, 257, 218, 113, 49, 247, 280, 20, 162, 216, 101, 268, 23, 133, 130, 97, 43, 231, 252, 194, 292, 261, 295, 132, 140, 10, 22, 141, 249, 4, 153, 210, 135, 284, 77, 147, 41, 134, 25, 142, 177, 171, 92, 69, 239, 106, 253, 61, 149, 208, 286, 223, 39, 235, 98, 285, 16, 60, 278, 234, 19, 27, 123, 183, 240, 168, 37, 108, 78, 76, 251, 96, 143, 158, 120, 207, 244, 35, 89, 276, 260, 75, 24, 170, 287, 36, 269, 254, 144, 282, 34, 82, 50, 196]
    COL_LGBM = [133, 17, 143, 231, 185, 71, 58, 125, 55, 87, 106, 84, 255, 113, 131, 117, 60, 157, 294, 90, 19, 62, 49, 274, 92, 146, 226, 239, 237, 174, 216, 258, 154, 74, 249, 179, 7, 122, 78, 83, 243, 172, 9, 98, 81, 103, 257, 204, 76, 206, 94, 27, 182, 283, 209, 11, 207, 222, 128, 263, 153, 299, 227, 85, 88, 21, 286, 86, 31, 141, 278, 217, 69, 284, 112, 95, 230, 215, 48, 57, 124, 232, 134, 104, 47, 4, 265, 233, 193, 238, 202, 22, 212, 80, 109, 221, 6, 218, 199, 23, 68, 82, 14, 43, 292, 123, 219, 73, 127, 32, 291, 46, 5, 67, 41, 18, 198, 272, 194, 142, 99, 241, 101, 175, 188, 270, 189, 13, 187, 70, 280, 111, 16, 135, 51, 96, 275, 229, 45, 52, 246, 38, 37, 234, 282, 205, 40, 214, 268, 191, 26, 29, 158, 254, 61, 63, 196, 164, 169, 297, 252, 260, 298, 77, 79, 271, 65, 171, 130, 276, 166, 2, 116, 167, 50, 140, 162, 97, 190, 0, 281, 136, 186, 28, 129, 201, 59, 242, 139, 115, 10, 224, 137, 35, 197, 75, 287, 24, 279, 20, 251, 160, 277, 173, 138, 44, 256, 290, 102, 208, 108, 33, 114, 107, 176, 152, 91, 100, 151, 200, 293, 144, 223, 132, 118, 163, 150, 170, 250, 120, 262, 36, 285, 183, 245, 168, 148, 296, 261, 266]
    COL_CATBOOST = [99, 270, 107, 17, 222, 130, 186, 171, 5, 111, 227, 230, 46, 274, 264, 47, 81, 283, 101, 48, 69, 98, 235, 176, 265, 79, 75, 90, 8, 37, 290, 55, 201, 257, 53, 71, 263, 153, 118, 141, 116, 253, 88, 205, 214, 162, 279, 228, 67, 59, 226, 208, 237, 223, 157, 172, 113, 149, 215, 280, 77, 70, 7, 89, 123, 36, 19, 276, 76, 234, 60, 21, 140, 108, 10, 145, 9, 204, 61, 200, 224, 119, 30, 87, 259, 177, 13, 282, 161, 261, 170, 129, 181, 50, 191, 173, 92, 292, 271, 249, 15, 154, 23, 189, 233, 182, 198, 56, 278, 299, 125, 144, 231, 41, 86, 174, 33, 254, 168, 20, 216, 83, 286, 104, 206, 65, 289, 131, 152, 247, 296, 268, 260, 269, 49, 210, 239, 22, 139, 244, 43, 134, 169, 127, 284, 147, 115, 293, 291, 245, 183, 160, 26, 165, 158, 225, 122, 109, 179, 16, 112, 255, 199, 78, 82, 213, 194, 3, 138, 143, 137, 251, 44, 91, 221, 4, 120, 73, 240, 2]

    # Random seed
    SEED = 42

    # Cross Validation
    KFOLD = 5

    # Number of class
    NUM_CLASS = 28

    # File Path
    ROOT = Path('./')
    TRAINING_FEATURE_PATH = ROOT / "x-ydatasets/X_train.csv"
    TRAINING_LABEL_PATH = ROOT / "x-ydatasets/y_train.csv"
    TEST_1_PATH = ROOT / 'x-ydatasets/X_test_1.csv'
    TEST_2_PATH = ROOT / 'x-ydatasets/X_test_2.csv'
    TEST_2_LABEL_PATH = ROOT / 'x-ydatasets/y_test_2_reduced.csv'
    TEST_FEATURE_PATH = ROOT / "x-ydatasets/X_test_1.csv"
    TEST_SHIFTED_PATH = ROOT / 'x-ydatasets/X_test_2.csv'
    TEST_SHIFTED_LABEL_PATH = ROOT / 'x-ydatasets/y_test_2_reduced.csv'

    BEST_PARAMETER_PATH = ROOT / "./Params/best_parameter_v2.csv"
    WEIGHT_TABLE_PATH = ROOT / './Params/weight_table_full.csv'

    # THRESHOLD of KS test
    KS_THRESHOLD = 0.05

    # GBDT models
    xgb_basic_params = {
        'objective': 'multi:softprob',
        'num_class': NUM_CLASS,  # Enable categorical features
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': SEED,  # Random seed
    }

    lgb_basic_params = {
        'objective': 'multiclass',
        'num_class': NUM_CLASS,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        # GPU
        'device': 'gpu',
        'n_jobs': 1,
    }

    cb_basic_params = {
        'loss_function': 'MultiClass',
        'verbose': False,
        'random_state': SEED,
        # GPU
        'task_type': 'GPU',
        'thread_count': 1,
        'bootstrap_type':    'Bernoulli',
    }
