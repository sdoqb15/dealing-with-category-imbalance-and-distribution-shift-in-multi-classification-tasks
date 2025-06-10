## Run Instruction

### Environment Setup

Linux Environment: install.sh

Windows Environment: install.bat

Anaconda (prefer):

``conda create -n Group42_test_env python=3.10``

``conda activate Group42_test_env ``

``pip install -r requirements.txt``

### Models

 The models are in https://unsw-my.sharepoint.com/:u:/g/personal/z5527738_ad_unsw_edu_au/EUib3bdQ8FpDt4P2JYWxd6oBafKM1yG8GV1C34rzBD2A-w?e=jIzhID. Please download the models and extract them under the project directory, make sure the models are in the path ``/models/``

### predict.py

Let directory  ``Params, models`` be in the same directory of ``predict.py``.  Fill the test dataset path at ``X_test_path`` and ``y_test_path`` then run the code.

### Other codes

Fill the necessary paths in CONFIG.py, then run the code, which will show the process of our experiments.

## Files Description

### predict.py

Use our pre-trained models to predict test set, and evaluate models' performance using weighted log loss.

### CONFIG.py

Configurations of experiments and models, etc.

### Experiment_bayes_GBDT.py

Experiment code of bayes tuning  on GBDT models

### Experiment_bayes_LR.py

Experiment code of bayes tuning  on meta model (Logistic Regression)

### Experiment_ColSampling.py

Experiment code of column sampling

### Experiment_RowSampeling.py

Experiment code of row sampling

### Experiment_Distribution_Shift_Detect.py

Experiment code of detecting distribution shift, which shows the way we detect covariate shift and label shift of the dataset.

### Experiment_Distribution_Shift_solve.py

Experiment code of solving distribution shift, which shows the way we solve the distribution shift using sample weight, label weight and retrain the model with shifted model.