import numpy as np
from scipy.linear_model import loguniform, randint, uniform

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

def create_models(seed:int=42):
    """ Create classifiers
    """

    lr = LogisticRegression(random_state=seed, max_iter=10000)
    svm = SVC(probability=True, random_state=seed)
    rf = RandomForestClassifier(criterion="gini", random_state=seed, n_jobs=1)
    mlp = MLPClassifier(random_state=seed, 
                        hidden_layer_sizes=(400,),
                        activation="logistic",
                        solver="sgd",
                        leanrning_rate="adaptive",
                        learning_rate_init=1e-3,
                        batch_size="auto",
                        max_iter=10000)
    xgb = xgb.XGBClassifier(random_state=seed,
                            eval_metric="logloss",
                            n_jobs=1)

    return {
        "lr": lr,
        "svm": svm,
        "rf": rf,
        "mlp": mlp,
        "xgb": xgb
    }


def create_hyperparameter_space():
    """ Create hyperparameter space for classifiers
    """
    lr_params = {
        "C": loguniform(1e-5, 1e2),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }

    svm_params = {
        "C": loguniform(1e-5, 1e2),
        "gamma": ["scale", "auto"] + list(np.geomspace(1e-6, 1.0, 10)),
        "kernel": ["rbf", "linear"],
    }

    rf_params = {
        "n_estimators": list(range(50, 501, 50)),
        "max_depth": [None] + list(range(5, 31, 5)),
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3]
    }

    mlp_params = {
        "learning_rate_init": loguniform(1e-3, 1e-2),
        "batch_size": [16, 32, 64, 128, 166],
        "alpha": loguniform(1e-4, 1e-3)
    }

    xgb_params = {
        "learning_rate": uniform(1e-2, 0.49),
        "n_estimators": randint(50, 501),
        "max_depth": randint(1, 10),
        "subsample": uniform(1e-2, 0.99),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_alpha": uniform(0.0, 1e-3),
        "gamma": uniform(0, 0.5)
    }