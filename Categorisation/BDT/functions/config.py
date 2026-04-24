"""BDT config shim.

Re-exports the shared STXS dicts from ``Categorisation.common.config`` and
defines BDT-specific XGBoost hyperparameters. Consumers (trainer.py, utils.py,
plotting.py, __init__.py) keep their ``from .config import ...`` imports
unchanged.
"""

# Re-export everything shared.
from Categorisation.common.config import *  # noqa: F401, F403

# BDT-specific hyperparameters.
DEFAULT_XGBOOST_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.02283927455859124,
    'n_estimators': 100,
    'subsample': 0.6744000233410434,
    'colsample_bytree': 0.852790213610976,
    'colsample_bylevel': 0.9266434685743434,
    'colsample_bynode': 0.9130460162125724,
    'reg_lambda': 1.9203085049229334,
    'reg_alpha': 1.7786874433209414,
    'min_child_weight': 15,
    'gamma': 0.07379840270459205,
}

OPTUNA_SEARCH_SPACE = {
    'max_depth': (3, 13),
    'learning_rate': (0.01, 0.1),
    'n_estimators': (500, 1000),
    'subsample': (0.4, 0.8),
    'colsample_bytree': (0.3, 1),
    'colsample_bylevel': (0.3, 1),
    'colsample_bynode': (0.3, 1),
    'reg_lambda': (0, 3.0),
    'reg_alpha': (0.0, 3.0),
    'min_child_weight': (1, 30),
    'gamma': (0, 0.2),
}
