"""DNN config shim — re-exports shared STXS dicts.

DNN has no xgboost-specific params. If DNN-specific hyperparameters are added
(e.g. learning rate, hidden dims), they belong here so
``from .config import ...`` calls in the DNN trainer keep working.
"""

from Categorisation.common.config import *  # noqa: F401, F403
