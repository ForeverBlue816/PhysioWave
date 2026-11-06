"""EEG C1 multi-route pretraining: routes, model, data, training, figures."""
from .routes import (DATASET_IDS, DOWNSTREAM_ONLY, PRETRAIN_DATASETS, RATE_KEYS,
                     ROUTE_IDS, ROUTES, Route, default_sampling_weights)

__all__ = ["ROUTES", "ROUTE_IDS", "RATE_KEYS", "Route", "PRETRAIN_DATASETS",
           "DATASET_IDS", "DOWNSTREAM_ONLY", "default_sampling_weights"]
