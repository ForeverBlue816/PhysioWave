"""sEMG C1 multi-route pretraining: the EEG C1 model and loop on sEMG routes."""
from .routes import (DATASET_IDS, DOWNSTREAM_ONLY, PRETRAIN_DATASETS, RATE_KEYS,
                     ROUTE_IDS, ROUTES)

__all__ = ["ROUTES", "ROUTE_IDS", "RATE_KEYS", "PRETRAIN_DATASETS",
           "DATASET_IDS", "DOWNSTREAM_ONLY"]
