"""
feature_engineering.py
=======================
Derived features on top of the raw BitcoinHeist columns
(year, day, length, weight, count, looped, neighbors, income).

These are engineered from domain intuition about ransomware wallet
behavior (rapid mixing, high fan-out, short holding periods) and are
deliberately simple, auditable ratios rather than opaque transforms —
that matters both for SHAP explanations later and for anyone reviewing
this project reading the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import get_logger

log = get_logger(__name__)

_EPS = 1e-9


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with additional derived columns. Silently skips
    a derived feature if its source columns aren't present, so this
    function is safe to call on datasets with slightly different schemas.
    """
    X = X.copy()
    cols = set(X.columns)

    if {"loops", "count"}.issubset(cols):
        X["loop_ratio"] = X["loops"] / (X["count"] + _EPS)
    elif {"looped", "count"}.issubset(cols):
        # BitcoinHeist's actual column name is "looped"
        X["loop_ratio"] = X["looped"] / (X["count"] + _EPS)

    if {"neighbors", "count"}.issubset(cols):
        X["neighbor_density"] = X["neighbors"] / (X["count"] + _EPS)

    if {"income", "count"}.issubset(cols):
        X["avg_income_per_tx"] = X["income"] / (X["count"] + _EPS)

    if {"income", "weight"}.issubset(cols):
        X["income_per_weight"] = X["income"] / (X["weight"] + _EPS)

    if "length" in cols:
        # Ransomware wallets tend to move funds through short, fast chains.
        X["is_short_chain"] = (X["length"] <= X["length"].quantile(0.25)).astype(int)

    if {"day", "year"}.issubset(cols):
        # Cyclical encoding of day-of-year rather than raw integer, so a
        # tree/boosting model doesn't treat day 365 and day 1 as maximally
        # different.
        day_frac = X["day"] / 365.0
        X["day_sin"] = np.sin(2 * np.pi * day_frac)
        X["day_cos"] = np.cos(2 * np.pi * day_frac)

    added = set(X.columns) - cols
    if added:
        log.info("Added engineered features: %s", sorted(added))
    return X
