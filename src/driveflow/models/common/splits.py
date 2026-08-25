"""Grouped train/val/test splitting, ported from paper_federative's `_grouped_split`/
`prepare_splits` (repo/02_local_baselines_etapa1/train_baseline.py) essentially verbatim -- the
leakage concern (no source recording split across train/val/test) applies identically to
driveflow's simulated scenario runs as it does to real recordings.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


def grouped_split(idx_pool: np.ndarray, y_pool: np.ndarray, groups_pool: np.ndarray, test_frac: float, seed: int) -> tuple:
    """Splits idx_pool into (train, test) so no group (source scenario run) appears on both
    sides. Falls back progressively: stratified grouped -> plain grouped -> ungrouped (with a
    warning) if there aren't enough groups for the stronger method."""
    n_groups = len(np.unique(groups_pool))
    n_splits = max(2, min(round(1 / test_frac), n_groups))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_i, test_i = next(splitter.split(idx_pool, y_pool, groups_pool))
    except ValueError:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
            train_i, test_i = next(splitter.split(idx_pool, y_pool, groups_pool))
        except ValueError:
            print("  [WARN] Grouped split failed -- using a plain split (risk of leakage between windows of the same scenario run)")
            train_i, test_i = train_test_split(np.arange(len(idx_pool)), test_size=test_frac, random_state=seed)
    return idx_pool[train_i], idx_pool[test_i]


def prepare_classification_splits(X: np.ndarray, y_str: np.ndarray, classes: list, groups: np.ndarray, seed: int = 42) -> tuple:
    """Encode + 70/20/10 split grouped by scenario run: all windows from the same run fall into
    a single split, so a model can't be tested on windows from a run it partly trained on.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder).
    """
    le = LabelEncoder()
    le.fit(classes)
    y = le.transform(y_str)
    groups = np.asarray(groups)
    idx = np.arange(len(X))

    idx_tv, idx_test = grouped_split(idx, y, groups, 0.10, seed)
    idx_train, idx_val = grouped_split(idx_tv, y[idx_tv], groups[idx_tv], 0.222, seed)

    assert not (set(groups[idx_train]) & set(groups[idx_test]))
    assert not (set(groups[idx_train]) & set(groups[idx_val]))
    assert not (set(groups[idx_val]) & set(groups[idx_test]))

    return (X[idx_train], X[idx_val], X[idx_test], y[idx_train], y[idx_val], y[idx_test], le)
