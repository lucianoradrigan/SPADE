"""Grouped train/val/test splitting, ported from paper_federative's `_grouped_split`/
`prepare_splits` (repo/02_local_baselines_etapa1/train_baseline.py) essentially verbatim -- the
leakage concern (no source recording split across train/val/test) applies identically to
driveflow's simulated scenario runs as it does to real recordings.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


class InsufficientGroupsError(ValueError):
    """Not enough independent groups (acquisitions) to build a leakage-free split -- raised
    instead of silently falling back to a row/window-level split (AUDIT AI-03)."""


def grouped_split(idx_pool: np.ndarray, y_pool: np.ndarray, groups_pool: np.ndarray, test_frac: float, seed: int, strict: bool = False) -> tuple:
    """Splits idx_pool into (train, test) so no group (source scenario run) appears on both
    sides. Falls back progressively: stratified grouped -> plain grouped -> ungrouped (with a
    warning) if there aren't enough groups for the stronger method.

    strict=True removes the last, leaky fallback: an InsufficientGroupsError is raised instead
    of splitting windows of the same run across train/test. The default stays False only for
    backward compatibility of experiments/train_model.py's historical runs."""
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
            if strict:
                raise InsufficientGroupsError(f"grouped split impossible with {n_groups} group(s) -- refusing an ungrouped (leaky) fallback")
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


def _hash_group(g) -> str:
    return hashlib.sha256(str(g).encode()).hexdigest()[:12]


@dataclass
class SplitManifest:
    """Everything needed to audit / reproduce a train/validation/test split (AUDIT AI-03, Q08/Q09):
    the method, seed and requested fractions, the window indices of each split, and the GROUPS
    (acquisitions) in each -- exported hashed, so private recording names never leave the run."""

    method: str
    seed: int
    fractions: dict
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    groups_train: list
    groups_val: list
    groups_test: list
    class_counts: dict = field(default_factory=dict)

    def is_disjoint(self) -> bool:
        a, b, c = set(self.groups_train), set(self.groups_val), set(self.groups_test)
        idx = [set(self.train_idx.tolist()), set(self.val_idx.tolist()), set(self.test_idx.tolist())]
        return not (a & b or a & c or b & c) and not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "seed": int(self.seed),
            "fractions": dict(self.fractions),
            "n_windows": {"train": int(len(self.train_idx)), "val": int(len(self.val_idx)), "test": int(len(self.test_idx))},
            "n_groups": {"train": len(self.groups_train), "val": len(self.groups_val), "test": len(self.groups_test)},
            "group_hashes": {
                "train": sorted(_hash_group(g) for g in self.groups_train),
                "val": sorted(_hash_group(g) for g in self.groups_val),
                "test": sorted(_hash_group(g) for g in self.groups_test),
            },
            "class_counts": self.class_counts,
            "disjoint": self.is_disjoint(),
        }


def grouped_train_val_test_split(groups: np.ndarray, val_frac: float = 0.2, test_frac: float = 0.1, seed: int = 0, y: np.ndarray | None = None) -> SplitManifest:
    """Three-way split by GROUP (acquisition) -- never by window -- so validation (used for
    training-time monitoring) and test (evaluated once, after every decision is frozen) are
    independent of each other and of train. No fallback: fewer than 3 groups, or fractions that
    would leave a split empty, raise InsufficientGroupsError.

    With `y` (classification), groups are assigned per class so each class with >= 3 groups is
    represented in every split; a class with fewer groups cannot be, and its coverage is reported
    in class_counts rather than hidden."""
    groups = np.asarray(groups)
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError(f"invalid fractions val={val_frac}, test={test_frac}")
    uniq = list(dict.fromkeys(groups.tolist()))
    if len(uniq) < 3:
        raise InsufficientGroupsError(f"need >= 3 independent groups for train/val/test, got {len(uniq)}")
    rng = np.random.default_rng(seed)

    def _allocate(pool: list) -> tuple:
        pool = list(pool)
        rng.shuffle(pool)
        n = len(pool)
        if n < 3:
            return pool, [], []
        n_test = max(1, int(round(test_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        if n - n_test - n_val < 1:
            raise InsufficientGroupsError(f"{n} groups cannot fill train/val/test at fractions {1 - val_frac - test_frac:.2f}/{val_frac:.2f}/{test_frac:.2f}")
        return pool[n_test + n_val :], pool[n_test : n_test + n_val], pool[:n_test]

    g_train, g_val, g_test = [], [], []
    if y is not None:
        y = np.asarray(y)
        group_label = {}
        for g, lab in zip(groups.tolist(), y.tolist()):
            group_label.setdefault(g, set()).add(lab)
        if any(len(v) > 1 for v in group_label.values()):
            raise ValueError("a group carries more than one label -- per-class group allocation is undefined")
        for lab in sorted({next(iter(v)) for v in group_label.values()}, key=str):
            tr, va, te = _allocate([g for g in uniq if next(iter(group_label[g])) == lab])
            g_train += tr
            g_val += va
            g_test += te
    else:
        g_train, g_val, g_test = _allocate(uniq)
    if not (g_train and g_val and g_test):
        raise InsufficientGroupsError(f"split left an empty set (train={len(g_train)}, val={len(g_val)}, test={len(g_test)} groups)")

    def _idx(gs):
        return np.flatnonzero(np.isin(groups, np.asarray(gs, dtype=groups.dtype)))

    manifest = SplitManifest(
        method="grouped_by_acquisition" + ("_per_class" if y is not None else ""),
        seed=seed,
        fractions={"train": round(1 - val_frac - test_frac, 4), "val": val_frac, "test": test_frac},
        train_idx=_idx(g_train), val_idx=_idx(g_val), test_idx=_idx(g_test),
        groups_train=g_train, groups_val=g_val, groups_test=g_test,
    )
    if y is not None:
        manifest.class_counts = {
            name: {str(k): int(v) for k, v in zip(*np.unique(y[ix], return_counts=True))}
            for name, ix in (("train", manifest.train_idx), ("val", manifest.val_idx), ("test", manifest.test_idx))
        }
    return manifest
