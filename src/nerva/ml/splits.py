"""
NERVA-ML Grouped Validation Splitter
Prevents cross-donor data leakage using strict GroupKFold on donor IDs
and Leave-One-Division-Out for anatomical domain generalization.
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

def get_donor_outer_splits(groups: np.ndarray, n_outer_folds: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates outer train/test index splits grouped by donor."""
    unique_donors = np.unique(groups)
    if n_outer_folds <= 1:
        # Hold out exactly one donor for testing
        test_donor = unique_donors[-1]
        tr_idx = np.where(groups != test_donor)[0]
        te_idx = np.where(groups == test_donor)[0]
        return [(tr_idx, te_idx)]
    if len(unique_donors) <= n_outer_folds:
        logo = LeaveOneGroupOut()
        return list(logo.split(np.zeros(len(groups)), groups=groups))
    gkf = GroupKFold(n_splits=n_outer_folds)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))

def get_donor_inner_splits(groups_train: np.ndarray, n_inner_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates inner cross-validation splits over training donors only."""
    unique_donors = np.unique(groups_train)
    n_splits = min(len(unique_donors), n_inner_folds)
    if n_splits <= 1:
        # Fallback to 2-fold random split if only 1 donor in fold
        from sklearn.model_selection import KFold
        return list(KFold(n_splits=2, shuffle=True, random_state=42).split(groups_train))
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups_train)), groups=groups_train))

def get_region_heldout_splits(divisions: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generates Leave-One-Division-Out splits for anatomical domain generalization."""
    unique_divs = np.unique(divisions)
    splits = []
    for div in unique_divs:
        test_idx = np.where(divisions == div)[0]
        train_idx = np.where(divisions != div)[0]
        splits.append((str(div), train_idx, test_idx))
    return splits
