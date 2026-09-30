"""
NERVA-ML Target Label Formulation
Constructs binary labels from expression enrichment Z-scores (DGEE)
and validates class balance strictly (fail-loud).
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from nerva.utils.errors import DataFormatError

def make_dgee_labels(
    row_level_E: pd.Series,
    tau_label: float = 1.645,
    label_mode: str = "threshold",
    disease_name: str = "disease"
) -> Tuple[np.ndarray, float]:
    """
    Creates binary labels: 1 if row enrichment Z-score >= tau_label else 0.
    In quantile mode: 1 if row enrichment Z-score >= 75th percentile.
    
    Strictly validates class balance in [10%, 50%]. If outside, raises DataFormatError
    recommending label_mode: quantile (fail-loud, never silently switches thresholds).
    """
    if label_mode == "quantile":
        cutoff = float(np.percentile(row_level_E.values, 75))
        y = (row_level_E.values >= cutoff).astype(int)
    else:
        y = (row_level_E.values >= tau_label).astype(int)

    pos_frac = float(np.mean(y))

    # Strict class-balance check per Section 5.3 & Section 13.2
    if pos_frac < 0.10 or pos_frac > 0.50:
        raise DataFormatError(
            f"Positive fraction is {pos_frac*100:.1f}% for {disease_name} (allowed 10-50%).",
            "Consider label_mode: quantile in config.yaml."
        )

    return y, pos_frac
