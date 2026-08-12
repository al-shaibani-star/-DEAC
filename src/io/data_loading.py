# -*- coding: utf-8 -*-
"""Data loading: CSV file reader with automatic label detection."""
import os
from typing import Optional, Tuple

import numpy as np


def load_csv(path: str, label_col: Optional[int] = -1) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load data from CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file.
    label_col : int or None
        Column index for labels (-1 = last column, None = no labels).

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray or None
        Label vector (None if no labels).
    name : str
        Dataset filename.
    """
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV path not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError("CSV appears invalid or too small.")

    if label_col is None:
        Xdf = df
        y = None
    else:
        if int(label_col) < 0:
            label_col = df.shape[1] + int(label_col)
        if int(label_col) < 0 or int(label_col) >= df.shape[1]:
            raise ValueError(f"label_col out of range: {label_col}")
        y = df.iloc[:, int(label_col)].to_numpy()
        Xdf = df.drop(df.columns[int(label_col)], axis=1)

    # Force numeric
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN
    all_nan_cols = Xdf.columns[Xdf.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        Xdf = Xdf.drop(columns=all_nan_cols)

    if Xdf.shape[1] == 0:
        raise ValueError("All feature columns became non-numeric/empty after coercion.")

    # Fill NaN with per-column mean
    means = Xdf.mean(axis=0, numeric_only=True)
    Xdf = Xdf.fillna(means).fillna(0.0)

    X = Xdf.to_numpy(dtype=np.float32)

    if y is not None:
        if y.dtype == object:
            ys = pd.to_numeric(pd.Series(y), errors="coerce")
            if ys.isna().any():
                y = pd.factorize(pd.Series(y).astype(str))[0]
            else:
                y = ys.fillna(-1).to_numpy(dtype=int)
        else:
            y = np.asarray(y, dtype=int)

    return X, y, os.path.basename(path)
