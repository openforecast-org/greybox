"""Transform functions for greybox."""

import numpy as np


def bc_transform(y: np.ndarray, lambda_bc: float) -> np.ndarray:
    """Box-Cox transformation."""
    if lambda_bc == 0:
        return np.log(y)
    else:
        return (np.power(y, lambda_bc) - 1) / lambda_bc


def bc_transform_inv(y: np.ndarray, lambda_bc: float) -> np.ndarray:
    """Inverse Box-Cox transformation."""
    if lambda_bc == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_bc + 1, 1 / lambda_bc)


def mean_fast(
    x: np.ndarray, df: None | int = None, trim: float = 0.0, side: str = "lower"
) -> float:
    """Fast mean calculation with optional trimming."""
    if df is None:
        df = len(x)

    if trim != 0:
        if side == "both":
            n = len(x)
            n_trim = int(n * trim)
            if n_trim > 0:
                x_sorted = np.sort(x)
                return np.mean(x_sorted[n_trim:-n_trim])
            return np.mean(x)
        else:
            # R's meanFast keeps the floor(n * (1 - trim)) highest values for
            # side="lower" and the lowest ones for side="upper"
            x_sorted = np.sort(x)
            if side == "lower":
                x_sorted = x_sorted[::-1]
            n_kept = int(np.floor(len(x) * (1 - trim)))
            return np.mean(x_sorted[:n_kept])
    else:
        return np.sum(x) / df
