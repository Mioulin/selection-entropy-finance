
import numpy as np
import pandas as pd

def selection_entropy_orderflow(df, window=600, level_bins=(-5, -3, -1, 0, 1, 3, 5)):
    """
    Compute a toy 'selection entropy' over order-flow selections.
    """
    type_map = {"MB":0, "MS":1, "LB":2, "LS":3, "C":4}
    tvec = df["event_type"].map(type_map).values
    levels = df["level"].values

    bins = np.array(level_bins, dtype=float)
    edges = np.unique(np.concatenate([[-np.inf], bins, [np.inf]]))
    pat = np.digitize(levels, edges[1:-1], right=True)

    n_types = 5
    n_bins = len(edges)-1
    n = len(df)
    se = np.full(n, np.nan, dtype=float)

    eps = 1e-12
    for i in range(window, n):
        pw = pat[i-window:i]
        sw = tvec[i-window:i]

        p_counts = np.bincount(pw, minlength=n_bins)
        p_probs = p_counts / (p_counts.sum() + eps)

        H = 0.0
        for pidx in range(n_bins):
            idx = np.where(pw == pidx)[0]
            if len(idx) == 0:
                continue
            s_vals = sw[idx]
            s_counts = np.bincount(s_vals, minlength=n_types).astype(float)
            if s_counts.sum() == 0:
                continue
            s_probs = s_counts / s_counts.sum()
            s_probs = np.clip(s_probs, eps, 1.0)
            H_p = -np.sum(s_probs * np.log(s_probs))
            H += p_probs[pidx] * H_p
        se[i] = H
    return pd.Series(se, index=df.index)
