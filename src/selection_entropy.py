
import numpy as np

def _entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p))

def selection_entropy_series(returns, window=200, n_bins=7, vol_window=60):
    """
    Demo 'selection entropy' proxy for financial time series.
    (Illustrative; not the exact research formulation.)
    """
    x = np.asarray(returns).astype(float)
    n = len(x)
    eps = 1e-12
    rolling_std = np.sqrt(np.convolve((x - np.mean(x))**2, np.ones(vol_window)/vol_window, mode='same') + eps)
    xn = x / (rolling_std + eps)
    vol = np.sqrt(np.convolve(x**2, np.ones(vol_window)/vol_window, mode='same') + eps)
    thresh = np.nanmedian(vol)
    regime = (vol > thresh).astype(int)

    quantiles = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(xn[~np.isnan(xn)], quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(np.nanmin(xn), np.nanmax(xn), n_bins+1)
    pattern = np.digitize(xn, edges[1:-1], right=True)

    se = np.full(n, np.nan)
    for t in range(window, n):
        pw = pattern[t-window:t-1]
        sw = regime[t-window+1:t]
        p_counts = np.bincount(pw, minlength=n_bins)
        p_probs = p_counts / (np.sum(p_counts) + 1e-12)
        H = 0.0
        for pval in range(n_bins):
            idx = np.where(pw == pval)[0]
            if len(idx) == 0:
                continue
            s_vals = sw[idx]
            s0 = np.sum(s_vals == 0)
            s1 = np.sum(s_vals == 1)
            probs = np.array([s0, s1], dtype=float)
            if probs.sum() == 0:
                continue
            probs /= probs.sum()
            H += p_probs[pval] * _entropy(probs)
        se[t] = H
    return se

def kl_divergence_series(returns, window=200, n_bins=30):
    x = np.asarray(returns).astype(float)
    n = len(x)
    kl = np.full(n, np.nan)
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    bins = np.linspace(lo, hi, n_bins+1)
    eps = 1e-12
    prev_hist, _ = np.histogram(x[:window], bins=bins, density=True)
    prev_hist = prev_hist + eps
    prev_hist /= prev_hist.sum()
    for t in range(window+1, n+1):
        cur_hist, _ = np.histogram(x[t-window:t], bins=bins, density=True)
        cur_hist = cur_hist + eps
        cur_hist /= cur_hist.sum()
        kl_val = np.sum(cur_hist * (np.log(cur_hist) - np.log(prev_hist)))
        kl[t-1] = kl_val
        prev_hist = cur_hist
    return kl
