# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import math
import numpy as np
from scipy.signal import find_peaks


def pool_adjacent_violators(h, inc):
    """
    Args:
       h, 1D-array, histogram
       inc, bool, saying if we want the non-decreasing (inc = 1) or decreasing regression

    Return:
        g, 1D-array, a new histogram to fit h
    """
    len_h = len(h)
    g = copy.deepcopy(h)
    if not inc: # keep descending
        for i in range(len_h):
            som = g[i]
            for j in range(i - 1, -1, -1):
                if (j == 0) or (g[j] * (i - j) >= som):
                    som /= (i - j)
                    for k in range(j + 1, i + 1):
                        g[k] = som
                    break
                else:
                    som += g[j]
    else: # keep increasing
        for i in range(len_h - 2, -1, -1):
            som = g[i]
            for j in range(i + 1, len_h):
                if (j == len_h - 1) or (g[j] * (j - i) >= som):
                    som /= (j - i)
                    for k in range(i, j):
                        g[k] = som
                    break
                else:
                    som += g[j]
    return g


def entrop(x, y):
    """
    Args
        x, float, in [0., 1.]
        y, float, in [0., 1.]
    Return:
        v, float, the KL distance between x and y
    """
    if (x == 0.):
        v = -math.log10(1 - y)
    elif (x == 1.0):
        v = -math.log10(y)
    else:
        v = (x * math.log10(x / max(y, 1e-6)) + (1.0 - x) * math.log10((1.0 - x) / max(1.0 - y, 1e-6)))
    return v


def max_entropy(h, e, inc):
    """ Compute the maximum entropy of the histogram h for the increasing or decreasing hypothesis
    Args:
       h, 1D-array, histogram
       e, float, parameter used to compute the entropy
       inc, bool, saying if we want the non-decreasing (inc = 1) or decreasing regression

    Return:
        max_entrop, float, the max entrop (KL distance) between h and the hypothesis
    """
    g = copy.deepcopy(h)
    decreas = pool_adjacent_violators(g, inc)
    L = len(g)

    # integrate signals
    g = np.cumsum(g)
    decreas = np.cumsum(decreas)

    # meaningfullness threshold
    N = g[-1]
    seuil = (math.log(L * (L + 1) / 2) + e * math.log(10)) / N

    # search the most meaningfull segment(gap or mode)
    max_entrop = 0.
    for i in range(L):
        for j in range(i, L):
            if (i == 0):
                r, p = g[j], decreas[j]
            else:
                r, p = g[j] - g[i - 1], decreas[j] - decreas[i - 1]
            r, p = r / N, p / N
            v = entrop(r, p)
            if (v > max_entrop):
                max_entrop = v
    max_entrop = (max_entrop - seuil) * N
    return max_entrop


def ftc_seg(H, e=1.0):
    """
    The following functions implement the Fine to Coarse Histogram Segmentation described in
    "J. Delon, A. Desolneux, J-L. Lisani and A-B. Petro, A non parametric approach for histogram segmentation,
    IEEE Transactions on Image Processing, vol.16, no 1, pp.253-261, Jan. 2007."

    Refer: https://github.com/judelo/2007-TIP-HistogramSegmentation

    Args:
       h, 1D-array, histogram
       e, float, parameter used to compute the entropy
       inc, bool, saying if we want the non-decreasing (inc = 1) or decreasing regression

    Return:
        g, 1D-array, a new histogram to fit h
    """
    lH = len(H)
    h = np.array(H)

    # find the list of local minima and maxima of H
    idx_max, _ = find_peaks(h)
    idx_min, _ = find_peaks(-h)
    idx = np.concatenate([idx_min, idx_max])
    idx = np.sort(idx)
    idx = idx.tolist()
    if idx[0] != 0:
        idx = [0] + idx
    if idx[-1] != (lH - 1):
        idx = idx + [lH - 1]

    # find if idx starts with a minimum or a maximum
    begins_with_min = (H[idx[0]] < H[idx[1]])

    len_idx = len(idx)
    val = [0] * (len_idx - 3)
    for k in range(len_idx - 3):
        inc = (
            (begins_with_min and (k % 2 == 0)) or
            ((not begins_with_min) and (k % 2 == 1))
        )
        sub_hist = H[idx[k]: idx[k + 3] + 1]
        val[k] = max_entropy(sub_hist, e, inc)

    # merging of modes
    kmin = np.argmin(val)
    valmin = val[kmin]
    while (len(val) > 0 and valmin < 0):
        idx = idx[: kmin + 1] + idx[kmin + 3:]
        val = val[: kmin + 1] + val[kmin + 3:]
        val = val[: len(idx) - 3]
        if len(val) < 1:
            break
        for j in range(
            max(kmin - 2, 0),
            min(kmin + 1, len(val))
        ):
            inc = (
                    (begins_with_min and (j % 2 == 0)) or
                    ((not begins_with_min) and (j % 2 == 1))
            )
            sub_hist = H[idx[j]: idx[j + 3] + 1]
            val[j] = max_entropy(sub_hist, e, inc)
        kmin = np.argmin(val)
        valmin = val[kmin]

    if begins_with_min:
        idx = idx[::2]
    else:
        idx = idx[1::2]
    return idx

