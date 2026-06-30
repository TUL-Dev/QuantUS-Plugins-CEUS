import numpy as np
from scipy.optimize import curve_fit

def bolus_lognormal(x, auc, mu, sigma, t0):
    with np.errstate(divide='ignore', invalid='ignore'):
        shifted = x - t0
        result = (auc / (2.5066 * sigma * shifted)) * np.exp((-1 / 2) * (((np.log(shifted) - mu) / sigma) ** 2))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def bolus_lognormal_no_t0(x, auc, mu, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (auc / (2.5066 * sigma * x)) * np.exp((-1 / 2) * (((np.log(x) - mu) / sigma) ** 2))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def compute_t0(time, curve, threshold_frac=0.08):
    """First frame where intensity exceeds baseline + threshold_frac * (peak - baseline)."""
    curve = np.array(curve, dtype=float)
    baseline = np.min(curve)
    peak = np.max(curve)
    threshold = baseline + threshold_frac * (peak - baseline)
    above = np.where(curve >= threshold)[0]
    if len(above) == 0:
        return np.nan
    return float(time[above[0]])


def fit_lognormal_curve(time, curve):
    """
    Fit a log-normal distribution to the given curve.
    
    Args:
        time (np.ndarray): The time array corresponding to the curve.
        curve (np.ndarray): The curve data to fit.
    
    Returns:
        tuple: Fitted parameters (auc, pe, tp, mtt, t0, mu, sigma, pe_loc).
    """
    curve = np.asarray(curve, dtype=float)
    amin = np.amin(curve)
    if amin < 0:
        curve = curve - amin
    if np.amax(curve) == 0:
        print("Curve is constant, cannot normalize.")
        return tuple(np.nan for _ in range(9))
    tmppv = np.amax(curve)
    curve = curve / tmppv  # Normalize

    mu_max = np.log(time[-1]) if time[-1] > 0 else 10.0
    auc_max = (np.sum(curve) * (time[1] - time[0])) * 10.0

    try:
        params, _ = curve_fit(
            bolus_lognormal,
            time,
            curve,
            p0=(1.0, 0.0, 1.0, 0.0),
            bounds=([0., 0., 0., 0.], [auc_max, mu_max, 5.0, time[-1]]),
            method='trf',
        )
    except Exception:
        return tuple(np.nan for _ in range(9))

    auc, mu, sigma, t0 = params
    mtt = np.exp(mu + sigma**2 / 2)
    tp = np.exp(mu - sigma**2)

    # Reject unreasonable fits (only reject if tp is beyond the time range)
    if tp > time[-1] or auc > auc_max:
        return tuple(np.nan for _ in range(9))

    fitted_curve = bolus_lognormal(time, *params)
    pe = np.max(fitted_curve)
    pe_loc = np.argmax(fitted_curve)

    return auc, pe, tp, mtt, t0, mu, sigma, pe_loc, tmppv


def fit_lognormal_curve_no_t0(time, curve):
    curve = np.asarray(curve, dtype=float)
    amin = np.amin(curve)
    if amin < 0:
        curve = curve - amin
    if np.amax(curve) == 0:
        print("Curve is constant, cannot normalize.")
        return tuple(np.nan for _ in range(8))
    tmppv = np.amax(curve)
    curve = curve / tmppv

    mu_max = np.log(time[-1]) if time[-1] > 0 else 10.0
    auc_max = (np.sum(curve) * (time[1] - time[0])) * 10.0

    try:
        params, _ = curve_fit(
            bolus_lognormal_no_t0,
            time,
            curve,
            p0=(1.0, 0.0, 1.0),
            bounds=([0., 0., 0.], [auc_max, mu_max, 5.0]),
            method='trf',
        )
    except Exception:
        return tuple(np.nan for _ in range(8))

    auc, mu, sigma = params
    mtt = np.exp(mu + sigma**2 / 2)
    tp = np.exp(mu - sigma**2)

    if tp > time[-1] or auc > auc_max:
        return tuple(np.nan for _ in range(8))

    fitted_curve = bolus_lognormal_no_t0(time, *params)
    pe = np.max(fitted_curve)
    pe_loc = np.argmax(fitted_curve)

    return auc, pe, tp, mtt, mu, sigma, pe_loc, tmppv