import numpy as np
from scipy.optimize import curve_fit

def bolus_lognormal(x, auc, mu, sigma, t0):
   curve_fit=(auc/(2.5066*sigma*(x-t0)))*np.exp(-1*(((np.log(x-t0)-mu)**2)/(2*sigma*sigma)))
   return np.nan_to_num(curve_fit)

def fit_lognormal_curve(time, curve):
    """
    Fit a log-normal distribution to the given curve.
    
    Args:
        time (np.ndarray): The time array corresponding to the curve.
        curve (np.ndarray): The curve data to fit.
    
    Returns:
        tuple: Fitted parameters (auc, pe, tp, mtt, t0, mu, sigma, pe_loc).
    """
    curve /= np.amax(curve)  # Normalize the curve to the maximum value
    params, _ = curve_fit(bolus_lognormal, time, curve, p0=(1.0,3.0,0.5,0.1),bounds=([0., 0., 0., -1.], [np.inf, np.inf, np.inf, 10.]),method='trf')
    timeconst = time[1] - time[0]  # Assuming uniform time intervals

    auc = params[0]; mu=params[1]; sigma=params[2]; t0=timeconst*params[3]; mtt=timeconst*np.exp(mu+sigma*sigma/2);
    tp = timeconst*np.exp(mu-sigma*sigma); wholecurve = bolus_lognormal(time, params[0], params[1], params[2], params[3]); pe = np.max(wholecurve); # took out pe normalization
    pe_loc = np.argmax(wholecurve)

    return auc, pe, tp, mtt, t0, mu, sigma, pe_loc