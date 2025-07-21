import numpy as np
from scipy.stats import skew, kurtosis, entropy

from ..ttc_analysis.ttc_curves.framework import TtcCurvesAnalysis
from .decorators import required_kwargs, dependencies
from .transforms import fit_lognormal_curve

def first_order(analysis_objs: TtcCurvesAnalysis, data_dict: dict, 
                n_frames_to_analyze: int, **kwargs) -> None:
    """
    Compute first-order statistics from the analysis objects and store them in the data dictionary.
    """
    assert isinstance(analysis_objs, TtcCurvesAnalysis), 'analysis_objs must be a TtcCurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    # Compute first-order statistics
    for name, curve in analysis_objs.curves.items():
        curve = curve[:n_frames_to_analyze]
        data_dict[f'Mean_{name}'] = curve.mean()
        data_dict[f'Std_{name}'] = curve.std()
        data_dict[f'Max_{name}'] = curve.max()
        data_dict[f'Min_{name}'] = curve.min()
        data_dict[f'Median_{name}'] = np.median(curve)
        data_dict[f'Variance_{name}'] = np.var(curve)
        if data_dict[f'Variance_{name}'] < 1e-10:
            data_dict[f'Skewness_{name}'] = 0.0
            data_dict[f'Kurtosis_{name}'] = 3.0  # Normal kurtosis
        else:
            data_dict[f'Skewness_{name}'] = skew(curve)
            data_dict[f'Kurtosis_{name}'] = kurtosis(curve)
        data_dict[f'Skewness_{name}'] = skew(curve)
        data_dict[f'Kurtosis_{name}'] = kurtosis(curve)
        data_dict[f'Range_{name}'] = curve.max() - curve.min()
        data_dict[f'Interquartile_Range_{name}'] = np.percentile(curve, 75) - np.percentile(curve, 25)
        data_dict[f'Entropy_{name}'] = entropy(curve)
        data_dict[f'Energy_{name}'] = np.sum(curve ** 2)

@required_kwargs('curves_to_fit')
def lognormal_fit(analysis_objs: TtcCurvesAnalysis, data_dict: dict, 
                  n_frames_to_analyze: int, **kwargs) -> None:
    """
    Fit a log-normal distribution to the given curves.
    """
    curves_to_fit = kwargs.get('curves_to_fit', [])

    all_curve_names = analysis_objs.curves.keys()
    for curve_name in curves_to_fit:
        matching_names = [name for name in all_curve_names if curve_name.lower() in name.lower()]
        for name in matching_names:
            curve = analysis_objs.curves[name][:n_frames_to_analyze]
            auc, pe, tp, mtt, t0, mu, sigma, pe_loc = fit_lognormal_curve(
                analysis_objs.time_arr[:n_frames_to_analyze], curve)
            data_dict[f'AUC_{name}'] = auc
            data_dict[f'PE_{name}'] = pe
            data_dict[f'TP_{name}'] = tp
            data_dict[f'MTT_{name}'] = mtt
            data_dict[f'T0_{name}'] = t0 if t0 >= 0 else 0
            data_dict[f'Mu_{name}'] = mu
            data_dict[f'Sigma_{name}'] = sigma
            data_dict[f'PE_Ix_{name}'] = pe_loc

@dependencies('lognormal_fit')
def wash_rates(analysis_objs: TtcCurvesAnalysis, data_dict: dict, 
               n_frames_to_analyze: int, **kwargs) -> None:
    """
    Compute wash-in and wash-out rates from the curves.
    """
    assert isinstance(analysis_objs, TtcCurvesAnalysis), 'analysis_objs must be a TtcCurvesAnalysis'
    assert isinstance(data_dict, dict), 'data_dict must be a dictionary'

    curves_to_fit = kwargs.get('curves_to_fit', [])
    if not len(curves_to_fit):
        return
    tic_name = curves_to_fit[0]
    matching_names = [name for name in analysis_objs.curves.keys() if tic_name in name]
    tic_name = matching_names[0]
    pe_ix = data_dict[f'PE_Ix_{tic_name}']

    for name, curve in analysis_objs.curves.items():
        # Compute wash-in rate as the slope of the line of best fit for curve[:pe_ix]
        if pe_ix > 1:
            cutoff_ix = min(pe_ix, n_frames_to_analyze)
            x_in = np.arange(cutoff_ix)
            y_in = curve[:cutoff_ix]
            coeffs = np.polyfit(x_in, y_in, 1)
            wash_in_rate = coeffs[0]
        else:
            wash_in_rate = np.nan
        # Compute wash-out rate as the slope of the line of best fit for curve[pe_ix:]
        if pe_ix < n_frames_to_analyze - 1:
            x_out = np.arange(pe_ix, n_frames_to_analyze)
            y_out = curve[pe_ix:n_frames_to_analyze]
            coeffs = np.polyfit(x_out, y_out, 1)
            wash_out_rate = coeffs[0]
        else:
            wash_out_rate = np.nan
        data_dict[f'WashInRate_{name}'] = wash_in_rate
        data_dict[f'WashOutRate_{name}'] = wash_out_rate