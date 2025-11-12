"""
CORRECTED: 3D Motion Compensation Evaluation Metrics
Measures B-mode intensity similarity WITHIN the VOI, not mask overlap
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd
from skimage import metrics

# ============================================================================
# VOI B-mode Similarity Metrics (CORRECTED)
# ============================================================================

def compute_roi_similarity_metrics(volume1, volume2, ref_mask, mask):
    """
    Compute similarity metrics between two volumes within a masked region.
    
    This measures how similar the B-mode intensities are within the VOI,
    which reflects whether the VOI is tracking the same tissue.
    
    Args:
        volume1: Reference volume (z, y, x)
        volume2: Current volume (z, y, x)
        ref_mask: Binary mask defining VOI in reference volume (z, y, x)
        mask: Binary mask defining VOI (z, y, x)
        
    Returns:
        dict: Similarity metrics
    """
    # Extract intensities within mask
    roi1 = volume1[ref_mask > 0]
    roi2 = volume2[mask > 0]
    
    if len(roi1) == 0 or len(roi2) == 0:
        return {
            'correlation': 0.0,
            'ssim': 0.0,
            'mse': np.inf,
            'mae': np.inf
        }
    
    # 1. Pearson Correlation
    if len(roi1) > 1:
        correlation, _ = pearsonr(roi1.flatten(), roi2.flatten())
    else:
        correlation = 0.0
    
    # # 2. Normalized Cross-Correlation
    # roi1_norm = (roi1 - np.mean(roi1)) / (np.std(roi1) + 1e-10)
    # roi2_norm = (roi2 - np.mean(roi2)) / (np.std(roi2) + 1e-10)
    # ncc = np.mean(roi1_norm * roi2_norm)
    
    # 3. Structural Similarity (simplified)
    ssim, _ = metrics.structural_similarity(roi1, roi2, full=True)
    
    # 4. Mean Squared Error
    mse = mean_squared_error(roi1, roi2)
    
    # 5. Mean Absolute Error
    mae = np.mean(np.abs(roi1 - roi2))
    
    return {
        'correlation': correlation,
        # 'ncc': ncc,
        'ssim': ssim,
        'mse': mse,
        'mae': mae
    }


def compute_voi_bmode_similarity_over_time(
    bmode_volumes,
    mask_with_mc,
    mask_without_mc,
    reference_frame=0
):
    """
    CORRECTED VERSION: Compute B-mode similarity WITHIN VOI over time.
    
    For MC case: Compare B-mode intensities within MC mask at each frame
                 to reference B-mode within reference mask
    
    For non-MC case: Compare B-mode intensities within static mask at each frame
                      to reference B-mode within reference mask
    
    This measures whether the VOI is tracking the SAME TISSUE (based on B-mode appearance)
    rather than just measuring mask overlap.
    
    Args:
        bmode_volumes: B-mode data (z, y, x, t) or (t, z, y, x)
        mask_with_mc: Motion compensated mask (z, y, x, t)
        mask_without_mc: Mask without MC (z, y, x, t)
        reference_frame: Reference frame index
        
    Returns:
        dict: Results for MC and non-MC cases
    """
    n_frames = bmode_volumes.shape[-1]
    ref_volume = bmode_volumes[..., reference_frame]
    
    # Initialize results
    results = {
        'with_mc': {
            'correlation': [],
            'ssim': [],
            'mse': [],
            'mae': []
        },
        'without_mc': {
            'correlation': [],
            'ssim': [],
            'mse': [],
            'mae': []
        }
    }
    
    # Reference masks
    ref_mask = mask_with_mc[..., reference_frame]

    print("Computing B-mode similarity within VOI over time...")
    print(f"This measures whether the VOI tracks the SAME TISSUE based on B-mode appearance.")
    
    for frame_idx in range(n_frames):
        if frame_idx % 50 == 0:
            print(f"  Processing frame {frame_idx}/{n_frames}...")
        
        current_volume = bmode_volumes[..., frame_idx]
        
        # ===== WITH MOTION COMPENSATION =====
        # Use the MC mask at this frame to extract B-mode intensities
        current_mask_mc = mask_with_mc[..., frame_idx]
        
        # Compare B-mode within current MC mask to B-mode within reference mask
        # This tells us: is the VOI still over the same tissue?
        metrics_mc = compute_roi_similarity_metrics(
            ref_volume, current_volume, ref_mask, current_mask_mc
        )
        
        for key in metrics_mc:
            results['with_mc'][key].append(metrics_mc[key])
        
        # ===== WITHOUT MOTION COMPENSATION =====
        # Use the static mask (same for all frames) to extract B-mode intensities
        current_mask_no_mc = mask_without_mc[..., frame_idx]
        
        # Compare B-mode within static mask to B-mode within reference mask
        # Since mask doesn't move but tissue does, similarity will DECREASE
        metrics_no_mc = compute_roi_similarity_metrics(
            ref_volume, current_volume, ref_mask,current_mask_no_mc
        )
        
        for key in metrics_no_mc:
            results['without_mc'][key].append(metrics_no_mc[key])
    
    print("  Done!")
    
    return results


def plot_voi_bmode_similarity_comparison(results, output_path=None):
    """
    Create comprehensive plots comparing B-mode similarity within VOI.
    
    Args:
        results: Dictionary from compute_voi_bmode_similarity_over_time
        output_path: Optional path to save figure
    """
    metrics = ['correlation', 'ssim']
    metric_names = ['Pearson Correlation', 'SSIM']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    n_frames = len(results['with_mc']['correlation'])
    frames = np.arange(n_frames)
    
    # Plot similarity metrics
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        with_mc = results['with_mc'][metric]
        without_mc = results['without_mc'][metric]
        
        # Plot curves
        ax.plot(frames, with_mc, 'g-', linewidth=2, alpha=0.7, 
               label='With Motion Compensation')
        ax.plot(frames, without_mc, 'r-', linewidth=2, alpha=0.7,
               label='Without Motion Compensation')
        
        # Calculate mean values
        mean_mc = np.mean(with_mc)
        mean_no_mc = np.mean(without_mc)
        
        # Add horizontal lines for means
        ax.axhline(mean_mc, color='g', linestyle='--', alpha=0.5,
                  label=f'Mean MC: {mean_mc:.3f}')
        ax.axhline(mean_no_mc, color='r', linestyle='--', alpha=0.5,
                  label=f'Mean No-MC: {mean_no_mc:.3f}')
        
        ax.set_xlabel('Frame Number', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} Within VOI Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set appropriate y-limits
        if metric in ['correlation', 'ssim']:
            ax.set_ylim([-0.1, 1.05])
    
    # Summary statistics in last subplot
    ax_summary = axes[-1]
    ax_summary.axis('off')
    
    summary_text = "B-mode Similarity Summary:\n\n"
    summary_text += "Measuring tissue appearance within VOI\n"
    summary_text += "(Higher = VOI tracking same tissue)\n\n"
    
    for metric, name in zip(metrics, metric_names):
        mean_mc = np.mean(results['with_mc'][metric])
        std_mc = np.std(results['with_mc'][metric])
        mean_no_mc = np.mean(results['without_mc'][metric])
        std_no_mc = np.std(results['without_mc'][metric])
        improvement = ((mean_mc - mean_no_mc) / (mean_no_mc + 1e-10)) * 100
        
        summary_text += f"{name}:\n"
        summary_text += f"  MC:     {mean_mc:.3f} ± {std_mc:.3f}\n"
        summary_text += f"  No-MC:  {mean_no_mc:.3f} ± {std_no_mc:.3f}\n"
        summary_text += f"  Improv: {improvement:+.1f}%\n\n"
    
    # MSE and MAE (lower is better)
    summary_text += "Error Metrics (lower is better):\n"
    mse_mc = np.mean(results['with_mc']['mse'])
    mse_no_mc = np.mean(results['without_mc']['mse'])
    mse_reduction = ((mse_no_mc - mse_mc) / (mse_no_mc + 1e-10)) * 100
    
    summary_text += f"MSE:\n"
    summary_text += f"  MC:     {mse_mc:.1f}\n"
    summary_text += f"  No-MC:  {mse_no_mc:.1f}\n"
    summary_text += f"  Reduct: {mse_reduction:.1f}%\n"
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=9, 
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved VOI B-mode similarity plot to {output_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# TIC Analysis and Lognormal Fitting (UNCHANGED)
# ============================================================================

def bolus_lognormal(t, auc, mu, sigma, t0):
    """Log-normal bolus function for contrast enhancement curve fitting."""
    t_shifted = t - t0
    t_shifted = np.maximum(t_shifted, 1e-10)
    exponent = -((np.log(t_shifted) - mu) ** 2) / (2 * sigma ** 2)
    return (auc / (t_shifted * sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)


def fit_lognormal_curve(time, curve):
    """
    Fit a log-normal distribution to the given curve.
    
    Args:
        time: Time array
        curve: Curve data to fit
        
    Returns:
        tuple: Fitted parameters and fitted curve values
    """
    # Prepare curve
    curve = curve - np.amin(curve)
    
    if np.amax(curve) == 0:
        print("Curve is constant, cannot normalize.")
        return tuple(np.nan for _ in range(8)), None
    
    curve_norm = curve / np.amax(curve)
    
    # Initial guesses
    auc_guess = np.sum(curve_norm) * (time[1] - time[0])
    peak_idx = np.argmax(curve_norm)
    mu_guess = np.log(time[peak_idx] + 1e-10)
    sigma_guess = 0.5
    t0_guess = time[peak_idx] * 0.15
    
    try:
        params, _ = curve_fit(
            bolus_lognormal,
            time,
            curve_norm,
            p0=(auc_guess, mu_guess, sigma_guess, t0_guess),
            bounds=([0., -10., 0.01, 0.], [np.inf, 10., 5.0, time[-1]]),
            method='trf',
            maxfev=10000
        )
    except Exception as e:
        print(f"Error fitting curve: {e}")
        return tuple(np.nan for _ in range(8)), None
    
    auc, mu, sigma, t0 = params
    
    # Calculate derived parameters
    mtt = np.exp(mu + sigma**2 / 2)
    tp = np.exp(mu - sigma**2)
    
    # Get fitted curve
    fitted_curve = bolus_lognormal(time, *params)
    pe = np.max(fitted_curve)
    pe_loc = np.argmax(fitted_curve)
    
    return (auc, pe, tp, mtt, t0, mu, sigma, pe_loc), fitted_curve


def evaluate_tic_fitting(time_arr, tic_mc, tic_no_mc):
    """
    Evaluate and compare TIC fitting quality for MC vs non-MC.
    
    Args:
        time_arr: Time array
        tic_mc: TIC with motion compensation
        tic_no_mc: TIC without motion compensation
        
    Returns:
        dict: Comprehensive fitting metrics
    """
    results = {}
    
    print("\nFitting TIC curves with lognormal model...")
    
    # Fit with MC
    print("  Fitting TIC with motion compensation...")
    params_mc, fitted_mc = fit_lognormal_curve(time_arr, tic_mc)
    
    # Fit without MC
    print("  Fitting TIC without motion compensation...")
    params_no_mc, fitted_no_mc = fit_lognormal_curve(time_arr, tic_no_mc)
    
    # Store parameters
    param_names = ['AUC', 'PE', 'TP', 'MTT', 'T0', 'mu', 'sigma', 'PE_loc']
    
    results['with_mc'] = {
        'params': dict(zip(param_names, params_mc)),
        'fitted_curve': fitted_mc
    }
    
    results['without_mc'] = {
        'params': dict(zip(param_names, params_no_mc)),
        'fitted_curve': fitted_no_mc
    }
    
    # Calculate fitting quality metrics
    if fitted_mc is not None:
        tic_mc_norm = (tic_mc - np.min(tic_mc)) / (np.max(tic_mc) - np.min(tic_mc) + 1e-10)
        
        ss_res_mc = np.sum((tic_mc_norm - fitted_mc) ** 2)
        ss_tot_mc = np.sum((tic_mc_norm - np.mean(tic_mc_norm)) ** 2)
        r2_mc = 1 - (ss_res_mc / ss_tot_mc)
        
        rmse_mc = np.sqrt(mean_squared_error(tic_mc_norm, fitted_mc))
        corr_mc, _ = pearsonr(tic_mc_norm, fitted_mc)
        
        results['with_mc']['r2'] = r2_mc
        results['with_mc']['rmse'] = rmse_mc
        results['with_mc']['correlation'] = corr_mc
        results['with_mc']['residual_sum_squares'] = ss_res_mc
    
    if fitted_no_mc is not None:
        tic_no_mc_norm = (tic_no_mc - np.min(tic_no_mc)) / (np.max(tic_no_mc) - np.min(tic_no_mc) + 1e-10)
        
        ss_res_no_mc = np.sum((tic_no_mc_norm - fitted_no_mc) ** 2)
        ss_tot_no_mc = np.sum((tic_no_mc_norm - np.mean(tic_no_mc_norm)) ** 2)
        r2_no_mc = 1 - (ss_res_no_mc / ss_tot_no_mc)
        
        rmse_no_mc = np.sqrt(mean_squared_error(tic_no_mc_norm, fitted_no_mc))
        corr_no_mc, _ = pearsonr(tic_no_mc_norm, fitted_no_mc)
        
        results['without_mc']['r2'] = r2_no_mc
        results['without_mc']['rmse'] = rmse_no_mc
        results['without_mc']['correlation'] = corr_no_mc
        results['without_mc']['residual_sum_squares'] = ss_res_no_mc
    
    # Calculate TIC variability
    results['with_mc']['cv'] = np.std(tic_mc) / (np.mean(tic_mc) + 1e-10)
    results['without_mc']['cv'] = np.std(tic_no_mc) / (np.mean(tic_no_mc) + 1e-10)
    
    print("  Done!")
    
    return results


def plot_tic_fitting_comparison(time_arr, tic_mc, tic_no_mc, fitting_results, 
                                output_path=None):
    """Create comprehensive TIC fitting comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: TIC Curves with Fits
    ax1 = axes[0, 0]
    
    ax1.plot(time_arr, tic_mc, 'g-', linewidth=2, alpha=0.7, label='MC (Raw)')
    ax1.plot(time_arr, tic_no_mc, 'r-', linewidth=2, alpha=0.7, label='No-MC (Raw)')
    
    if fitting_results['with_mc']['fitted_curve'] is not None:
        fitted_mc_scaled = (fitting_results['with_mc']['fitted_curve'] * 
                           (np.max(tic_mc) - np.min(tic_mc)) + np.min(tic_mc))
        ax1.plot(time_arr, fitted_mc_scaled, 'g--', linewidth=2,
                label=f"MC (Fit, R²={fitting_results['with_mc']['r2']:.3f})")
    
    if fitting_results['without_mc']['fitted_curve'] is not None:
        fitted_no_mc_scaled = (fitting_results['without_mc']['fitted_curve'] * 
                              (np.max(tic_no_mc) - np.min(tic_no_mc)) + np.min(tic_no_mc))
        ax1.plot(time_arr, fitted_no_mc_scaled, 'r--', linewidth=2,
                label=f"No-MC (Fit, R²={fitting_results['without_mc']['r2']:.3f})")
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('TIC Curves with Lognormal Fits', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[0, 1]
    
    if fitting_results['with_mc']['fitted_curve'] is not None:
        tic_mc_norm = (tic_mc - np.min(tic_mc)) / (np.max(tic_mc) - np.min(tic_mc))
        residuals_mc = tic_mc_norm - fitting_results['with_mc']['fitted_curve']
        ax2.plot(time_arr, residuals_mc, 'g-', linewidth=2, alpha=0.7, label='MC Residuals')
    
    if fitting_results['without_mc']['fitted_curve'] is not None:
        tic_no_mc_norm = (tic_no_mc - np.min(tic_no_mc)) / (np.max(tic_no_mc) - np.min(tic_no_mc))
        residuals_no_mc = tic_no_mc_norm - fitting_results['without_mc']['fitted_curve']
        ax2.plot(time_arr, residuals_no_mc, 'r-', linewidth=2, alpha=0.7, label='No-MC Residuals')
    
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_title('Fitting Residuals', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Quality Metrics
    ax3 = axes[1, 0]
    
    metrics = ['r2', 'correlation']
    metric_names = ['R²', 'Correlation']
    
    mc_values = [fitting_results['with_mc'].get(m, 0) for m in metrics]
    no_mc_values = [fitting_results['without_mc'].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, mc_values, width, label='With MC', color='green', alpha=0.7)
    ax3.bar(x + width/2, no_mc_values, width, label='Without MC', color='red', alpha=0.7)
    
    ax3.set_xlabel('Metric', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Fitting Quality Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "TIC Fitting Summary\n" + "="*40 + "\n\n"
    
    summary_text += "Fitting Quality:\n"
    summary_text += f"  R² (MC):      {fitting_results['with_mc'].get('r2', 0):.4f}\n"
    summary_text += f"  R² (No-MC):   {fitting_results['without_mc'].get('r2', 0):.4f}\n"
    r2_imp = ((fitting_results['with_mc'].get('r2', 0) - fitting_results['without_mc'].get('r2', 0)) / 
             (fitting_results['without_mc'].get('r2', 1) + 1e-10)) * 100
    summary_text += f"  Improvement:  {r2_imp:+.1f}%\n\n"
    
    summary_text += f"  RMSE (MC):    {fitting_results['with_mc'].get('rmse', 0):.4f}\n"
    summary_text += f"  RMSE (No-MC): {fitting_results['without_mc'].get('rmse', 0):.4f}\n"
    rmse_imp = ((fitting_results['without_mc'].get('rmse', 0) - fitting_results['with_mc'].get('rmse', 0)) / 
               (fitting_results['without_mc'].get('rmse', 1) + 1e-10)) * 100
    summary_text += f"  Improvement:  {rmse_imp:+.1f}%\n\n"
    
    summary_text += "TIC Variability:\n"
    summary_text += f"  CV (MC):      {fitting_results['with_mc']['cv']:.4f}\n"
    summary_text += f"  CV (No-MC):   {fitting_results['without_mc']['cv']:.4f}\n"
    cv_imp = ((fitting_results['without_mc']['cv'] - fitting_results['with_mc']['cv']) / 
             (fitting_results['without_mc']['cv'] + 1e-10)) * 100
    summary_text += f"  Improvement:  {cv_imp:+.1f}%\n"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved TIC fitting comparison to {output_path}")
    
    plt.show()
    
    return fig


def generate_comprehensive_report(voi_results, tic_results, output_path='mc_evaluation_report.txt'):
    """Generate comprehensive text report."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("3D MOTION COMPENSATION EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # VOI B-mode Similarity Section
        f.write("1. VOI B-MODE SIMILARITY METRICS\n")
        f.write("-"*70 + "\n")
        f.write("Measuring tissue appearance within VOI\n")
        f.write("(Higher values = VOI tracking same tissue)\n\n")
        
        metrics = ['correlation', 'ssim']
        metric_names = ['Pearson Correlation', 'SSIM']
        
        for metric, name in zip(metrics, metric_names):
            mc_values = voi_results['with_mc'][metric]
            no_mc_values = voi_results['without_mc'][metric]
            
            mean_mc = np.mean(mc_values)
            std_mc = np.std(mc_values)
            mean_no_mc = np.mean(no_mc_values)
            std_no_mc = np.std(no_mc_values)
            
            improvement = ((mean_mc - mean_no_mc) / (mean_no_mc + 1e-10)) * 100
            
            f.write(f"{name}:\n")
            f.write(f"  With MC:    {mean_mc:.4f} ± {std_mc:.4f}\n")
            f.write(f"  Without MC: {mean_no_mc:.4f} ± {std_no_mc:.4f}\n")
            f.write(f"  Improvement: {improvement:+.2f}%\n\n")
        
        # TIC Analysis
        f.write("\n2. TIME-INTENSITY CURVE ANALYSIS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Lognormal Fit Quality:\n")
        f.write(f"  R² (MC):      {tic_results['with_mc'].get('r2', 0):.4f}\n")
        f.write(f"  R² (No-MC):   {tic_results['without_mc'].get('r2', 0):.4f}\n")
        r2_imp = ((tic_results['with_mc'].get('r2', 0) - tic_results['without_mc'].get('r2', 0)) / 
                 (tic_results['without_mc'].get('r2', 1) + 1e-10)) * 100
        f.write(f"  Improvement:  {r2_imp:+.2f}%\n\n")
        
        f.write("TIC Variability:\n")
        f.write(f"  CV (MC):      {tic_results['with_mc']['cv']:.4f}\n")
        f.write(f"  CV (No-MC):   {tic_results['without_mc']['cv']:.4f}\n")
        cv_imp = ((tic_results['without_mc']['cv'] - tic_results['with_mc']['cv']) / 
                 (tic_results['without_mc']['cv'] + 1e-10)) * 100
        f.write(f"  Improvement:  {cv_imp:+.2f}%\n\n")
    
    print(f"\nComprehensive report saved to: {output_path}")