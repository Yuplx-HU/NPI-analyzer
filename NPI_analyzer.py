import os
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def _save_plot(fig: plt.Figure, save_plot_dir: str, save_name: str):
    os.makedirs(save_plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    plot_save_path = os.path.join(save_plot_dir, f"{save_name}-{timestamp}.png")
    fig.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    return plot_save_path


def stern_volmer(F0: list[float], F: list[float], Q: list[float], tau0: float,
                 return_plot: bool = False, save_plot: bool = False, save_plot_dir: str = "output/plots/"):
    """
    Description:
        Perform Stern-Volmer quenching analysis to determine quenching constants.
        
    Required Parameters:
        - F0: Fluorescence intensities without quencher
            - Type: list[float]
            - Purpose: Baseline fluorescence data for quenching calculation
            - Constraints: Must have same length as F and Q
        - F: Fluorescence intensities with quencher
            - Type: list[float]
            - Purpose: Quenched fluorescence data for Stern-Volmer analysis
            - Constraints: Must have same length as F0 and Q
        - Q: Quencher concentrations
            - Type: list[float]
            - Purpose: Concentration values for quenching analysis
            - Constraints: Must be positive values; same length as F0 and F
            - Units: M (mol/L)
        - tau0: Fluorescence lifetime without quencher
            - Type: float
            - Purpose: Used to calculate bimolecular quenching rate constant
            - Constraints: Must be positive
            - Units: ns
    
    Optional Parameters:
        - return_plot: Whether to return the plot figure
            - Default: False
            - Type: bool
            - Purpose: Control whether to include matplotlib figure in return dict
        - save_plot: Whether to save the plot to file
            - Default: False
            - Type: bool
            - Purpose: Control automatic saving of Stern-Volmer plot
        - save_plot_dir: Directory for saving plots
            - Default: "output/plots/"
            - Type: str
            - Purpose: Specify custom directory for plot saving
    
    Returns:
        - Dictionary containing analysis results
            - Type: dict
            - Content: Includes k_SV (Stern-Volmer constant), k_q (bimolecular quenching rate constant), 
                      r_squared (regression quality), std_error (fitting error), plot (if return_plot=True), 
                      and save_plot_path (if save_plot=True)
    """    
    F0, F, Q = map(np.array, [F0, F, Q])
    F0_over_F = F0 / F
    
    slope, intercept, r_value, _, std_error = stats.linregress(Q, F0_over_F)
    k_SV = slope
    k_q = k_SV / tau0
    
    results = {
        "k_SV": k_SV,
        "k_q": k_q,
        "r_squared": r_value ** 2,
        "std_error": std_error,
    }
    
    if return_plot or save_plot:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(Q, F0_over_F, color='blue', alpha=0.7, s=80, label='Data')
        
        Q_fit = np.linspace(Q.min(), Q.max(), 100)
        ax.plot(Q_fit, slope * Q_fit + intercept, 'r-', linewidth=2, 
                label=f'Fit (k_SV = {k_SV:.2e} M⁻¹)')
        
        ax.set(xlabel='[Q] (M)', ylabel='F₀/F', title="Stern-Volmer Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        text_str = f'k_SV = {k_SV:.2e} M⁻¹\nk_q = {k_q:.2e} M⁻¹s⁻¹\nR² = {r_value**2:.4f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if return_plot:
            results["plot"] = fig
        if save_plot:
            results["save_plot_path"] = _save_plot(fig, save_plot_dir, "stern_volmer")
    
    return results


def hill(F0: list[float], F: list[float], Q: list[float],
         return_plot: bool = False, save_plot: bool = False, save_plot_dir: str = "output/plots/"):
    """
    Description:
        Perform Hill plot analysis to determine binding affinity and cooperativity.
        
    Required Parameters:
        - F0: Fluorescence intensities without quencher
            - Type: list[float]
            - Purpose: Baseline fluorescence for Hill transformation
            - Constraints: Must have same length as F and Q
        - F: Fluorescence intensities with quencher
            - Type: list[float]
            - Purpose: Quenched fluorescence for Hill analysis
            - Constraints: Must have same length as F0 and Q
        - Q: Quencher concentrations
            - Type: list[float]
            - Purpose: Concentration values for Hill plot
            - Constraints: Positive values; same length as F0 and F
            - Units: M (mol/L)
    
    Optional Parameters:
        - return_plot: Whether to return the plot figure
            - Default: False
            - Type: bool
            - Purpose: Control whether to include matplotlib figure in return dict
        - save_plot: Whether to save the plot to file
            - Default: False
            - Type: bool
            - Purpose: Control automatic saving of Hill plot
        - save_plot_dir: Directory for saving plots
            - Default: "output/plots/"
            - Type: str
            - Purpose: Specify custom directory for plot saving
    
    Returns:
        - Dictionary containing Hill analysis results
            - Type: dict
            - Content: Includes k_a (association constant), n (Hill coefficient), 
                      r_squared (regression quality), std_error (fitting error), 
                      plot (if return_plot=True), and save_plot_path (if save_plot=True)
    """
    F0, F, Q = map(np.array, [F0, F, Q])
    mask = Q > 0
    Q, F0, F = Q[mask], F0[mask], F[mask]
    
    x = np.log10(Q)
    y = np.log10((F0 - F) / F)
    
    slope, intercept, r_value, _, std_error = stats.linregress(x, y)
    n = slope
    k_a = 10 ** intercept
    
    results = {
        "k_a": k_a,
        "n": n,
        "r_squared": r_value ** 2,
        "std_error": std_error,
    }
    
    if return_plot or save_plot:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(x, y, color='green', alpha=0.7, s=80, label='Data')
        
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_fit, n * x_fit + intercept, 'r-', linewidth=2, 
                label=f'Fit (n = {n:.3f})')
        
        ax.set(xlabel='log[Q]', ylabel='log[(F₀-F)/F]', title="Hill Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        text_str = f'k_a = {k_a:.2e} M⁻¹\nn = {n:.3f}\nR² = {r_value**2:.4f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if return_plot:
            results["plot"] = fig
        if save_plot:
            results["save_plot_path"] = _save_plot(fig, save_plot_dir, "hill")
    
    return results


def _vant_hoff_linear(T: np.ndarray, ka: np.ndarray, R: float):
    inv_T, ln_ka = 1 / T, np.log(ka)
    slope, intercept, r_value, _, std_error = stats.linregress(inv_T, ln_ka)
    
    delta_H = -slope * R / 1000
    delta_S = intercept * R
    
    return {
        'deltaH': delta_H,
        'deltaS': delta_S,
        'deltaG': delta_H - T * delta_S / 1000,
        'T': T,
        'ka': ka,
        'inv_T': inv_T,
        'ln_ka': ln_ka,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'std_error': std_error
    }


def _vant_hoff_quadratic(T: np.ndarray, ka: np.ndarray, R: float):
    def model(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x ** 2
    
    inv_T, ln_ka = 1 / T, np.log(ka)
    popt, pcov = curve_fit(model, inv_T, ln_ka)
    a0, a1, a2 = popt
    
    ln_ka_pred = model(inv_T, *popt)
    r_squared = 1 - np.sum((ln_ka - ln_ka_pred) ** 2) / np.sum((ln_ka - np.mean(ln_ka)) ** 2)
    
    delta_H = -R * (a1 + 2 * a2 / T) / 1000
    delta_S = R * (a0 - a2 / T ** 2)
    
    return {
        'deltaH': delta_H,
        'deltaS': delta_S,
        'deltaG': delta_H - T * delta_S / 1000,
        'T': T,
        'ka': ka,
        'inv_T': inv_T,
        'ln_ka': ln_ka,
        'a0': a0, 'a1': a1, 'a2': a2,
        'r_squared': r_squared,
        'covariance': pcov
    }


def vant_hoff(T: list[float], ka: list[float], R: float = 8.314,
              return_plot: bool = False, save_plot: bool = False, save_plot_dir: str = "output/plots/"):
    """
    Description:
        Perform Van't Hoff analysis to determine thermodynamic parameters from temperature-dependent binding data.
        
    Required Parameters:
        - T: Temperature values
            - Type: list[float]
            - Purpose: Temperature data for thermodynamic analysis
            - Constraints: Must have same length as ka; must be positive
            - Units: K
        - ka: Association constants
            - Type: list[float]
            - Purpose: Binding constants at corresponding temperatures
            - Constraints: Must have same length as T; must be positive
    
    Optional Parameters:
        - R: Universal gas constant
            - Default: 8.314
            - Type: float
            - Purpose: Used in thermodynamic calculations
            - Constraints: Typically 8.314 for SI units
            - Units: J/(mol·K)
        - return_plot: Whether to return the plot figure
            - Default: False
            - Type: bool
            - Purpose: Control whether to include matplotlib figure in return dict
        - save_plot: Whether to save the plot to file
            - Default: False
            - Type: bool
            - Purpose: Control automatic saving of Van't Hoff plot
        - save_plot_dir: Directory for saving plots
            - Default: "output/plots/"
            - Type: str
            - Purpose: Specify custom directory for plot saving
    
    Returns:
        - Dictionary containing thermodynamic analysis results
            - Type: dict
            - Content: Includes deltaH (enthalpy change), deltaS (entropy change), deltaG (Gibbs free energy change),
                      selected_model (linear or quadratic), aic (Akaike Information Criterion), r_squared (regression quality),
                      T, ka, inv_T, ln_ka, plot (if return_plot=True), and save_plot_path (if save_plot=True)
    """
    T, ka = map(np.array, [T, ka])
    
    linear = _vant_hoff_linear(T, ka, R)
    quadratic = _vant_hoff_quadratic(T, ka, R)
    
    n = len(T)
    residuals_linear = linear['ln_ka'] - (linear['slope'] * linear['inv_T'] + linear['intercept'])
    residuals_quad = quadratic['ln_ka'] - (quadratic['a0'] + quadratic['a1'] * quadratic['inv_T'] + quadratic['a2'] * quadratic['inv_T'] ** 2)
    
    aic_linear = n * np.log(np.sum(residuals_linear ** 2) / n) + 4
    aic_quadratic = n * np.log(np.sum(residuals_quad ** 2) / n) + 6
    
    results = quadratic if aic_quadratic < aic_linear else linear
    
    if return_plot or save_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        is_linear = aic_quadratic > aic_linear
        model_label = 'Linear' if is_linear else 'Quadratic'
        
        ax1.scatter(results['inv_T'], results['ln_ka'], color='blue', alpha=0.7, s=80, label='Data')
        inv_T_fit = np.linspace(results['inv_T'].min(), results['inv_T'].max(), 100)
        
        if is_linear:
            ln_ka_fit = results['slope'] * inv_T_fit + results['intercept']
            text_str = f'ΔH = {results["deltaH"][0]:.2f} kJ/mol\nΔS = {results["deltaS"][0]:.2f} J/(mol·K)\nR² = {results["r_squared"]:.4f}'
        else:
            ln_ka_fit = results['a0'] + results['a1'] * inv_T_fit + results['a2'] * inv_T_fit ** 2
            text_str = f'a0 = {results["a0"]:.4f}\na1 = {results["a1"]:.4f}\na2 = {results["a2"]:.4f}\nR² = {results["r_squared"]:.4f}'
        
        ax1.plot(inv_T_fit, ln_ka_fit, 'r-', linewidth=2, label=f'{model_label} Fit')
        ax1.set(xlabel='1/T (K⁻¹)', ylabel='ln(ka)', title="Van't Hoff Plot (Linear)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.scatter(results['T'], results['ka'], color='green', alpha=0.7, s=80, label='Data')
        T_fit = np.linspace(results['T'].min(), results['T'].max(), 100)
        
        if is_linear:
            ka_fit = np.exp(results['slope'] / T_fit + results['intercept'])
        else:
            ka_fit = np.exp(results['a0'] + results['a1'] / T_fit + results['a2'] / T_fit ** 2)
        
        ax2.plot(T_fit, ka_fit, 'r-', linewidth=2, label='Fit')
        ax2.set(xlabel='Temperature (K)', ylabel='ka', title="Temperature Dependence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if return_plot:
            results["plot"] = fig
        if save_plot:
            results["save_plot_path"] = _save_plot(fig, save_plot_dir, "vant_hoff")
    
    return results
