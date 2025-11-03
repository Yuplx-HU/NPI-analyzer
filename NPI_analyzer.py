import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def convert_to_ndarray(data: Any):
    """
    Description:
        Convert various data types to numpy ndarray with automatic type detection and conversion.

    Required Parameters:
        - data: Input data to be converted
            - Type: Any
            - Purpose: Source data that needs to be converted to numpy array format
            - Constraints: Must be convertible to numpy array (list, tuple, number, pandas Series, or existing ndarray)

    Returns:
        - Converted numpy array or None
            - Type: N Dimension Array
            - Purpose: Standardized numpy array representation of input data
            - Constraints: Returns None if input is None, otherwise returns numpy ndarray
    """
    if data is None:
        return None
    
    if isinstance(data, np.ndarray):
        return data
    
    if isinstance(data, list):
        from numbers import Number
        if all(isinstance(x, Number) for x in data):
            return np.array(data, dtype=float)
        else:
            return np.array(data)
    
    if isinstance(data, tuple):
        return np.array(data)
    
    try:
        if isinstance(data, pd.Series):
            return data.values
    except ImportError:
        pass
    
    if isinstance(data, Number):
        return np.array([data])
    
    try:
        return np.asarray(data)
    except Exception as e:
        raise TypeError(f"Can not convert {type(data)} into ndarray: {e}")


def stern_volmer(F0: List[float], F: List[float], Q: List[float], tau0: float = 4.9e-9):
    """
    Description:
        Performs Stern-Volmer quenching analysis to determine quenching constants and type.

    Required Parameters:
        - F0: Fluorescence intensities without quencher
            - Type: Float Array
            - Purpose: Reference fluorescence intensities for quenching calculation
            - Constraints: Must be same length as F and Q arrays
        - F: Fluorescence intensities with quencher
            - Type: Float Array  
            - Purpose: Quenched fluorescence intensities for ratio calculation
            - Constraints: Must be same length as F0 and Q arrays
        - Q: Quencher concentrations
            - Type: Float Array
            - Purpose: Concentrations used for linear regression
            - Constraints: Must be same length as F0 and F arrays

    Optional Parameters:
        - tau0: Fluorescence lifetime without quencher
            - Default: 4.9e-9 (typical value for many fluorophores)
            - Type: Float
            - Purpose: Used to calculate bimolecular quenching constant
            - Constraints: Must be positive, unit: seconds

    Returns:
        - Type: Dictory
        - Purpose: Contains quenching constants, type, and regression metrics
    """
    converted_Q = convert_to_ndarray(Q)
    converted_F0 = convert_to_ndarray(F0)
    converted_F = convert_to_ndarray(F)
    
    F0_over_F = converted_F0 / converted_F
    slope, intercept, r_value, p_value, std_err = stats.linregress(converted_Q, F0_over_F)
    
    k_SV = slope
    k_q = k_SV / tau0
    quenching_type = "static" if k_q > 1e10 else "dynamic"
    
    return {
        'k_SV': k_SV,
        'k_q': k_q,
        'quenching_type': quenching_type,
        
        'Q': converted_Q,
        'slope': slope,
        'F0_over_F': F0_over_F,
        'intercept': intercept,
        
        'metrics': {
            'r_squared': r_value**2,
            'std_err': std_err,
            'p_value': p_value,
        }
    }


def plot_stern_volmer(results: Dict, title: str = "Stern-Volmer Plot", dpi: int = 300, plot_save_dir: str = "plots", plot_save_name: str = "stern_volmer"):
    """
    Description:
        Generates Stern-Volmer quenching plot with linear fit and quenching parameters.

    Required Parameters:
        - results: Stern-Volmer analysis results
            - Type: Dictory
            - Purpose: Contains quenching data and regression results for plotting
            - Constraints: Must contain keys 'Q', 'F0_over_F', 'slope', 'intercept', 'k_SV', 'k_q', and 'metrics'

    Optional Parameters:
        - title: Plot title
            - Default: "Stern-Volmer Plot" (descriptive default for quenching analysis)
            - Type: String
            - Purpose: Title displayed on the plot
        - dpi: Image resolution
            - Default: 300 (high resolution suitable for publications)
            - Type: Integer
            - Purpose: Controls output image quality
            - Constraints: Must be positive integer
        - plot_save_dir: Directory to save plot
            - Default: "plots" (common directory for output files)
            - Type: String
            - Purpose: Location where plot image will be saved
        - plot_save_name: Plot filename
            - Default: "stern_volmer" (descriptive default name)
            - Type: String
            - Purpose: Base filename for saved plot (without extension)

    Returns:
        - Type: matplotlib.figure.Figure
        - Purpose: Figure object containing the Stern-Volmer plot
    """
    if plot_save_dir:
        os.makedirs(plot_save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(results['Q'], results['F0_over_F'], color='blue', alpha=0.7, 
               s=80, label='Experimental Data')
    
    Q_fit = np.linspace(min(results['Q']), max(results['Q']), 100)
    F0_over_F_fit = results['slope'] * Q_fit + results['intercept']
    ax.plot(Q_fit, F0_over_F_fit, 'r-', linewidth=2, 
            label=f'Linear Fit (k_SV = {results["k_SV"]:.2e} M⁻¹)')
    
    ax.set_xlabel('Quencher Concentration [Q] (M)', fontsize=12)
    ax.set_ylabel('F₀/F', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    text_str = (f'k_SV = {results["k_SV"]:.2e} M⁻¹\n'
                f'k_q = {results["k_q"]:.2e} M⁻¹s⁻¹\n'
                f'R² = {results["metrics"]["r_squared"]:.4f}')
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if plot_save_name:
        save_path = Path(plot_save_dir) / f"{plot_save_name}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def hill(F0: List[float], F: List[float], Q: List[float]):
    """
    Description:
        Performs Hill equation analysis to determine binding affinity and cooperativity.

    Required Parameters:
        - F0: Fluorescence intensities without ligand
            - Type: Float Array
            - Purpose: Reference fluorescence intensities for binding calculation
            - Constraints: Must be same length as F and Q arrays, must be positive
        - F: Fluorescence intensities with ligand
            - Type: Float Array
            - Purpose: Ligand-bound fluorescence intensities for ratio calculation
            - Constraints: Must be same length as F0 and Q arrays, must be positive
        - Q: Ligand concentrations
            - Type: Float Array
            - Purpose: Concentrations used for logarithmic transformation and linear regression
            - Constraints: Must be same length as F0 and F arrays, must contain positive values for valid analysis

    Returns:
        - Type: Dictory
        - Purpose: Contains binding constant (k_a), Hill coefficient (n), and regression metrics
    """
    converted_Q = convert_to_ndarray(Q)
    converted_F0 = convert_to_ndarray(F0)
    converted_F = convert_to_ndarray(F)
    
    mask = converted_Q > 0
    Q_valid = converted_Q[mask]
    F0_valid = converted_F0[mask]
    F_valid = converted_F[mask]
    
    x = np.log10(Q_valid)
    y = np.log10((F0_valid - F_valid) / F_valid)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    k_a = 10 ** intercept
    n = slope
    
    return {
        'k_a': k_a,
        'n': n,
        
        'log_x': x,
        'log_y': y,
        'intercept': intercept,
        
        'metrics': {
            'r_squared': r_value**2,
            'std_err': std_err,
            'p_value': p_value,
        }
    }


def plot_hill(results: Dict, title: str = "Hill Plot", dpi: int = 300, plot_save_dir: str = "plots", plot_save_name: str = "hill"):
    """
    Description:
        Generates Hill plot with linear fit to visualize binding affinity and cooperativity.

    Required Parameters:
        - results: Hill equation analysis results
            - Type: Dictory
            - Purpose: Contains logarithmic data and regression results for plotting
            - Constraints: Must contain keys 'log_x', 'log_y', 'n', 'intercept', 'k_a', and 'metrics'

    Optional Parameters:
        - title: Plot title
            - Default: "Hill Plot" (standard name for this type of analysis)
            - Type: String
            - Purpose: Title displayed on the plot
        - dpi: Image resolution
            - Default: 300 (high resolution suitable for publications)
            - Type: Integer
            - Purpose: Controls output image quality
            - Constraints: Must be positive integer
        - plot_save_dir: Directory to save plot
            - Default: "plots" (common directory for output files)
            - Type: String
            - Purpose: Location where plot image will be saved
        - plot_save_name: Plot filename
            - Default: "hill" (descriptive default name)
            - Type: String
            - Purpose: Base filename for saved plot (without extension)

    Returns:
        - Type: matplotlib.figure.Figure
        - Purpose: Figure object containing the Hill plot
    """
    if plot_save_dir:
        os.makedirs(plot_save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(results['log_x'], results['log_y'], color='green', alpha=0.7, 
               s=80, label='Experimental Data')
    
    x_fit = np.linspace(min(results['log_x']), max(results['log_x']), 100)
    y_fit = results['n'] * x_fit + results['intercept']
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
            label=f'Linear Fit (n = {results["n"]:.3f})')
    
    ax.set_xlabel('log[Q]', fontsize=12)
    ax.set_ylabel('log[(F₀-F)/F]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    text_str = (f'k_a = {results["k_a"]:.2e} M⁻¹\n'
                f'n = {results["n"]:.3f}\n'
                f'R² = {results["metrics"]["r_squared"]:.4f}')
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if plot_save_name:
        save_path = Path(plot_save_dir) / f"{plot_save_name}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def _vant_hoff_linear(T: List[float], ka: List[float], R: float):
    """
    Description:
        Performs linear van't Hoff analysis to calculate thermodynamic parameters from temperature-dependent binding data.

    Required Parameters:
        - T: Absolute temperatures
            - Type: Float Array
            - Purpose: Temperature values for van't Hoff analysis
            - Constraints: Must be same length as ka array, must be positive, unit: K
        - ka: Binding constants
            - Type: Float Array
            - Purpose: Association constants at corresponding temperatures
            - Constraints: Must be same length as T array, must be positive
        - R: Gas constant
            - Type: Float
            - Purpose: Used to calculate thermodynamic parameters
            - Constraints: Must be positive, unit: J·mol⁻¹·K⁻¹

    Returns:
        - Type: Dictory
        - Purpose: Contains ΔH, ΔS, ΔG thermodynamic parameters and regression data
    """
    converted_T = convert_to_ndarray(T)
    converted_ka = convert_to_ndarray(ka)
    
    inv_T = 1 / converted_T
    ln_ka = np.log(converted_ka)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_ka)
    
    delta_H = -slope * R
    delta_S = intercept * R
    delta_G = delta_H - converted_T * delta_S
    
    return {
        'delta_H': delta_H,
        'delta_S': delta_S,
        'delta_G': delta_G,
        
        'T': converted_T,
        'ka': converted_ka,
        'inv_T': inv_T,
        'ln_ka': ln_ka,
        'slope': slope,
        'intercept': intercept,
        
        'metrics': {
            'r_squared': r_value**2,
            'std_err': std_err,
            'p_value': p_value,
        }
    }


def _vant_hoff_quadratic(T: List[float], ka: List[float], R: float):
    """
    Description:
        Performs quadratic van't Hoff analysis to account for temperature-dependent heat capacity changes.

    Required Parameters:
        - T: Absolute temperatures
            - Type: Float Array
            - Purpose: Temperature values for van't Hoff analysis
            - Constraints: Must be same length as ka array, must be positive, unit: K
        - ka: Binding constants
            - Type: Float Array
            - Purpose: Association constants at corresponding temperatures
            - Constraints: Must be same length as T array, must be positive
        - R: Gas constant
            - Type: Float
            - Purpose: Used to calculate thermodynamic parameters
            - Constraints: Must be positive, unit: J·mol⁻¹·K⁻¹

    Returns:
        - Type: Dictory
        - Purpose: Contains temperature-dependent ΔH, ΔS, ΔG parameters with quadratic fitting
    """
    def quadratic_model(x, a0, a1, a2):
        return a0 + a1*x + a2*x**2
    
    converted_T = convert_to_ndarray(T)
    converted_ka = convert_to_ndarray(ka)
    
    inv_T = 1 / T
    ln_ka = np.log(converted_ka)
    
    popt, pcov = curve_fit(quadratic_model, inv_T, ln_ka)
    a0, a1, a2 = popt
    
    delta_H = -R * (a1 + 2 * a2 / converted_T)
    delta_S = R * (a0 - a2 / converted_T**2)
    delta_G = delta_H - converted_T * delta_S
    
    ln_ka_pred = quadratic_model(inv_T, a0, a1, a2)
    r_squared = 1 - np.sum((ln_ka - ln_ka_pred)** 2) / np.sum((ln_ka - np.mean(ln_ka))**2)
    
    return {
        'delta_H': delta_H,
        'delta_S': delta_S,
        'delta_G': delta_G,
        
        'T': converted_T,
        'ka': converted_ka,
        'inv_T': inv_T,
        'ln_ka': ln_ka,
        'a0': a0,
        'a1': a1,
        'a2': a2,
        
        'metrics': {
            'r_squared': r_squared,
            'covariance': pcov,
        }
    }


def vant_hoff(T: List[float], ka: List[float], R: float = 8.314):
    """
    Description:
        Performs van't Hoff analysis with automatic model selection (linear vs quadratic) using AIC criterion to calculate thermodynamic constants.

    Required Parameters:
        - T: Absolute temperatures
            - Type: Float Array
            - Purpose: Temperature values for van't Hoff analysis
            - Constraints: Must be same length as ka array, must be positive, unit: K
        - ka: Binding constants
            - Type: Float Array
            - Purpose: Association constants at corresponding temperatures
            - Constraints: Must be same length as T array, must be positive

    Optional Parameters:
        - R: Gas constant
            - Default: 8.314 (standard value in SI units)
            - Type: Float
            - Purpose: Used to calculate thermodynamic parameters
            - Constraints: Must be positive, unit: J·mol⁻¹·K⁻¹

    Returns:
        - Type: Dictory
        - Purpose: Contains thermodynamic parameters and model selection results
    """
    converted_T = convert_to_ndarray(T)
    converted_ka = convert_to_ndarray(ka)
    
    results_linear = _vant_hoff_linear(converted_T, converted_ka, R=R)
    results_quadratic = _vant_hoff_quadratic(converted_T, converted_ka, R=R)
    
    n = len(converted_T)
    k_linear = 2
    k_quadratic = 3
    
    residuals_linear = results_linear['ln_ka'] - (results_linear['slope'] * results_linear['inv_T'] + results_linear['intercept'])
    aic_linear = n * np.log(np.sum(residuals_linear**2) / n) + 2 * k_linear
    
    residuals_quad = results_quadratic['ln_ka'] - (results_quadratic['a0'] + results_quadratic['a1'] * results_quadratic['inv_T'] + results_quadratic['a2'] * results_quadratic['inv_T']**2)
    aic_quadratic = n * np.log(np.sum(residuals_quad**2) / n) + 2 * k_quadratic
    
    if aic_quadratic < aic_linear:
        results_quadratic['selected_model'] = 'quadratic'
        results_quadratic['aic'] = aic_quadratic
        results_quadratic['aic_linear'] = aic_linear
        return results_quadratic
    else:
        results_linear['selected_model'] = 'linear'
        results_linear['aic'] = aic_linear
        results_linear['aic_quadratic'] = aic_quadratic
        return results_linear


def _plot_vant_hoff_linear(results: Dict, title: str, dpi: int, plot_save_dir: str, plot_save_name: str):
    """
    Description:
        Generates linear van't Hoff plot showing both inverse temperature and temperature dependence.

    Required Parameters:
        - results: Linear van't Hoff analysis results
            - Type: Dictory
            - Purpose: Contains thermodynamic parameters and regression data for plotting
            - Constraints: Must contain keys from linear van't Hoff analysis
        - title: Plot title
            - Type: String
            - Purpose: Title displayed on the plot
        - dpi: Image resolution
            - Type: Integer
            - Purpose: Controls output image quality
            - Constraints: Must be positive integer
        - plot_save_dir: Directory to save plot
            - Type: String
            - Purpose: Location where plot image will be saved
        - plot_save_name: Plot filename
            - Type: String
            - Purpose: Base filename for saved plot

    Returns:
        - Type: matplotlib.figure.Figure
        - Purpose: Figure object containing the linear van't Hoff plot
    """
    if plot_save_dir:
        os.makedirs(plot_save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(results['inv_T'], results['ln_ka'], color='blue', alpha=0.7, 
                s=80, label='Experimental Data')
    inv_T_fit = np.linspace(min(results['inv_T']), max(results['inv_T']), 100)
    ln_ka_fit = results['slope'] * inv_T_fit + results['intercept']
    ax1.plot(inv_T_fit, ln_ka_fit, 'r-', linewidth=2, label='Linear Fit')
    ax1.set_xlabel('1/T (K⁻¹)')
    ax1.set_ylabel('ln(ka)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    text_str = (f'ΔH = {results["delta_H"]:.2f} J/mol\n'
                f'ΔS = {results["delta_S"]:.2f} J/(mol·K)\n'
                f'R² = {results["metrics"]["r_squared"]:.4f}')
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.scatter(results['T'], results['ka'], color='green', alpha=0.7, 
                s=80, label='Experimental Data')
    T_fit = np.linspace(min(results['T']), max(results['T']), 100)
    ka_fit = np.exp(results['slope']/T_fit + results['intercept'])
    ax2.plot(T_fit, ka_fit, 'r-', linewidth=2, label='Fit')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('ka')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if plot_save_name:
        save_path = Path(plot_save_dir) / f"{plot_save_name}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def _plot_vant_hoff_quadratic(results: Dict, title: str, dpi: int, plot_save_dir: str, plot_save_name: str):
    """
    Description:
        Generates quadratic van't Hoff plot accounting for temperature-dependent heat capacity.

    Required Parameters:
        - results: Quadratic van't Hoff analysis results
            - Type: Dictory
            - Purpose: Contains thermodynamic parameters and quadratic fitting data for plotting
            - Constraints: Must contain keys from quadratic van't Hoff analysis
        - title: Plot title
            - Type: String
            - Purpose: Title displayed on the plot
        - dpi: Image resolution
            - Type: Integer
            - Purpose: Controls output image quality
            - Constraints: Must be positive integer
        - plot_save_dir: Directory to save plot
            - Type: String
            - Purpose: Location where plot image will be saved
        - plot_save_name: Plot filename
            - Type: String
            - Purpose: Base filename for saved plot

    Returns:
        - Type: matplotlib.figure.Figure
        - Purpose: Figure object containing the quadratic van't Hoff plot
    """
    if plot_save_dir:
        os.makedirs(plot_save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    a0, a1, a2 = results['a0'], results['a1'], results['a2']
    
    ax1.scatter(results['inv_T'], results['ln_ka'], color='blue', alpha=0.7, 
                s=80, label='Experimental Data')
    inv_T_fit = np.linspace(min(results['inv_T']), max(results['inv_T']), 100)
    ln_ka_fit = a0 + a1 * inv_T_fit + a2 * inv_T_fit**2
    ax1.plot(inv_T_fit, ln_ka_fit, 'r-', linewidth=2, label='Quadratic Fit')
    ax1.set_xlabel('1/T (K⁻¹)')
    ax1.set_ylabel('ln(ka)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    text_str = (f'a0 = {a0:.4f}\na1 = {a1:.4f}\na2 = {a2:.4f}\n'
                f'R² = {results["metrics"]["r_squared"]:.4f}')
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.scatter(results['T'], results['ka'], color='green', alpha=0.7, 
                s=80, label='Experimental Data')
    T_fit = np.linspace(min(results['T']), max(results['T']), 100)
    ka_fit = np.exp(a0 + a1/T_fit + a2/T_fit**2)
    ax2.plot(T_fit, ka_fit, 'r-', linewidth=2, label='Fit')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('ka')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if plot_save_name:
        save_path = Path(plot_save_dir) / f"{plot_save_name}.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_vant_hoff(results: Dict, title: str = "Van't Hoff Plot", dpi: int = 300, plot_save_dir: str = "plots", plot_save_name: str = "vant_hoff"):
    """
    Description:
        Generates van't Hoff plot with automatic selection between linear and quadratic models.

    Required Parameters:
        - results: van't Hoff analysis results
            - Type: Dictory
            - Purpose: Contains thermodynamic parameters and model selection results
            - Constraints: Must contain 'selected_model' key to determine plot type

    Optional Parameters:
        - title: Plot title
            - Default: "Van't Hoff Plot" (standard name for thermodynamic analysis)
            - Type: String
            - Purpose: Title displayed on the plot
        - dpi: Image resolution
            - Default: 300 (high resolution suitable for publications)
            - Type: Integer
            - Purpose: Controls output image quality
            - Constraints: Must be positive integer
        - plot_save_dir: Directory to save plot
            - Default: "plots" (common directory for output files)
            - Type: String
            - Purpose: Location where plot image will be saved
        - plot_save_name: Plot filename
            - Default: "vant_hoff" (descriptive default name)
            - Type: String
            - Purpose: Base filename for saved plot (without extension)

    Returns:
        - Type: matplotlib.figure.Figure
        - Purpose: Figure object containing the van't Hoff plot
    """
    if results['selected_model'] == 'linear':
        return _plot_vant_hoff_linear(results, title, dpi, plot_save_dir, plot_save_name)
    else:
        return _plot_vant_hoff_quadratic(results, title, dpi, plot_save_dir, plot_save_name)
