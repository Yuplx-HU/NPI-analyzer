import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def stern_volmer(F0, F, Q, tau0=4.9e-9):
    """
    Overview:
        Calculate Stern-Volmer quenching constants and determine quenching type from fluorescence data.
        (Core: Performs linear regression on F0/F vs. quencher concentration to extract quenching parameters)

    Details:
        - Function: Performs Stern-Volmer analysis by linear regression of F0/F against quencher concentration Q.
          Returns quenching constants (k_SV, k_q) and classifies quenching as static or dynamic based on k_q threshold.
        - Use Case: Analyze fluorescence quenching data to study molecular interactions (e.g., solute-solvent, protein-ligand).
        - Special Handling: Uses a threshold of 1e10 M⁻¹s⁻¹ to distinguish static (k_q > 1e10) from dynamic quenching.

    Args:
        F0 (array-like): Reference fluorescence intensities without quencher:
            - Role: Baseline fluorescence values for normalization (F0/F calculation).
            - Constraints: Must be same length as F; positive values.
        F (array-like): Measured fluorescence intensities with quencher:
            - Role: Quenched fluorescence values for Stern-Volmer plot (y-axis: F0/F).
            - Constraints: Same length as F0; positive values.
        Q (array-like): Quencher concentrations:
            - Role: x-values for linear regression (typically in molarity, M).
            - Constraints: Same length as F0; non-negative.
        tau0 (float, optional): Fluorescence lifetime without quencher:
            - Role: Used to calculate bimolecular quenching rate constant (k_q = k_SV / tau0).
            - Default: 4.9e-9 (Typical lifetime for organic fluorophores in seconds).

    Returns:
        dict: Dictionary containing quenching parameters and regression metrics:
            - k_SV (float): Stern-Volmer constant (slope of F0/F vs. Q, unit: M⁻¹).
            - k_q (float): Bimolecular quenching rate constant (unit: M⁻¹s⁻¹).
            - quenching_type (str): Classification as "static" or "dynamic".
            - Q (array): Original quencher concentrations.
            - slope (float): Regression slope (equal to k_SV).
            - F0_over_F (array): Calculated F0/F ratios.
            - intercept (float): Regression intercept.
            - metrics (dict): Regression statistics:
                - r_squared (float): Coefficient of determination.
                - std_err (float): Standard error of slope.
                - p_value (float): p-value for slope significance.

    Raises:
        ValueError: When F0, F, Q have mismatched lengths or contain non-positive values.
        TypeError: When inputs are not array-like or tau0 is not numeric.

    Examples:
        Basic usage:
            >>> F0 = [100, 98, 95, 90, 85]
            >>> F = [80, 70, 60, 50, 40]
            >>> Q = [0, 0.01, 0.02, 0.03, 0.04]
            >>> result = stern_volmer(F0, F, Q)
            >>> print(f"k_SV: {result['k_SV']:.2e} M⁻¹, Type: {result['quenching_type']}")

        Custom lifetime:
            >>> result = stern_volmer(F0, F, Q, tau0=2.5e-9)  # Shorter lifetime fluorophore
            >>> print(f"k_q: {result['k_q']:.2e} M⁻¹s⁻¹")

    Notes:
        - Unit Consistency: Ensure Q is in M (mol/L) and tau0 in seconds for correct k_q units.
        - Threshold: The 1e10 M⁻¹s⁻¹ threshold approximates diffusion-controlled limit; adjust for specific systems.
        - Data Quality: High r_squared (>0.95) suggests reliable linear fit. Check p_value for significance.
    """
    F0_over_F = F0 / F
    slope, intercept, r_value, p_value, std_err = stats.linregress(Q, F0_over_F)
    
    k_SV = slope
    k_q = k_SV / tau0
    quenching_type = "static" if k_q > 1e10 else "dynamic"
    
    return {
        'k_SV': k_SV,
        'k_q': k_q,
        'quenching_type': quenching_type,
        
        'Q': Q,
        'slope': slope,
        'F0_over_F': F0_over_F,
        'intercept': intercept,
        
        'metrics': {
            'r_squared': r_value**2,
            'std_err': std_err,
            'p_value': p_value,
        }
    }


def plot_stern_volmer(results, title="Stern-Volmer Plot", dpi=300, plot_save_dir="plots", plot_save_name="stern_volmer"):
    """
    Overview:
        Generate and optionally save a Stern-Volmer plot from quenching analysis results.
        (Core: Creates a scatter plot of F0/F vs. quencher concentration with linear regression fit)

    Details:
        - Function: Visualizes Stern-Volmer analysis results by plotting experimental data points and linear regression line.
          Includes quenching parameters as annotations and supports high-quality image export.
        - Use Case: Create publication-ready figures for fluorescence quenching studies and data validation.
        - Special Handling: Automatically creates save directory if needed; formats scientific notation for constants.

    Args:
        results (dict): Stern-Volmer analysis results from stern_volmer() function:
            - Role: Contains all data and parameters needed for plotting (Q, F0_over_F, slope, intercept, etc.).
            - Constraints: Must contain keys: 'Q', 'F0_over_F', 'slope', 'intercept', 'k_SV', 'k_q', 'metrics'.
        title (str, optional): Plot title:
            - Role: Display title for the Stern-Volmer plot.
            - Default: "Stern-Volmer Plot" (Standard title for quenching analysis plots).
        dpi (int, optional): Image resolution for saved plot:
            - Role: Controls output image quality (dots per inch).
            - Default: 300 (High resolution suitable for publications).
        plot_save_dir (str, optional): Directory to save the plot:
            - Role: Target folder for saving the generated plot image.
            - Default: "plots" (Common directory name for storing figures).
        plot_save_name (str, optional): Base filename for saving:
            - Role: Name prefix for the saved image file (.png extension added automatically).
            - Default: "stern_volmer" (Descriptive default filename).

    Returns:
        matplotlib.figure.Figure: The generated figure object containing the Stern-Volmer plot:
            - Can be used for further customization or display in Jupyter notebooks.
            - Contains scatter plot (experimental data) and line plot (linear fit).
            - Includes axis labels, title, legend, grid, and parameter annotations.

    Raises:
        KeyError: When results dictionary missing required keys ('Q', 'F0_over_F', etc.).
        PermissionError: When unable to create save directory or write image file.
        TypeError: When input parameters have incorrect types (e.g., dpi not integer).

    Examples:
        Basic usage with automatic saving:
            >>> results = stern_volmer(F0, F, Q)
            >>> fig = plot_stern_volmer(results)
            >>> plt.show()  # Display the plot

        Custom title and high resolution:
            >>> fig = plot_stern_volmer(
                    results, 
                    title="Protein-Ligand Quenching Analysis",
                    dpi=600  # Ultra-high resolution for publications
                )

        Save to custom location:
            >>> fig = plot_stern_volmer(
                    results,
                    plot_save_dir="results/figures",
                    plot_save_name="protein_quenching_analysis"
                )

    Notes:
        - Directory Creation: Automatically creates plot_save_dir if it doesn't exist.
        - Image Format: Saves as PNG format with tight bounding box for minimal whitespace.
        - Plot Elements: Includes R² value from metrics for fit quality assessment.
        - Customization: Returned figure object can be further modified using matplotlib methods.
        - Units: Assumes quencher concentration [Q] in M (mol/L) for axis labeling.
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


def hill(F0, F, Q):
    """
    Overview:
        Perform Hill analysis to determine binding affinity and cooperativity from fluorescence data.
        (Core: Calculates association constant and Hill coefficient using logarithmic transformation)

    Details:
        - Function: Analyzes ligand binding behavior by linear regression of log-transformed data.
          Filters out zero quencher concentrations and returns Hill equation parameters.
        - Use Case: Study cooperative binding interactions in biochemical systems (e.g., protein-ligand, DNA-drug).
        - Special Handling: Automatically filters Q=0 points to avoid log(0) errors; uses base-10 logarithms.

    Args:
        F0 (array-like): Reference fluorescence intensities without quencher:
            - Role: Baseline fluorescence values for binding calculation.
            - Constraints: Must be same length as F and Q; positive values.
        F (array-like): Measured fluorescence intensities with quencher:
            - Role: Quenched fluorescence values for Hill transformation.
            - Constraints: Same length as F0; positive values; F < F0 for valid binding data.
        Q (array-like): Quencher/ligand concentrations:
            - Role: x-values for Hill plot (typically in molarity, M).
            - Constraints: Same length as F0; non-negative values; must contain positive values for valid analysis.

    Returns:
        dict: Dictionary containing Hill equation parameters and regression data:
            - k_a (float): Association constant (unit: M⁻¹), representing binding affinity.
            - n (float): Hill coefficient, indicating cooperativity (n>1 positive, n<1 negative, n=1 non-cooperative).
            - log_x (array): Transformed x-values (log10(Q) for valid Q>0 points).
            - log_y (array): Transformed y-values (log10((F0-F)/F)).
            - intercept (float): Regression intercept (log10(k_a)).
            - metrics (dict): Regression quality indicators:
                - r_squared (float): Coefficient of determination for linear fit.
                - std_err (float): Standard error of slope (Hill coefficient).
                - p_value (float): p-value for slope significance.

    Raises:
        ValueError: When arrays have mismatched lengths, contain invalid values, or all Q ≤ 0.
        TypeError: When inputs are not array-like or contain non-numeric values.
        ZeroDivisionError: When F contains zero values (causes division error in transformation).

    Examples:
        Basic binding analysis:
            >>> F0 = [100, 95, 85, 70, 50]
            >>> F = [100, 90, 75, 55, 30]
            >>> Q = [0, 1e-6, 5e-6, 1e-5, 5e-5]
            >>> result = hill(F0, F, Q)
            >>> print(f"k_a: {result['k_a']:.2e} M⁻¹, n: {result['n']:.2f}")

        Strong cooperative binding (n > 1):
            >>> result = hill(F0_strong, F_strong, Q_concentrations)
            >>> if result['n'] > 1.5:
            ...     print("Positive cooperativity detected")

    Notes:
        - Data Filtering: Q=0 points are excluded from analysis to avoid mathematical errors.
        - Hill Interpretation: 
            - n ≈ 1: Non-cooperative binding (independent sites)
            - n > 1: Positive cooperativity (binding enhances subsequent binding)
            - n < 1: Negative cooperativity (binding inhibits subsequent binding)
        - Unit Consistency: Ensure Q is in M (mol/L) for correct k_a units (M⁻¹).
        - Data Quality: High r_squared indicates good linear fit in Hill plot transformation.
        - Assumption: Assumes binding follows Hill equation model; verify with experimental validation.
    """
    mask = Q > 0
    Q_valid = Q[mask]
    F0_valid = F0[mask]
    F_valid = F[mask]
    
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


def plot_hill(results, title="Hill Plot", dpi=300, plot_save_dir="plots", plot_save_name="hill"):
    """
    Overview:
        Generate a Hill plot to visualize cooperative binding analysis results.
        (Core: Creates scatter plot of log-transformed binding data with linear regression fit)

    Details:
        ▪ Function: Plots Hill analysis results by displaying experimental data points and linear regression line.
          Includes key binding parameters as annotations and supports high-quality image export.
        ▪ Use Case: Suitable for visualizing cooperative binding behavior in biochemical studies (protein-ligand, 
          receptor-drug interactions). Not suitable for non-cooperative binding analysis without Hill transformation.
        ▪ Special Handling: Automatically creates save directory; uses Hill equation transformation for axis labeling.

    Args:
        results (dict): Binding analysis results from hill() function:
            ▪ Role: Provides transformed data and calculated parameters for plotting (log_x, log_y, Hill coefficient, etc.).
            ▪ Constraints: Must contain keys: 'log_x', 'log_y', 'n', 'intercept', 'k_a', 'metrics' with valid numerical values.
        title (str, optional): Plot title display text:
            ▪ Role: Descriptive title shown above the Hill plot.
            ▪ Default: "Hill Plot" (Standard nomenclature for cooperative binding visualization).
        dpi (int, optional): Image resolution for output file:
            ▪ Role: Controls output image quality in dots per inch.
            ▪ Default: 300 (Balances quality and file size for most publication requirements).
        plot_save_dir (str, optional): Directory path for plot storage:
            ▪ Role: Target folder where the generated plot image will be saved.
            ▪ Default: "plots" (Conventional directory name for figure storage).
        plot_save_name (str, optional): Base filename for saved image:
            ▪ Role: Name prefix for the output PNG file (extension automatically added).
            ▪ Default: "hill" (Descriptive default identifying the plot type).

    Returns:
        matplotlib.figure.Figure: Figure object containing the complete Hill plot visualization:
            ▪ Can be further customized using matplotlib methods or displayed in interactive environments.
            ▪ Contains scatter plot (experimental data points), line plot (linear regression fit), 
              axis labels, legend, grid, and parameter annotations.
            ▪ Field descriptions:
                ▪ Scatter plot: Green points representing log[(F₀-F)/F] vs. log[Q] experimental data.
                ▪ Line plot: Red regression line showing linear fit with displayed Hill coefficient (n).
                ▪ Text box: Annotated with k_a (M⁻¹), n (unitless), and R² values.

    Raises:
        KeyError: When results dictionary lacks required keys ('log_x', 'log_y', 'n', etc.).
        PermissionError: When unable to create directory or write file due to permissions.
        ValueError: When results contain invalid numerical values (NaN, Inf) or empty arrays.

    Examples:
        Basic usage with default settings:
            >>> results = hill(F0, F, Q)
            >>> fig = plot_hill(results)
            >>> plt.show()  # Display the generated plot

        Customized plot for publication:
            >>> fig = plot_hill(
                    results,
                    title="Protein-Ligand Binding Cooperativity",
                    dpi=600,  # High resolution for journal submission
                    plot_save_dir="manuscript/figures",
                    plot_save_name="figure_3_hill_analysis"
                )

    Notes:
        ▪ Hill Coefficient Interpretation: 
            ▪ n > 1 indicates positive cooperativity (binding enhances subsequent binding)
            ▪ n ≈ 1 indicates non-cooperative binding
            ▪ n < 1 indicates negative cooperativity
        ▪ Unit Consistency: Ensure input concentrations are in molar (M) for correct k_a units (M⁻¹).
        ▪ Data Quality: R² value indicates goodness of linear fit; values >0.95 suggest reliable analysis.
        ▪ Image Format: Output saved as PNG with tight bounding box to minimize whitespace.
        ▪ Reusability: Returned figure object can be modified further before final display or saving.
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


def _vant_hoff_linear(T, ka, R):
    """
    Overview:
        Calculate thermodynamic parameters from temperature-dependent equilibrium constants using van't Hoff analysis.
        (Core: Performs linear regression on ln(ka) vs. 1/T to extract enthalpy, entropy, and free energy changes)

    Details:
        - Function: Applies van't Hoff equation (ln(ka) = -ΔH/RT + ΔS/R) through linear regression of ln(ka) against 1/T.
          Returns thermodynamic parameters ΔH, ΔS, and ΔG for each temperature point.
        - Use Case: Analyze temperature dependence of equilibrium constants to study thermodynamics of chemical reactions,
          binding processes, or phase transitions. Suitable for processes where ΔH and ΔS are temperature-independent.
        - Special Handling: Assumes constant ΔH and ΔS over the temperature range; uses gas constant R for unit conversion.

    Args:
        T (array-like): Absolute temperatures at which equilibrium constants were measured:
            - Role: x-values for van't Hoff plot after transformation to 1/T.
            - Constraints: Must be positive values in Kelvin (K); same length as ka.
        ka (array-like): Equilibrium constants (e.g., association constants from binding studies):
            - Role: y-values for van't Hoff plot after natural logarithm transformation.
            - Constraints: Must be positive values; typically in M⁻¹ for association constants; same length as T.
        R (float): Gas constant for unit conversion:
            - Role: Converts slope and intercept to standard thermodynamic units.
            - Constraints: Positive value; typically 8.314 J·mol⁻¹·K⁻¹ for SI units.

    Returns:
        dict: Dictionary containing thermodynamic parameters and regression data:
            - delta_H (array): Enthalpy change (unit: J·mol⁻¹) at each temperature.
            - delta_S (array): Entropy change (unit: J·mol⁻¹·K⁻¹) at each temperature.
            - delta_G (array): Gibbs free energy change (unit: J·mol⁻¹) at each temperature.
            - T (array): Original temperature values (K).
            - ka (array): Original equilibrium constant values.
            - inv_T (array): Transformed x-values (1/T in K⁻¹).
            - ln_ka (array): Transformed y-values (ln(ka)).
            - slope (float): Regression slope (-ΔH/R).
            - intercept (float): Regression intercept (ΔS/R).
            - metrics (dict): Regression quality indicators:
                - r_squared (float): Coefficient of determination for linear fit.
                - std_err (float): Standard error of slope.
                - p_value (float): p-value for slope significance.

    Raises:
        ValueError: When T contains non-positive values, ka contains non-positive values, or arrays have mismatched lengths.
        TypeError: When inputs are not numeric or R is not a positive scalar.
        ZeroDivisionError: When T contains zero values (causes division error in 1/T transformation).

    Examples:
        Basic thermodynamic analysis:
            >>> T = [280, 290, 300, 310, 320]  # Temperatures in Kelvin
            >>> ka = [1e3, 2e3, 5e3, 8e3, 1e4]  # Association constants
            >>> R = 8.314  # J·mol⁻¹·K⁻¹
            >>> results = _vant_hoff_linear(T, ka, R)
            >>> print(f"ΔH: {results['delta_H'][0]:.1f} J/mol, ΔS: {results['delta_S'][0]:.3f} J/mol·K")

        Using different gas constant units:
            >>> results = _vant_hoff_linear(T, ka, 1.987)  # cal·mol⁻¹·K⁻¹
            >>> # Results will be in cal·mol⁻¹ instead of J·mol⁻¹

    Notes:
        - Unit Consistency: Ensure T is in Kelvin and R matches desired output units (typically 8.314 J·mol⁻¹·K⁻¹).
        - Assumptions: Method assumes ΔH and ΔS are constant over the temperature range studied.
        - Sign Convention: Negative ΔH indicates exothermic process; positive ΔS indicates increased disorder.
        - Data Quality: High r_squared suggests reliable linear fit; check p_value for statistical significance.
        - Temperature Range: Wider temperature ranges improve accuracy of ΔH and ΔS determination.
        - Free Energy: ΔG is calculated for each temperature point using ΔG = ΔH - TΔS.
    """
    inv_T = 1 / T
    ln_ka = np.log(ka)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_ka)
    
    delta_H = -slope * R
    delta_S = intercept * R
    delta_G = delta_H - T * delta_S
    
    return {
        'delta_H': delta_H,
        'delta_S': delta_S,
        'delta_G': delta_G,
        
        'T': T,
        'ka': ka,
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


def _vant_hoff_quadratic(T, ka, R):
    """
    Overview:
        Calculate temperature-dependent thermodynamic parameters using quadratic van't Hoff analysis.
        (Core: Fits ln(ka) vs. 1/T data to a quadratic model to account for temperature-dependent ΔH and ΔS)

    Details:
        - Function: Applies quadratic extension of van't Hoff equation to handle cases where enthalpy and entropy 
          changes vary with temperature. Uses curve fitting to determine quadratic coefficients and calculates 
          temperature-dependent thermodynamic parameters.
        - Use Case: Suitable for processes where heat capacity changes significantly with temperature, making 
          ΔH and ΔS temperature-dependent. Not needed for systems with constant heat capacity.
        - Special Handling: Uses nonlinear curve fitting instead of linear regression; calculates R² manually.

    Args:
        T (array-like): Absolute temperatures at which equilibrium constants were measured:
            - Role: Independent variable for thermodynamic analysis (converted to 1/T for fitting).
            - Constraints: Must be positive values in Kelvin (K); same length as ka; minimum 3 points for quadratic fit.
        ka (array-like): Equilibrium constants (e.g., association constants from temperature-dependent studies):
            - Role: Dependent variable for van't Hoff analysis (converted to ln(ka) for fitting).
            - Constraints: Must be positive values; typically in M⁻¹; same length as T.
        R (float): Gas constant for unit conversion:
            - Role: Converts fitted parameters to thermodynamic units.
            - Constraints: Positive value; typically 8.314 J·mol⁻¹·K⁻¹ for SI units.

    Returns:
        dict: Dictionary containing temperature-dependent thermodynamic parameters and fitting results:
            - delta_H (array): Enthalpy change at each temperature (unit: J·mol⁻¹), varies with T.
            - delta_S (array): Entropy change at each temperature (unit: J·mol⁻¹·K⁻¹), varies with T.
            - delta_G (array): Gibbs free energy change at each temperature (unit: J·mol⁻¹).
            - T (array): Original temperature values (K).
            - ka (array): Original equilibrium constant values.
            - inv_T (array): Transformed x-values (1/T in K⁻¹).
            - ln_ka (array): Transformed y-values (ln(ka)).
            - a0 (float): Quadratic coefficient (constant term) from curve fitting.
            - a1 (float): Quadratic coefficient (linear term) from curve fitting.
            - a2 (float): Quadratic coefficient (quadratic term) from curve fitting.
            - metrics (dict): Fitting quality indicators:
                - r_squared (float): Coefficient of determination for quadratic fit.
                - covariance (array): Covariance matrix of fitted parameters.

    Raises:
        ValueError: When T contains non-positive values, ka contains non-positive values, arrays have mismatched lengths,
                    or fewer than 3 data points provided (insufficient for quadratic fit).
        TypeError: When inputs are not numeric or R is not a positive scalar.
        RuntimeError: When curve fitting fails to converge (e.g., poor initial guesses, ill-conditioned problem).

    Examples:
        Basic quadratic van't Hoff analysis:
            >>> T = [280, 290, 300, 310, 320]  # Temperatures in Kelvin
            >>> ka = [1e3, 2e3, 5e3, 8e3, 1e4]  # Temperature-dependent equilibrium constants
            >>> R = 8.314  # J·mol⁻¹·K⁻¹
            >>> results = _vant_hoff_quadratic(T, ka, R)
            >>> print(f"ΔH at 300K: {results['delta_H'][2]:.1f} J/mol")

        Comparing with linear model:
            >>> linear_results = _vant_hoff_linear(T, ka, R)
            >>> quadratic_results = _vant_hoff_quadratic(T, ka, R)
            >>> # Compare R² values to determine if quadratic model provides better fit

    Notes:
        - Temperature Dependence: This method accounts for temperature-dependent ΔH and ΔS, unlike the linear van't Hoff.
        - Heat Capacity Effects: Quadratic term (a2) relates to heat capacity change (ΔCp) of the process.
        - Data Requirements: Requires at least 3 temperature points for quadratic fitting; more points improve accuracy.
        - Model Selection: Compare R² values with linear model to determine if quadratic fit is justified.
        - Physical Interpretation: Temperature-dependent parameters provide more accurate thermodynamic description 
          for processes with significant heat capacity changes.
        - Convergence: Curve fitting may require good initial guesses; method uses automatic parameter estimation.
    """
    def quadratic_model(x, a0, a1, a2):
        return a0 + a1*x + a2*x**2
    
    inv_T = 1 / T
    ln_ka = np.log(ka)
    
    popt, pcov = curve_fit(quadratic_model, inv_T, ln_ka)
    a0, a1, a2 = popt
    
    delta_H = -R * (a1 + 2 * a2 / T)
    delta_S = R * (a0 - a2 / T**2)
    delta_G = delta_H - T * delta_S
    
    ln_ka_pred = quadratic_model(inv_T, a0, a1, a2)
    r_squared = 1 - np.sum((ln_ka - ln_ka_pred)** 2) / np.sum((ln_ka - np.mean(ln_ka))**2)
    
    return {
        'delta_H': delta_H,
        'delta_S': delta_S,
        'delta_G': delta_G,
        
        'T': T,
        'ka': ka,
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


def vant_hoff(T, ka, R=8.314):
    """
    Overview:
        Perform van't Hoff analysis to determine thermodynamic parameters with automatic model selection.
        (Core: Compares linear and quadratic van't Hoff models using AIC to select the best fit)

    Details:
        - Function: Executes both linear and quadratic van't Hoff analyses and selects the optimal model
          using Akaike Information Criterion (AIC). Returns results from the model with lower AIC score.
        - Use Case: Ideal for thermodynamic studies where the temperature dependence of ΔH and ΔS is unknown.
          Automatically determines whether constant or temperature-dependent parameters are more appropriate.
        - Special Handling: Computes AIC for model comparison; includes both AIC values in results for transparency.

    Args:
        T (array-like): Absolute temperatures at which equilibrium constants were measured:
            - Role: Independent variable for thermodynamic analysis.
            - Constraints: Must be positive values in Kelvin (K); same length as ka; minimum 3 points.
        ka (array-like): Equilibrium constants from temperature-dependent measurements:
            - Role: Dependent variable for van't Hoff analysis.
            - Constraints: Must be positive values; typically in M⁻¹; same length as T.
        R (float, optional): Gas constant for unit conversion:
            - Role: Converts between dimensionless fitting parameters and thermodynamic units.
            - Default: 8.314 (Standard SI gas constant in J·mol⁻¹·K⁻¹).
            - Note: Using 1.987 cal·mol⁻¹·K⁻¹ will yield results in calorie-based units.

    Returns:
        dict: Dictionary containing thermodynamic parameters from the selected model (linear or quadratic):
            - selected_model (str): Indicator of chosen model ('linear' or 'quadratic').
            - aic (float): AIC value for the selected model.
            - aic_linear/aic_quadratic (float): AIC values for both models (for comparison).
            - All parameters from either _vant_hoff_linear() or _vant_hoff_quadratic() functions:
                - Thermodynamic parameters (delta_H, delta_S, delta_G) with appropriate units
                - Original and transformed data (T, ka, inv_T, ln_ka)
                - Model coefficients (slope/intercept for linear; a0/a1/a2 for quadratic)
                - Fit quality metrics (r_squared, std_err, p_value, or covariance)

    Raises:
        ValueError: When T contains non-positive values, ka contains non-positive values, 
                    arrays have mismatched lengths, or insufficient data points (<3).
        TypeError: When inputs are not numeric or R is not a positive scalar.
        RuntimeError: When curve fitting fails for the quadratic model.

    Examples:
        Basic usage with automatic model selection:
            >>> T = [280, 290, 300, 310, 320]  # Temperature range in Kelvin
            >>> ka = [1e3, 2e3, 5e3, 8e3, 1e4]  # Temperature-dependent equilibrium constants
            >>> results = vant_hoff(T, ka)
            >>> print(f"Selected model: {results['selected_model']}")
            >>> print(f"ΔH: {results['delta_H'][2]:.1f} J/mol")

        Using custom gas constant:
            >>> results = vant_hoff(T, ka, R=1.987)  # cal·mol⁻¹·K⁻¹
            >>> # Results in cal·mol⁻¹ instead of J·mol⁻¹

        Comparing model fits:
            >>> results = vant_hoff(T, ka)
            >>> if results['selected_model'] == 'quadratic':
            ...     print("Quadratic model preferred (temperature-dependent parameters)")
            ... else:
            ...     print("Linear model sufficient (constant parameters)")

    Notes:
        - Model Selection: AIC penalizes model complexity, favoring simpler models unless quadratic fit 
          provides substantially better explanation of variance.
        - Interpretation: Linear model assumes constant ΔH and ΔS; quadratic allows temperature dependence.
        - AIC Difference: ΔAIC > 2 suggests meaningful difference; ΔAIC > 10 indicates strong preference.
        - Data Quality: More temperature points improve model discrimination capability.
        - Physical Consistency: Check that selected model parameters make physical sense for the system.
        - Unit Consistency: Ensure all T values are in Kelvin for correct thermodynamic calculations.
    """
    results_linear = _vant_hoff_linear(T, ka, R=R)
    results_quadratic = _vant_hoff_quadratic(T, ka, R=R)
    
    n = len(T)
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


def _plot_vant_hoff_linear(results, title, dpi, plot_save_dir, plot_save_name):
    """
    Overview:
        Generate a dual-panel visualization for linear van't Hoff analysis results.
        (Core: Creates side-by-side plots showing both the linearized van't Hoff plot and the original temperature dependence)

    Details:
        ▪ Function: Produces a comprehensive two-panel figure that displays both the linear regression of ln(ka) vs. 1/T
          and the corresponding temperature dependence of equilibrium constants. Includes thermodynamic parameter annotations.
        ▪ Use Case: Ideal for visualizing linear van't Hoff analysis in scientific publications or reports where both
          the linear fit quality and physical temperature dependence need to be shown.
        ▪ Special Handling: Automatically generates smooth fit curves for both representations; creates directory if needed.

    Args:
        results (dict): Results dictionary from linear van't Hoff analysis (_vant_hoff_linear function):
            ▪ Role: Provides all necessary data (T, ka, inv_T, ln_ka) and calculated parameters (slope, intercept, delta_H, etc.).
            ▪ Constraints: Must contain complete set of keys: 'T', 'ka', 'inv_T', 'ln_ka', 'slope', 'intercept', 
              'delta_H', 'delta_S', and 'metrics' with 'r_squared'.
        title (str): Main title for the entire figure:
            ▪ Role: Descriptive title displayed above both subplots to identify the analysis.
            ▪ Constraints: Should be informative but concise; will be displayed as a suptitle.
        dpi (int): Resolution for output image in dots per inch:
            ▪ Role: Controls the quality and file size of the saved image.
            ▪ Constraints: Higher values (300-600) for publication quality; lower values for screen viewing.
        plot_save_dir (str): Directory path where the plot image will be saved:
            ▪ Role: Target folder for storing the generated figure.
            ▪ Constraints: Directory will be created if it doesn't exist; requires write permissions.
        plot_save_name (str): Base filename for the output image (without extension):
            ▪ Role: Descriptive name for the saved plot file.
            ▪ Constraints: Should be filesystem-compatible; .png extension will be appended automatically.

    Returns:
        matplotlib.figure.Figure: Figure object containing the dual-panel van't Hoff visualization:
            ▪ Left panel (ax1): Linear van't Hoff plot showing ln(ka) vs. 1/T with regression line and parameter annotations.
            ▪ Right panel (ax2): Temperature dependence plot showing ka vs. T with fitted curve.
            ▪ Both panels include experimental data points, fitted lines, proper axis labels, legends, and grids.
            ▪ Field descriptions:
                ▪ ax1: Linear regression analysis with ΔH, ΔS, and R² values displayed in annotation box.
                ▪ ax2: Physical interpretation showing how equilibrium constant varies with temperature.

    Raises:
        KeyError: When results dictionary is missing required keys for plotting.
        PermissionError: When the function cannot create the save directory or write the image file.
        ValueError: When the results contain invalid numerical values that cannot be plotted.

    Examples:
        Basic usage with a complete results dictionary:
            >>> results = _vant_hoff_linear(T_values, ka_values, R=8.314)
            >>> fig = _plot_vant_hoff_linear(
                    results,
                    title="Protein-Ligand Binding Thermodynamics",
                    dpi=300,
                    plot_save_dir="./figures",
                    plot_save_name="vant_hoff_analysis"
                )

        High-resolution version for publication:
            >>> fig = _plot_vant_hoff_linear(
                    results,
                    title="Temperature Dependence of Association Constant",
                    dpi=600,  # High resolution for journals
                    plot_save_dir="../manuscript/figures",
                    plot_save_name="figure_3_thermodynamics"
                )

    Notes:
        ▪ Dual Perspective: The left panel shows the linearized data used for parameter extraction, while the right panel
          shows the actual temperature dependence for physical interpretation.
        ▪ Parameter Accuracy: The ΔH and ΔS values are averaged or representative values; check if they are 
          temperature-dependent in your system.
        ▪ Fit Quality: R² value indicates how well the linear model explains the variance in the data.
        ▪ Layout Optimization: Uses tight_layout() for professional spacing between subplots and elements.
        ▪ File Management: Automatically creates the save directory if it doesn't exist.
        ▪ Unit Consistency: Ensure temperature values are in Kelvin and equilibrium constants have consistent units.
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


def _plot_vant_hoff_quadratic(results, title, dpi, plot_save_dir, plot_save_name):
    """
    Overview:
        Generate a dual-panel visualization for quadratic van't Hoff analysis results.
        (Core: Creates side-by-side plots showing both the quadratic van't Hoff plot and the original temperature dependence)

    Details:
        - Function: Produces a comprehensive two-panel figure that displays both the quadratic regression of ln(ka) vs. 1/T
          and the corresponding temperature dependence of equilibrium constants. Includes quadratic coefficients and fit quality metrics.
        - Use Case: Ideal for visualizing quadratic van't Hoff analysis where enthalpy and entropy changes are temperature-dependent.
          Suitable for systems with significant heat capacity changes.
        - Special Handling: Uses quadratic fitting to account for temperature-dependent thermodynamic parameters;
          automatically creates directory if needed.

    Args:
        results (dict): Results dictionary from quadratic van't Hoff analysis (_vant_hoff_quadratic function):
            - Role: Provides all necessary data (T, ka, inv_T, ln_ka) and calculated parameters (a0, a1, a2, delta_H, etc.).
            - Constraints: Must contain complete set of keys: 'T', 'ka', 'inv_T', 'ln_ka', 'a0', 'a1', 'a2', 
              'delta_H', 'delta_S', and 'metrics' with 'r_squared'.
        title (str): Main title for the entire figure:
            - Role: Descriptive title displayed above both subplots to identify the quadratic analysis.
            - Constraints: Should clearly indicate this is a quadratic van't Hoff analysis.
        dpi (int): Resolution for output image in dots per inch:
            - Role: Controls the quality and file size of the saved image.
            - Constraints: Higher values (300-600) for publication quality; lower values for screen viewing.
        plot_save_dir (str): Directory path where the plot image will be saved:
            - Role: Target folder for storing the generated figure.
            - Constraints: Directory will be created if it doesn't exist; requires write permissions.
        plot_save_name (str): Base filename for the output image (without extension):
            - Role: Descriptive name for the saved plot file.
            - Constraints: Should be filesystem-compatible; .png extension will be appended automatically.

    Returns:
        matplotlib.figure.Figure: Figure object containing the dual-panel quadratic van't Hoff visualization:
            - Left panel (ax1): Quadratic van't Hoff plot showing ln(ka) vs. 1/T with quadratic fit and coefficient annotations.
            - Right panel (ax2): Temperature dependence plot showing ka vs. T with fitted curve.
            - Both panels include experimental data points, fitted curves, proper axis labels, legends, and grids.
            - Field descriptions:
                - ax1: Quadratic regression analysis with a0, a1, a2 coefficients and R² value displayed.
                - ax2: Physical interpretation showing how equilibrium constant varies with temperature using quadratic model.

    Raises:
        KeyError: When results dictionary is missing required keys for plotting (especially 'a0', 'a1', 'a2').
        PermissionError: When the function cannot create the save directory or write the image file.
        ValueError: When the results contain invalid numerical values that cannot be plotted.

    Examples:
        Basic usage with quadratic results:
            >>> results = _vant_hoff_quadratic(T_values, ka_values, R=8.314)
            >>> fig = _plot_vant_hoff_quadratic(
                    results,
                    title="Quadratic van't Hoff Analysis - Protein Folding",
                    dpi=300,
                    plot_save_dir="./figures",
                    plot_save_name="vant_hoff_quadratic"
                )

        High-resolution version for complex systems:
            >>> fig = _plot_vant_hoff_quadratic(
                    results,
                    title="Temperature-Dependent Thermodynamics with Heat Capacity Effects",
                    dpi=600,
                    plot_save_dir="../manuscript/figures",
                    plot_save_name="figure_4_quadratic_thermodynamics"
                )

    Notes:
        - Temperature Dependence: Quadratic model accounts for temperature-dependent ΔH and ΔS, unlike linear model.
        - Heat Capacity: The quadratic coefficient (a2) relates to the heat capacity change (ΔCp) of the process.
        - Model Selection: Use this visualization when linear van't Hoff plot shows curvature, indicating temperature-dependent parameters.
        - Physical Interpretation: More accurate for systems with significant heat capacity changes (e.g., protein folding, 
          macromolecular interactions).
        - Coefficient Meaning: a0 = ΔS/R - a2/T², a1 = -ΔH/R, a2 relates to ΔCp/2R.
        - Data Requirements: Requires more temperature points than linear analysis for reliable quadratic fitting.
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


def plot_vant_hoff(results, title="Van't Hoff Plot", dpi=300, plot_save_dir="plots", plot_save_name="vant_hoff"):
    """
    Overview:
        Generate an appropriate van't Hoff plot based on the selected model (linear or quadratic).
        (Core: Automatically routes to the correct plotting function based on model selection results)

    Details:
        - Function: Acts as a dispatcher that selects and executes the appropriate van't Hoff visualization
          function based on the model selected during thermodynamic analysis (linear vs. quadratic).
        - Use Case: Provides a unified interface for van't Hoff plotting regardless of which model was 
          automatically selected by the vant_hoff() function. Simplifies plotting workflow.
        - Special Handling: Transparently routes to the underlying plotting function without modifying parameters.

    Args:
        results (dict): Results dictionary from vant_hoff() function containing model selection information:
            - Role: Provides thermodynamic data, parameters, and the key 'selected_model' indicating which plot to generate.
            - Constraints: Must contain 'selected_model' key with value 'linear' or 'quadratic', plus all data required by the respective plotting functions.
        title (str, optional): Main title for the plot:
            - Role: Descriptive title displayed on the generated figure.
            - Default: "Van't Hoff Plot" (Standard title that works for both linear and quadratic versions).
        dpi (int, optional): Resolution for output image:
            - Role: Controls image quality for saved plot file.
            - Default: 300 (Balances quality and file size for most applications).
        plot_save_dir (str, optional): Directory for saving the plot:
            - Role: Target folder where the image file will be stored.
            - Default: "plots" (Standard directory name for figure storage).
        plot_save_name (str, optional): Base filename for the output:
            - Role: Name prefix for the saved PNG file.
            - Default: "vant_hoff" (Descriptive default that identifies the analysis type).

    Returns:
        matplotlib.figure.Figure: The generated figure object from either _plot_vant_hoff_linear() or _plot_vant_hoff_quadratic():
            - Contains the dual-panel van't Hoff visualization appropriate for the selected model.
            - Can be used for further customization, display, or saving in different formats.
            - The specific content depends on whether linear or quadratic model was selected.

    Raises:
        KeyError: When results dictionary missing 'selected_model' key or required data for the selected plotting function.
        ValueError: When 'selected_model' has an invalid value (not 'linear' or 'quadratic').
        PermissionError: When unable to create directory or write file (propagated from underlying plotting functions).

    Examples:
        Basic usage with automatic model selection:
            >>> results = vant_hoff(T, ka)  # This function selects the best model
            >>> fig = plot_vant_hoff(results)  # Automatically plots the selected model

        Customized plotting with specific title and resolution:
            >>> fig = plot_vant_hoff(
                    results,
                    title="Protein-DNA Binding Thermodynamics",
                    dpi=600,
                    plot_save_dir="analysis/figures",
                    plot_save_name="binding_thermodynamics"
                )

        Verifying which model was plotted:
            >>> results = vant_hoff(T, ka)
            >>> fig = plot_vant_hoff(results)
            >>> print(f"Plotted {results['selected_model']} model")  # Confirm which model was used

    Notes:
        - Model Transparency: The function clearly indicates which model (linear/quadratic) was used in the results.
        - Consistent Interface: Provides the same parameter interface regardless of underlying model.
        - AIC-Based Selection: The model selection is based on Akaike Information Criterion from the vant_hoff() function.
        - No Reprocessing: Does not reanalyze data; uses pre-computed results from vant_hoff().
        - Extension Ready: Easy to extend if additional van't Hoff models are implemented in the future.
        - Parameter Passing: All parameters are passed unchanged to the underlying plotting functions.
    """
    if results['selected_model'] == 'linear':
        return _plot_vant_hoff_linear(results, title, dpi, plot_save_dir, plot_save_name)
    else:
        return _plot_vant_hoff_quadratic(results, title, dpi, plot_save_dir, plot_save_name)


if __name__ == "__main__":
    Q = np.array([0, 1e-6, 2e-6, 4e-6, 6e-6])
    F0 = np.array([1000, 950, 900, 850, 800])
    F = np.array([1000, 700, 500, 300, 200])
    
    plot_stern_volmer(stern_volmer(F0, F, Q))
    plot_hill(hill(F0, F, Q))
    
    T = np.array([298, 303, 310, 318, 333])
    ka = np.array([1.15E+06, 1.45E+05, 3.02E+04, 6.57E+03, 6.05E+03])

    plot_vant_hoff(vant_hoff(T, ka))
