# Fluorescence Quenching Analysis Toolkit

A Python package for analyzing fluorescence quenching data, featuring Stern-Volmer, Hill, and Van't Hoff analyses with automatic plotting capabilities.

## Installation

### Requirements
- Python 3.8+
- Required packages:
  - numpy
  - scipy
  - matplotlib

### Installation
```bash
pip install numpy scipy matplotlib
```

## Features

- **Stern-Volmer Analysis**: Calculate quenching constants (k_SV, k_q) from fluorescence data
- **Hill Plot Analysis**: Determine binding affinity (k_a) and cooperativity (n)
- **Van't Hoff Analysis**: Extract thermodynamic parameters (ΔH, ΔS, ΔG) from temperature-dependent data
- **Automatic Plotting**: Generate publication-quality plots with customizable saving options
- **Model Selection**: Automatically selects best-fit model for Van't Hoff analysis

## Quick Start

### Basic Usage

```python
import numpy as np
from NPI_analyzer import stern_volmer, hill, vant_hoff

# Sample data
F0 = [100, 98, 95, 90, 85, 80]  # Fluorescence without quencher
F = [100, 85, 72, 60, 50, 42]   # Fluorescence with quencher
Q = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Quencher concentration (M)
tau0 = 1e-9  # Fluorescence lifetime (s)

# Perform Stern-Volmer analysis
results_sv = stern_volmer(F0, F, Q, tau0, save_plot=True)
print(f"Stern-Volmer constant: {results_sv['k_SV']:.2e} M⁻¹")
print(f"Quenching rate constant: {results_sv['k_q']:.2e} M⁻¹s⁻¹")

# Perform Hill analysis
results_hill = hill(F0, F, Q, save_plot=True)
print(f"Association constant: {results_hill['k_a']:.2e} M⁻¹")
print(f"Hill coefficient: {results_hill['n']:.3f}")
```

## Function Reference

### `stern_volmer()`
Performs Stern-Volmer analysis to determine quenching constants.

**Parameters:**
- `F0`: Baseline fluorescence intensities (list[float])
- `F`: Quenched fluorescence intensities (list[float])
- `Q`: Quencher concentrations in M (list[float])
- `tau0`: Fluorescence lifetime in s (float)
- `return_plot`: Return matplotlib figure (bool, default: False)
- `save_plot`: Save plot to file (bool, default: False)
- `save_plot_dir`: Directory for saving plots (str, default: "output/plots/")

**Returns:**
Dictionary with keys: `k_SV`, `k_q`, `r_squared`, `std_error`, `plot` (optional), `save_plot_path` (optional)

### `hill()`
Performs Hill plot analysis to determine binding parameters.

**Parameters:**
- `F0`: Baseline fluorescence intensities (list[float])
- `F`: Quenched fluorescence intensities (list[float])
- `Q`: Quencher concentrations in M (list[float])
- `return_plot`: Return matplotlib figure (bool, default: False)
- `save_plot`: Save plot to file (bool, default: False)
- `save_plot_dir`: Directory for saving plots (str, default: "output/plots/")

**Returns:**
Dictionary with keys: `k_a`, `n`, `r_squared`, `std_error`, `plot` (optional), `save_plot_path` (optional)

### `vant_hoff()`
Performs Van't Hoff analysis for thermodynamic parameter determination.

**Parameters:**
- `T`: Temperatures in K (list[float])
- `ka`: Association constants (list[float])
- `R`: Gas constant (float, default: 8.314 J/(mol·K))
- `return_plot`: Return matplotlib figure (bool, default: False)
- `save_plot`: Save plot to file (bool, default: False)
- `save_plot_dir`: Directory for saving plots (str, default: "output/plots/")

**Returns:**
Dictionary with keys: `deltaH`, `deltaS`, `deltaG`, `selected_model`, `aic`, `r_squared`, `plot` (optional), `save_plot_path` (optional)

## Advanced Examples

### Van't Hoff Analysis with Temperature Data
```python
# Temperature-dependent association constants
T = [280, 285, 290, 295, 300]  # Temperatures (K)
ka = [1.2e4, 1.5e4, 1.8e4, 2.2e4, 2.7e4]  # Association constants

# Perform Van't Hoff analysis
results_vh = vant_hoff(T, ka, save_plot=True)

print(f"Enthalpy change: {results_vh['deltaH'][0]:.2f} kJ/mol")
print(f"Entropy change: {results_vh['deltaS'][0]:.2f} J/(mol·K)")
print(f"Selected model: {results_vh.get('selected_model', 'linear')}")
```

### Custom Plot Saving
```python
# Customize plot output
results = stern_volmer(
    F0, F, Q, tau0,
    return_plot=True,
    save_plot=True,
    save_plot_dir="my_analysis/plots/"
)

# Access the saved plot path
print(f"Plot saved to: {results['save_plot_path']}")
```

## Output Examples

### Stern-Volmer Analysis Output
```python
{
    'k_SV': 12.5,           # Stern-Volmer constant (M⁻¹)
    'k_q': 2.5e9,          # Bimolecular quenching rate constant (M⁻¹s⁻¹)
    'r_squared': 0.998,    # Regression coefficient
    'std_error': 0.05,     # Standard error of fit
    'plot': Figure,        # Matplotlib figure (if return_plot=True)
    'save_plot_path': 'output/plots/stern_volmer-2024-01-15-143022.png'
}
```

### Van't Hoff Analysis Output
```python
{
    'deltaH': -25.3,       # Enthalpy change (kJ/mol)
    'deltaS': -45.2,       # Entropy change (J/(mol·K))
    'deltaG': [-12.1, -11.8, -11.5, -11.2, -10.9],  # Gibbs free energy (kJ/mol)
    'T': array([280, 285, 290, 295, 300]),
    'r_squared': 0.995
}
```

## Plot Customization

The generated plots can be customized by modifying the returned matplotlib figure:

```python
results = stern_volmer(F0, F, Q, tau0, return_plot=True)
fig = results['plot']
ax = fig.axes[0]

# Customize the plot
ax.set_title("Custom Stern-Volmer Plot", fontsize=14)
ax.set_xlabel("[Quencher] (M)", fontsize=12)
ax.set_ylabel("F₀/F", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# Save customized plot
fig.savefig("custom_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

## Units and Conventions

- Concentrations: Molar (mol/L)
- Temperature: Kelvin (K)
- Time: seconds (s) for fluorescence lifetime
- Thermodynamic parameters:
  - ΔH: kJ/mol
  - ΔS: J/(mol·K)
  - ΔG: kJ/mol
- Rate constants:
  - k_SV: M⁻¹
  - k_q: M⁻¹s⁻¹
  - k_a: M⁻¹
