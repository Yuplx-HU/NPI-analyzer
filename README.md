# NPI-analyzer
NPI-analyzer: Python toolkit for Nanoparticle-Protein Interaction analysis. Performs fluorescence quenching, binding affinity, and thermodynamic analyses with automated model selection and publication-ready visualization.


## What it does
- **Stern-Volmer analysis**: Calculate quenching constants and identify quenching mechanism
- **Hill analysis**: Determine binding affinity and cooperativity  
- **Van't Hoff analysis**: Extract thermodynamic parameters (ΔH, ΔS, ΔG)
- **Auto model selection**: Chooses best fit (linear/quadratic) using AIC
- **Auto-plotting**: Generates publication-ready figures

## Quick use

```python
from NPI_analyzer import stern_volmer, plot_stern_volmer
from NPI_analyzer import hill, plot_hill
from NPI_analyzer import vant_hoff, plot_vant_hoff

Q = [0, 1e-6, 2e-6, 4e-6, 6e-6]
F0 = [1000, 950, 900, 850, 800]
F = [1000, 700, 500, 300, 200]

plot_stern_volmer(stern_volmer(F0, F, Q))
plot_hill(hill(F0, F, Q))

T = [298, 303, 310, 318, 333]
ka = [1.15E+06, 1.45E+05, 3.02E+04, 6.57E+03, 6.05E+03]

plot_vant_hoff(vant_hoff(T, ka))
```

## Requirements

`numpy`, `scipy`, `matplotlib`

Just download `npi_analyzer.py` and import into your analysis scripts.


## For bionano researchers

Useful for characterizing NP-protein binding affinity, mechanism, and thermodynamics from fluorescence quenching experiments.
