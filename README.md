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
from NPI_analyzer import stern_volmer, plot_stern_volmer, hill, plot_hill, vant_hoff, plot_vant_hoff
    
plot_stern_volmer(
    stern_volmer(
        [1000, 950, 900, 850, 800],
        [1000, 700, 500, 300, 200],
        [0, 1e-6, 2e-6, 4e-6, 6e-6],
        2.6e-9
    )
)
plot_hill(
    hill(
        [1000, 950, 900, 850, 800],
        [1000, 700, 500, 300, 200],
        [0, 1e-6, 2e-6, 4e-6, 6e-6]
    )
)
plot_vant_hoff(
    vant_hoff(
        [298, 303, 310, 318, 333],
        [1.15e+6, 1.45e+5, 3.02e+4, 6.57e+3, 6.05e+3]
    )
)
```

## Requirements

`numpy`, `scipy`, `matplotlib`

Just download `NPI_analyzer.py` and import into your analysis scripts.


## For bionano researchers

Useful for characterizing NP-protein binding affinity, mechanism, and thermodynamics from fluorescence quenching experiments.
