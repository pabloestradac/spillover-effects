# Spillover Effects in Randomized Experiments

This repository implements weighted least squares (WLS) estimator for spillover effects in randomized experiments. The WLS estimator is based on the work of [Gao and Ding (2023)](https://arxiv.org/abs/2309.07476).

## Installation

You can install the package via pip:
    
```bash
pip install spillover-effects
```

## Usage

The package provides a class `WLS` that can be used to estimate spillover effects when the propensity score is known. The following example demonstrates how to use the package:

| Attempt | #1    | #2    |
| :---:   | :---: | :---: |
| Seconds | 301   | 283   |

```python
import spillover_effects as spef

# Load data
data, distance_matrix = spef.utils.load_data()

# WLS estimator
wls_results = spef.WLS(name_y='Y', name_z=['exposure0', 'exposure1'], name_pscore=['pscore0', 'pscore1'], data=data, kernel_weights=distance_matrix, name_x='X')
print(wls_results.summary)
```
The package also provides functions to calculate the propensity score, spillover exposure, and kernel weights matrix for the WLS estimator. Detailed examples can be found in the [examples](https://github.com/pabloestradac/spillover-effects/blob/main/example.ipynb) notebook.
 
<!-- https://github.com/MichaelKim0407/tutorial-pip-package?tab=readme-ov-file -->