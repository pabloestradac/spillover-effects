# Spillover Effects in Randomized Experiments

This repository implements weighted least squares (WLS) estimators for spillover effects in randomized experiments. The WLS estimator is based on the work of [Gao and Ding (2023)](https://arxiv.org/abs/2309.07476) in the standard case with complete data. The package also includes bounds for the spillover effects when there are missing outcomes. This is based on the work of [Estrada (2024)](https://github.com/pabloestradac/Spillover_Bounds).


## Installation

You can install the package via pip:
    
```bash
pip install spillover-effects
```

## Usage

The package provides a class `WLS` that can be used to estimate spillover effects when the propensity score is known. The following example demonstrates how to use the package:

```python
import spillover_effects as spef

# Load data and kernel matrix
data, kernel_mat = spef.utils.load_data()

# Estimate spillover effects
wls_results = spef.WLS(name_y='Y', name_z='exposure', name_pscore='pscore',
                       data=data, kernel_weights=kernel_mat, name_x='X')
print(wls_results.summary)
```

The output of the previous code is:

|            | coef |  se  | t-val | p-val | ci-low | ci-up |
|------------|-----:|-----:|------:|------:|-------:|------:|
| spillover  | 0.71 | 0.30 |  2.36 |  0.02 |   0.12 |  1.30 |
| exposure0  | -4.01| 0.31 |-12.95 |  0.00 |  -4.62 | -3.40 |
| exposure1  | -3.30| 0.23 |-14.42 |  0.00 |  -3.75 | -2.85 |
| exposure0*X| -2.08| 0.14 |-14.49 |  0.00 |  -2.37 | -1.80 |
| exposure1*X| -2.21| 0.11 |-19.57 |  0.00 |  -2.43 | -1.99 |

When the outcome has missing values, the package allows to calculate bounds for the spillover effects. The following example demonstrates how to use the package:

```python
# Estimate spillover bounds
wls_bounds = spef.BoundsML((name_y='Y', name_z='exposure', name_pscore='pscore',
                            name_x=name_covariates, data=data, kernel_weights=distances,
                            n_splits=10, n_cvs=10, method='automl')
print(wls_bounds.summary)
```

The output of the previous code is:

Warning: 170 observations have missing values (127 missing outcomes)\
Warning: 34 observations have propensity scores outside (0.01, 0.99)
|           | lower-bound | upper-bound | ci-low | ci-up |
|-----------|------------:|------------:|-------:|------:|
| spillover |        0.21 |        0.28 |   0.12 |  0.32 |

The two inputs that the WLS class requires are a pandas DataFrame with the data and a sparse matrix for the kernel weights. The package provides helper functions to calculate the propensity score (pscore column), spillover exposure (exposure column), and kernel weights (sparse matrix) for the WLS estimator. Detailed examples can be found in the [WLS Examples](notebooks/example_wls.ipynb) and [Bounds Examples](notebooks/example_bounds.ipynb) notebooks.

The two data structures the user needs to use this package are 1) the data and 2) the edge list. The data should be a pandas DataFrame with columns such as:

| ID | Y | D | X |
|----|---|---|---|
| 1  | 5 | 1 | 1 |
| 2  | 8 | 0 | 0 |
| 3  | 2 | 1 | 1 |

The edge list should be a pandas DataFrame with up to $K+1$ columns where $K$ is the number of targets. The first column should be the source ID and the rest of the columns should be the target IDs. The edge list should have the following format:

| Source_ID | Target1_ID | Target2_ID | ... | TargetK_ID |
|-----------|------------|------------|-----|------------|
| 1         | 2          | 3          | ... | 4          |
| 2         | 1          | 3          | ... | 4          |
| 3         | 1          | 2          | ... | 4          |

An important note is to avoid selecting subsets of the data and distance matrix before running the WLS estimator. Instead use option `subsample` to select a subset of the data. This will ensure that the distances between $i$ and $j$ are calculated correctly.
 
<!-- https://github.com/MichaelKim0407/tutorial-pip-package?tab=readme-ov-file -->