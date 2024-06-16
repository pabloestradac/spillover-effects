# Spillover Effects in Randomized Experiments

This repository implements weighted least squares (WLS) estimator for spillover effects in randomized experiments. The WLS estimator is based on the work of [Gao and Ding (2023)](https://arxiv.org/abs/2309.07476).

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
wls_results = spef.WLS(name_y='Y', 
                       name_z=['exposure0', 'exposure1'], 
                       name_pscore=['pscore0', 'pscore1'], 
                       data=data, 
                       kernel_weights=kernel_mat, 
                       name_x='X')
print(wls_results.summary)
```

The output of the previous code is:

|            | coef |  se  | t-val | p-val | ci-low | ci-up |
|------------|------|------|-------|-------|--------|-------|
| spillover  | 0.71 | 0.30 |  2.36 |  0.02 |   0.12 |  1.30 |
| exposure0  | -4.01| 0.31 |-12.95 |  0.00 |  -4.62 | -3.40 |
| exposure1  | -3.30| 0.23 |-14.42 |  0.00 |  -3.75 | -2.85 |
| exposure0*X| -2.08| 0.14 |-14.49 |  0.00 |  -2.37 | -1.80 |
| exposure1*X| -2.21| 0.11 |-19.57 |  0.00 |  -2.43 | -1.99 |

The two inputs that the WLS class requires are a pandas DataFrame with the data and a sparse matrix for the kernel weights. The package provides helper functions to calculate the propensity score (pscore column), spillover exposure (exposure column), and kernel weights (sparse matrix) for the WLS estimator. Detailed examples can be found in the [examples](https://github.com/pabloestradac/spillover-effects/blob/main/example.ipynb) notebook. 

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

An important note is to avoid selecting subsets of the data and distance matrix before running the WLS estimator. Instead use option `subsample` in the `WLS` class to select a subset of the data. This will ensure that the distances between $i$ and $j$ are calculated correctly.
 
<!-- https://github.com/MichaelKim0407/tutorial-pip-package?tab=readme-ov-file -->