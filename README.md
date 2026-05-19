# AdaKoop: Efficient Modeling of Nonlinear Dynamics from Nonstationary Data Streams with Koopman Operator Regression

<div align="left">

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31211/)
[![Pyenv](https://img.shields.io/badge/Pyenv-2.6.7-yellow.svg)](https://github.com/pyenv/pyenv#installation)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
</div>

This repository contains the implementation of the [KDD 2026](https://kdd2026.kdd.org/) paper, "AdaKoop: Efficient Modeling of Nonlinear Dynamics from Nonstationary Data Streams with Koopman Operator Regression," by Naoki Chihara, Ren Fujiwara, Yasuko Matsubara, Yasushi Sakurai.

<p align="center">
<img src=".\docs\assets\overview.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of AdaKoop.
</p>

## Usage
1. Clone this repository.
    ```bash
    git clone https://github.com/C-Naoki/AdaKoop.git
    ```
2. Construct a virtual environment and install the required packages.
    ```bash
    make install
    ```
    - Note that it requires to [pyenv](https://github.com/pyenv/pyenv#installation) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
    - If you prefer not to use them, you can also use [`requirements.txt`](https://github.com/C-Naoki/AdaKoop/blob/main/requirements.txt) created based on pyproject.toml.

    Specifically, the above command performs the following steps:
    1. if necessary, install Python 3.12.11 using pyenv, and then switch to this version.
    2. create a virtual environment based on Python 3.12.11
    3. install packages in `pyproject.toml`.
    4. attach the path file (i.e., `*.pth`) in the `site-packages/` for extending module search path.

    Please check the [`Makefile`](https://github.com/C-Naoki/AdaKoop/blob/main/Makefile) for more details.

3. Prepare the datasets.
    ```bash
    make prepare
    ```
    - This command generates the datasets and saves them in the [`data/dysts/`](./data/dysts) directory.

4. Run quick demos of AdaKoop
    ```bash
    bash bin/demo.sh
    ```
    If you want the command to continue running after logging out, you prepare `nohup/` directory and use `-n` option as shown below (using nohup).
    ```bash
    bash bin/demo.sh -n
    ```
    - The execution log is saved in `nohup/` directory.

## Minimal Example
```python
import numpy as np

from src.models.adakoop import AdaKoop


# Build the model and set parameters.
model = AdaKoop(verbose=False)
model.init_params(d=d, lcurr=lcurr)

# Initialization.
# - X_train, X_test : np.ndarray, shape (n, d)
# - X_test is time series immediately after X_train.
model.initialize(X_train)
model.set_initial_model(X_train[-lcurr:])

# Online forecasting.
pred = np.full_like(X_test, np.nan)
for tc in range(len(X_test) - lstep):
    x_new = X_test[tc]
    model.model_selection(x_new)
    model.update(x_new)
    forecast, _, _ = model.forecast(lstep=lstep)
    pred[tc + lstep] = forecast[-1]
```

## Parameters
The main parameters of AdaKoop are summarized in the table below.

| Argument | Default | Description |
| --- | --- | --- |
| `lcurr` | Required | Current window length. |
| `lstep` | Required | Forecasting horizon. |
| `nu` | `1e-3` | Threshold for basis orthogonalization. |
| `gamma` | `3e-3` | Online forgetting factor. |
| `kernel_type` | `rbf` | Kernel used for window comparison and dictionary features. |
| `lambda_A` | `1e-6` | Ridge regularization strength for the transition matrix `A`. |
| `em_iters` | `3` | Maximum number of EM refinement iterations. |
| `em_tol` | `1e-6` | Convergence tolerance for EM refinement. |
| `m_max` | `100` | Maximum dictionary size. |
| `chi2_p` | `0.99` | Chi-square quantile used to compute the innovation-distance threshold. |

## Datasets
We used the [`dysts`](https://github.com/GilpinLab/dysts) benchmark datasets for evaluating of our proposed method. This benchmark offers various types of chaotic dynamical systems spanning diverse fields, including astrophysics, climatology, and biochemistry.

## Citation
If you use this code for your research, please consider citing our paper.

```bibtex
```
