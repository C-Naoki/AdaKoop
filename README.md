# AdaKoop: Efficient Modeling of Nonlinear Dynamics from Nonstationary Data Streams with Koopman Operator Regression

<div align="left">

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3915/)
[![Pyenv](https://img.shields.io/badge/Pyenv-2.6.7-yellow.svg)](https://github.com/pyenv/pyenv#installation)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
</div>

<p align="center">
<img src=".\docs\assets\overview.png" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of AdaKoop.
</p>

## Requirements
This source code is tested with the following dependencies:

- Python == 3.12.11
- hydra-core == 1.3.2
- scikit-learn == 1.8.0
- scipy == 1.17.0

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

## Datasets
We used the `dysts` benchmark datasets for evaluating of our proposed method. This benchmark offers various types of chaotic dynamical systems spanning diverse fields, including astrophysics, climatology, and biochemistry.
