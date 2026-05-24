# Qlib: AI-Oriented Quantitative Investment Platform

This file provides a comprehensive overview of the Microsoft `Qlib` repository, its architecture, guidelines for data preparation, and execution procedures for quantitative trading workflows. Use this as a reference guide for all development, execution, and troubleshooting tasks in this workspace.

---

## 1. Project Overview

**Qlib** is an open-source, AI-oriented quantitative investment platform developed by Microsoft. It covers the full machine learning pipeline of quantitative investment, including:
*   **Data Server**: High-performance storage and retrieval of financial data (supports daily/intraday frequencies, point-in-time database, and caching).
*   **Data Handlers & Processors**: Feature engineering modules (e.g., `Alpha158`, `Alpha360`) that compile mathematical formulas into dense predictive feature matrices.
*   **Models (Paper Zoo)**: Built-in state-of-the-art machine learning models including `LGBModel` (LightGBM), `CatBoostModel`, `XGBModel`, and deep neural networks (`LSTM`, `GRU`, `Transformer`, `TabNet`, etc.).
*   **Strategy**: Decisions generators (e.g., `TopkDropoutStrategy`) mapping raw model predictions into optimal portfolio weights.
*   **Backtest & Evaluation**: Highly realistic simulators checking portfolio returns, standard deviations, maximum drawdowns, information ratios, and transaction costs.
*   **Workflow / Recorder**: Integrates the entire loop, tracking metrics and archiving files automatically via **MLflow**.

---

## 2. Environment and Installation

### Virtual Environment setup
Ensure you are using the dedicated Python virtual environment located at `.venv`. To run scripts or initialize the Qlib CLI, always prefix commands with the active python interpreter or activate it:

```bash
# Activation
source .venv/bin/activate

# Verification of python version (runs on Python 3.14 on macOS)
.venv/bin/python --version
```

### Dependencies & System Requirements
On macOS, models like `LightGBM` require OpenMP to run. If you hit `dlopen() Library not loaded: @rpath/libomp.dylib` errors, reinstall/link `libomp` via Homebrew:
```bash
brew reinstall libomp
```

---

## 3. Data Preparation

Qlib requires preprocessed stock dataset archives (e.g. daily trading calendars, instruments, and features) to run backtests. 

### Automated Downloader
We have provided an optimized, robust downloading script: `download_qlib_data.sh`. It downloads the latest Yahoo-finance preprocessed China market data (`qlib_bin.tar.gz`) from GitHub Releases, performs checksum/gzip-compression validation, and handles automatic cleanup on failure.

Run the script to download data directly to your local workspace or system-wide directory:
```bash
chmod +x download_qlib_data.sh
./download_qlib_data.sh
```

### Data Storage Conventions
By default, Qlib and the benchmark YAML configurations look for data in:
1.  **System Home Directory (Default Qlib location)**: `~/.qlib/qlib_data/cn_data`
2.  **Local Workspace Directory**: `./qlib_data/cn_data` or `./qlib/qlib_data/cn_data` (a symbolic link can be created to keep them synced: `ln -sfn ../../qlib_data/cn_data ./qlib/qlib_data/cn_data`).

---

## 4. Running Workflows (`qrun`)

`qrun` is the entry point for automated quant workflows. It parses a YAML configuration file to load the data, fit/train the model, output predictions, run backtests, and log metrics.

### Execution Commands

You can run workflows in the active virtual environment using either of the following commands:

```bash
# Using Python module execution (Recommended for explicit debugging)
.venv/bin/python -m qlib.cli.run <path_to_config.yaml>

# Using the registered CLI entry point
qrun <path_to_config.yaml>
```

### Custom Benchmark Configurations (2016-2026)
We have added specific, optimized configs for testing the LightGBM model on the CSI300 and CSI500 indices over a 10-year historical span (Train: 2016-2021, Valid: 2022-2023, Test/Backtest: 2024-2026).

```bash
# Run CSI300 LightGBM Workfow (2016-2026)
.venv/bin/python -m qlib.cli.run examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_2016_2026.yaml

# Run CSI500 LightGBM Workflow (2016-2026)
.venv/bin/python -m qlib.cli.run examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500_2016_2026.yaml
```

---

## 5. Quantitative Indicator Analysis

When a backtest finishes, Qlib outputs tabular risk reports.

### Key Outputs and Mathematical Meaning:
*   **`annualized_return` (Annualized Excess Return with Cost)**:
    *   **Meaning**: Represents the **annualized excess return** over the chosen index benchmark (e.g., CSI300 or CSI500) **after deducting all transaction fees** (such as bid-ask spreads, slippage, and commission fees defined in the YAML configuration under `exchange_kwargs`).
    *   **Computation**: Qlib uses an arithmetic summation (`mode="sum"`) by default to aggregate daily excess returns rather than compounded product multiplication, preventing exponential distortions:
        $$\text{Annualized Excess Return with Cost} = \text{Mean}(\text{Daily Portfolio Return} - \text{Daily Benchmark Return} - \text{Daily Transaction Cost}) \times N$$
        *(where $N$ represents the annualized trading days scaler, typically 238 or 252)*.

---

## 6. Data Processors Conventions (`infer_processors` vs `learn_processors`)

Qlib's datasets use data processors to transform raw tabular metrics into model inputs and training targets.

### `infer_processors` (Feature Preprocessing)
*   **Purpose**: Applied to features to ensure model compatibility (Z-score normalization, imputations, clipping, etc.).
*   **Tree Models (LightGBM, XGBoost, CatBoost)**: Keep `infer_processors: []` (empty list). Decision trees are monotonically invariant and do not require scaled input features. Alpha158's intricate technical indicators do not benefit from standardization.
*   **Neural Networks (MLP, GRU, LSTM, Transformers)**: Must explicitly supply an `infer_processors` block to normalize features, otherwise models will fail to converge:
    ```yaml
    infer_processors: [
        {"class": "ProcessInf", "kwargs": {}},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
        {"class": "Fillna", "kwargs": {}}
    ]
    ```

### `learn_processors` (Label Preprocessing)
*   **Purpose**: Applied to training targets (Labels).
*   **Default Behavior (`_DEFAULT_LEARN_PROCESSORS`)**: 
    If not specified in YAML, Qlib automatically applies cross-sectional Z-score normalization (`CSZScoreNorm`) and null removal (`DropnaLabel`):
    ```python
    _DEFAULT_LEARN_PROCESSORS = [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]
    ```
*   **Rationale**: For stock selection models, we want to predict a stock's **relative strength** within the same day's pool (cross-sectional ranking) rather than its absolute future nominal yield. Cross-sectional scaling strips away general market beta, highlighting alpha.

---

## 7. Experiment Management & Visualization

Qlib leverages **MLflow** for tracking parameter configurations, models, metric indicators (IC, ICIR, returns, drawdowns), and generated plot artifacts.

### Visualizing Results
To view detailed logs, comparison charts, and artifacts from your backtests, start the local MLflow dashboard:

```bash
# Run in project root
.venv/bin/mlflow ui
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser to access the graphical panel.

---

## 8. Testing

Unit and integration tests are powered by `pytest`. Configurations are set in `tests/pytest.ini`.

*   To run the complete test suite:
    ```bash
    pytest tests/
    ```
*   To skip slow-running integration tests:
    ```bash
    pytest tests/ -m "not slow"
    ```
