# 003_Advanced_Trading_Strategies
Project 003 for **Microstructure and Trading systems** class. This repo implements a full pipeline to train ML models for trading signals, backtest strategies, and visualize results via a dashboard.

> Codebase includes modules like `main.py` (training), `backtest.py` (simulation), and `dashboard.py` (visualization), plus utilities under `utils.py`, `signals.py`, `metrics.py`, etc. The repo is licensed under MIT. ([GitHub][1])

---

## Table of Contents

* [Features](#features)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Quickstart (macOS / Apple Silicon friendly)](#quickstart-macos--apple-silicon-friendly)
* [How to Run](#how-to-run)

  * [1) Train / Generate Signals](#1-train--generate-signals)
  * [2) Backtest](#2-backtest)
  * [3) Dashboard](#3-dashboard)
* [Configuration](#configuration)
* [Data](#data)
* [Experiment Tracking (optional)](#experiment-tracking-optional)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Features

* **ML models** for trading signals (`mlp.py`, `cnn.py`, `models.py`, `classbalance.py`).
* **Signal processing & normalization** (`signals.py`, `normalization.py`).
* **Backtesting** with performance metrics (`backtest.py`, `metrics.py`, `plots.py`).
* **Dashboard/visualization** for results (`dashboard.py`).
* **Utilities & parameters** (`utils.py`, `params.py`).
* **Reproducible environments** via `requirements.txt`.
* **MIT License** for academic/educational use. ([GitHub][1])

---

## Repository Structure

```
.
├─ backtest.py         # Run historical simulations / performance calc
├─ dashboard.py        # Visualization / dashboard app
├─ datadrift.py        # Drift checks (e.g., KS tests, rolling windows)
├─ main.py             # Training / pipeline entry-point
├─ signals.py          # Signal engineering helpers
├─ normalization.py    # Feature scaling / normalization
├─ metrics.py          # Metrics and evaluation utilities
├─ plots.py            # Plotting functions
├─ models.py           # Model wrappers / helpers
├─ mlp.py, cnn.py      # Example model architectures
├─ classbalance.py     # Class balancing utilities
├─ utils.py            # IO, misc helpers
├─ params.py           # Central configuration (paths, hyperparams)
├─ requirements.txt    # Python dependencies
├─ trainlog.py         # (Optional) training logs helpers
├─ mlruns/             # (Optional) MLflow local runs folder
└─ README.md
```

> File names based on the public listing of the repository. ([GitHub][1])

---

## Requirements

* **Python**: 3.10+ recommended
* **OS**: macOS (Apple Silicon M2 compatible), Linux, or Windows
* **Packages**: Install from `requirements.txt`

> A `requirements.txt` is present in the repo root. ([GitHub][1])

---

## Quickstart (macOS / Apple Silicon friendly)

```bash
# 0) Clone
git clone https://github.com/AnaSofiaHinojosa/003_Advanced_Trading_Strategies.git
cd 003_Advanced_Trading_Strategies

# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .\.venv\Scripts\activate           # Windows PowerShell

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt
```

> If you hit Apple Silicon wheels issues (e.g., with `scipy`/`numpy`), try:
> `pip install --no-cache-dir --only-binary=:all: numpy` then re-run `pip install -r requirements.txt`.

---

## How to Run

### 1) Train / Generate Signals

Most projects wire training through `main.py`. Run:

```bash
python main.py
```

This typically:

* Loads/creates datasets & features
* Trains the selected model(s)
* Saves artifacts, logs, and optionally results used by backtesting

> If there’s a `params.py`, you can adjust hyperparameters, tickers/universe, paths, or flags there before running.

### 2) Backtest

Run the backtester on available signals/results:

```bash
python backtest.py
```

This will compute performance metrics (e.g., returns, drawdowns) and—depending on the code—save figures/tables via `plots.py` & `metrics.py`.

### 3) Dashboard

There are two common patterns for `dashboard.py`:

* **Dash/Plotly/Bokeh/Plain script**:

  ```bash
  python dashboard.py
  ```

  Then open the printed local URL (often `http://127.0.0.1:8050/`).

* **Streamlit app** (if the file imports `streamlit`):

  ```bash
  streamlit run dashboard.py
  ```

  It will auto-open your browser.

> If you’re unsure which applies, open `dashboard.py` and look for imports:
>
> * If you see `import streamlit as st` → use **Streamlit** command.
> * If you see `from dash import Dash` or `import dash` → use **python dashboard.py**.

---

## Configuration

* **`params.py`**: Set default tickers, date ranges, feature flags, train/test splits, model configs, and output directories.
* **Environment variables** (if any): If your data loader needs credentials (e.g., an API key), export them first:

  ```bash
  export DATA_API_KEY="your_token_here"       # macOS/Linux
  # setx DATA_API_KEY "your_token_here"       # Windows (new shell)
  ```

> Adjust the values to match your environment, especially file paths and data locations.

---

## Data

* If the project expects local CSV/Parquet files, place them in a `data/` folder (or as configured in `params.py`).
* If it fetches data online (e.g., yfinance, APIs), ensure you have internet access and any needed keys.
* Typical columns (for OHLCV): `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, plus any engineered features.

---

## Experiment Tracking (optional)

The repo contains an `mlruns/` directory, which suggests **MLflow** local tracking is/was used. If MLflow is integrated:

```bash
# Start MLflow UI (optional)
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open the UI at `http://127.0.0.1:5000` to browse experiments.
(If not using MLflow, you can ignore this section.) ([GitHub][1])

---

## Troubleshooting

* **Package install errors (Apple Silicon)**:
  Try installing `numpy`/`scipy` wheels first, or use Conda (`conda create -n ats python=3.10`).

* **Streamlit not found**:
  `pip install streamlit` or ensure it’s in `requirements.txt`.

* **Dash not found**:
  `pip install dash` or ensure it’s in `requirements.txt`.

* **Matplotlib/Plotly backend issues**:
  Use a standard Python terminal (not VS Code’s restricted mode) and ensure GUI backends are available, or prefer browser-based dashboards (Streamlit/Dash).

* **Paths**:
  If scripts can’t find `data/` or output folders, adjust paths in `params.py`.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](./LICENSE) for details. ([GitHub][1])

---

### Notes for reviewers/instructors

* The **primary entry points** are `main.py` (training), `backtest.py` (evaluation), and `dashboard.py` (visualization).
* Core utilities include `signals.py`, `normalization.py`, `metrics.py`, and `plots.py` for a reproducible ML + backtesting workflow. ([GitHub][1])

---
