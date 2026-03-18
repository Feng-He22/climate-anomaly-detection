# Climate Anomaly Detection

This repository restructures the original notebook prototype into a standard Python project for climate anomaly detection on HadUK-Grid daily data. It supports two complementary unsupervised models:

- `LSTM Autoencoder` for sequence reconstruction and anomaly scoring
- `Isolation Forest` for tree-based anomaly detection on flattened temporal windows

The project expects raw data under `data/raw/hadukgrid_60km_last10y/` and currently matches your existing `rainfall`, `tasmax`, and `tasmin` layout.

## Project Structure

```text
climate_anomaly_detection/
├── config/
│   └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
│   ├── lstm_autoencoder.py
│   └── isolation_forest.py
├── src/
│   ├── data_loader.py
│   ├── anomaly_detector.py
│   └── visualization.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── tests/
│   └── test_*.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── README.md
└── main.py
```

## Installation

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - Windows: `venv\Scripts\activate`
   - Linux or macOS: `source venv/bin/activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Verify the installation:

   ```bash
   pytest
   ```

## Usage

Run the default pipeline on `tasmax`:

```bash
python main.py
```

Limit the number of NetCDF files loaded:

```bash
python main.py --variable rainfall --max-files 12
```

Run without TensorFlow:

```bash
python main.py --skip-lstm
```

Disable synthetic fallback:

```bash
python main.py --no-synthetic-fallback
```

## B.2 Hyperparameter Configurations

### LSTM Autoencoder

| Hyperparameter | Value |
| --- | --- |
| `SEQUENCE_LENGTH` | `30` |
| `LSTM_UNITS` | `[128, 64, 64, 128]` |
| `LEARNING_RATE` | `0.001` |
| `BATCH_SIZE` | `32` |
| `EPOCHS` | `100` |
| `VALIDATION_SPLIT` | `0.2` |
| `EARLY_STOPPING_PATIENCE` | `10` |
| `DROPOUT_RATE` | `0.2` |
| `RECURRENT_DROPOUT` | `0.1` |
| `OPTIMIZER` | `adam` |
| `LOSS_FUNCTION` | `mse` |

### Isolation Forest

| Hyperparameter | Value |
| --- | --- |
| `N_ESTIMATORS` | `100` |
| `CONTAMINATION` | `0.1` |
| `MAX_SAMPLES` | `256` |
| `MAX_FEATURES` | `1.0` |
| `BOOTSTRAP` | `False` |
| `RANDOM_STATE` | `42` |
| `N_JOBS` | `-1` |

## B.3 Mathematical Formulas

### Standardization

\[
z = \frac{x - \mu}{\sigma}
\]

Where:

- `x` is the raw value
- `μ` is the mean of the training data
- `σ` is the standard deviation of the training data
- `z` is the standardized value

### Reconstruction Error (MSE)

\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2
\]

Where:

- `n` is the sequence length
- `x_i` is the original value at time step `i`
- `x̂_i` is the reconstructed value at time step `i`

### Isolation Forest Anomaly Score

\[
s(x, n) = 2^{-E[h(x)] / c(n)}
\]

\[
c(n) = 2H(n - 1) - \frac{2(n - 1)}{n}
\]

Where:

- `s(x, n)` is the anomaly score for point `x`
- `E[h(x)]` is the average path length of point `x`
- `n` is the number of external nodes
- `c(n)` is the average path length of an unsuccessful search
- `H(i)` is the harmonic number, approximated by `ln(i) + 0.5772156649`

### Evaluation Metrics

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Where `TP`, `FP`, `TN`, and `FN` denote true positives, false positives, true negatives, and false negatives.

## B.4 Software Requirements

The project dependencies are listed in `requirements.txt` and include:

- Core scientific computing: NumPy, Pandas, Xarray, SciPy
- NetCDF backends: `h5netcdf`, `netCDF4`
- Machine learning: TensorFlow, scikit-learn
- Distributed computing: Dask, Distributed
- Visualization: Matplotlib, Seaborn, Plotly
- Notebook support: Jupyter, IPykernel
- Utilities and testing: tqdm, PyYAML, python-dotenv, pytest, pytest-cov

## Outputs

Running the pipeline generates:

- `data/processed/<variable>_series.csv`
- `data/synthetic/<variable>_synthetic.csv` when fallback data is used
- `results/metrics/anomaly_results.csv`
- `results/metrics/model_metrics.csv`
- `results/metrics/summary.txt`
- `results/figures/*.png`
- `results/models/lstm_autoencoder.h5`
- `results/models/isolation_forest.pkl`
