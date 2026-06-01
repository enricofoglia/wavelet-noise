# wavelet_noise
Wavelet transform post-processing for airfoil trailing-edge noise

## Installation

### Prerequisites: SURD-states

The `surd-states` package is not on PyPI and must be cloned manually from GitHub:

```bash
git clone https://github.com/MathEXLab/SURD-states.git
```

Note the path where you cloned it — you will need it in the steps below.

---

### Using uv (recommended)

1. Clone this repository and enter it:
   ```bash
   git clone <repo-url>
   cd wavelet-noise
   ```

2. Edit `pyproject.toml` to point `surd-states` to your local clone:
   ```toml
   [tool.uv.sources]
   surd-states = { path = "/path/to/SURD-states", editable = true }
   ```

3. Install all dependencies:
   ```bash
   uv sync
   ```

To also install the documentation dependencies:
```bash
uv sync --extra doc
```

---

### Using pip

1. Install `surd-states` from your local clone:
   ```bash
   pip install -e /path/to/SURD-states
   ```

2. Install this package:
   ```bash
   pip install -e .
   ```

---

## Usage

The main script reads all parameters from a `config.yaml` file in the current directory.

### Running with uv

```bash
uv run main.py
```

Or via the installed entry point:
```bash
uv run wavelet-noise
```

### Running with Python

```bash
python main.py
```

Or, if the package is installed:
```bash
wavelet-noise
```

---

### Configuration file

Create a `config.yaml` in your working directory. Example:

```yaml
# --- I/O ---
data_dir: "/path/to/data"        # input data directory
out_dir_root: "./out"            # root output directory
case_name: "my-case.h5"         # file name (beamforming) or label (lbm)

# --- Dataset type ---
# Set to "beamforming" for experimental data, "lbm" for numerical data.
# The lbm case is only read when compute_all is false.
type: "beamforming"              # "lbm" or "beamforming"
compute_all: false               # if true, process all files in data_dir

# --- Sensor selection ---
rmp_index: 3                     # RMP sensor index to analyse
micro_index: 0                   # microphone index to analyse

# --- Physical parameters ---
p_ref: 2.0e-5                    # reference pressure [Pa]
sound_speed: 343.0               # speed of sound [m/s]
microphone_distance: 1.45        # microphone–source distance [m]

# --- Wavelet ---
wavelet: db24                    # PyWavelets wavelet name

# --- Calibration ---
calibration:
  apply: false
  nperseg: 1024
  calibration_dir: "/path/to/calibration/data"

# --- Welch PSD ---
welch:
  nperseg_factor: 3              # nperseg = nperseg_factor * 1024
  window: hann

# --- Signal conditioning ---
conditioning:
  standardize: false
  detrend: true
  detrend_degree: 0              # 0 = remove mean
  bandpass_filter:
    apply: true
    lowcut: 100.0                # [Hz]
    highcut: 10000.0             # [Hz]

# --- Plot limits ---
plots:
  frequency_range:
    min: 20.0
    max: 20000.0
  spectrum_range:                # RMP spectrum [dB]
    min: -60
    max: 80
  spectrum_micro_range:          # microphone spectrum [dB]
    min: -60
    max: 40
```

> [!NOTE]
> Set `type: beamforming` to read an experimental dataset, and `type: lbm` to read the numerical case. The `lbm` case is only read when `compute_all` is set to `false`. When analysing the `lbm` case, `case_name` is used only to name the output folder and can be set freely.

---

## Building the documentation

```bash
cd docs
make html
```

The HTML documentation can be opened from `docs/build/html/index.html`.

---
