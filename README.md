# wavelet_noise
Wavelet transform post-processing for airfoil trailing-edge noise

> [!IMPORTANT]
> The package SURD-states in not on PyPI at the moment. To use it, clone it directly from github and provide the relative path in the pyproject.toml file.

## Building the doc
Once the code has been installed it is possible to build the documentation by:

```
cd docs
make html
```

The `.html` documentation can be opened in any brower from the `docs/build/html/index.html` file.


## Calibration 

```
rmp_locations = [2, 3, 5, 6, 22, 23, 24, 25, 26, 27]

for rmp in rmp_locations:
    file_path_calibration = '/storage/renj3003/cd-airfoil/exp/data/rmp/02-CALIBRATION/01-SPSPEAKER'
    file_name_calibration = f'rmp{rmp}_data.txt'

    file_path_experiments = '/storage/renj3003/cd-airfoil/exp/data/rmp/01-ISAE-BEAMFORMING'
    file_name_experiments = f'CD-ISAE-5deg-rmp{rmp}.txt'

    p_ref, p_rmp = load_calibration_data(file_path_calibration, file_name_calibration)
    time, p_target = load_experimental_data(file_path_experiments, file_name_experiments)
    f, H, Cxy, delay, group_delay, p_calibrated_t = calibrate_rmp_signal(p_ref, p_rmp, p_target, time)

    f,Pxx = signal.welch(p_calibrated_t, fs=1/(time[1] - time[0]), nperseg=1024)
```
