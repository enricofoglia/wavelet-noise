import numpy as np

import scipy.signal as sg
import scipy.stats as stats
import scipy.integrate as integrate

from rich.console import Console
from rich.table import Table
from rich import box


from typing import Callable

console = Console()


def _butter_bandpass(lowcut, highcut, fs, order=5, output="sos"):
    # Butterworth bandpass filter design
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    out = sg.butter(order, [low, high], btype="band", output=output)
    return out


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, form="ba") -> np.ndarray:
    """
    Filter the data using a Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data to be filtered.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter. Default is 2.
    form : str, optional
        Form of the filter coefficients. Default is 'sos' (second-order sections).

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    out = _butter_bandpass(lowcut, highcut, fs, order=order, output=form)
    match form:
        case "sos":
            y = sg.sosfilt(out, data, axis=0)
        case "ba":
            y = sg.lfilter(out[0], out[1], data, axis=0)
        case _:
            raise ValueError("Invalid filter form. Use 'sos' or 'ba'.")
    return y


def conditioning(
    signal: np.ndarray,
    standardize: bool = False,
    detrend: bool = False,
    detrend_degree: int = 0,
    filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Perform standard conditioning on the input signal, including optional detrending and filtering.

    Arguments
    ----------
    signal : np.ndarray
        Input signal to be conditioned.
    standardize : bool, optional
        If True, standardize the signal to have zero mean and unit variance.
        Default is False.
    detrend : bool, optional
        If True, detrend the signal. Default is False.
    detrend_degree : int, optional
        Degree of the polynomial for detrending. Default is 0 (constant).
    filter : callable[[np.ndarray], np.ndarray]|None, optional
        A filtering function that takes the signal as input and returns the
        filtered signal. If None, no filtering is applied. Default is None.
    Returns
    -------
    np.ndarray
        Conditioned signal.
    """
    conditioned_signal = signal.copy()

    if detrend:
        t = np.arange(conditioned_signal.shape[0])
        pp = np.polynomial.Polynomial.fit(
            t,
            conditioned_signal,
            deg=detrend_degree,
        )
        conditioned_signal -= pp(t)

    if filter is not None:
        conditioned_signal = filter(conditioned_signal)

    if standardize:
        mean = np.mean(conditioned_signal, axis=0)
        std = np.std(conditioned_signal, axis=0)
        conditioned_signal = (conditioned_signal - mean) / std

    return conditioned_signal


def windowed_autocorrelation(
    data: np.ndarray,
    nperseg: int = 256,
    noverlap: int | None = None,
    return_std: bool = False,
) -> np.ndarray:
    """ """
    if noverlap is None:
        noverlap = nperseg // 2

    n = data.shape[0]
    step = nperseg - noverlap
    n_windows = (n - noverlap) // step

    correlations = []
    for i in range(n_windows):
        start = i * step
        end = start + nperseg
        segment = data[start:end]
        segment_avg = np.mean(segment, axis=0)
        segment_var = np.var(segment, axis=0)
        autocorr = sg.correlate(
            segment - segment_avg,
            segment - segment_avg,
            mode="full",
        )[nperseg - 1 :] / (nperseg * segment_var)
        correlations.append(autocorr)

    correlations = np.array(correlations)
    if return_std:
        return correlations.mean(axis=0), correlations.std(axis=0)
    return correlations.mean(axis=0)


def compute_integral_time_scale(
    data: np.ndarray, dt: float = 1.0, corr_threshold: float | None = 0.0
) -> float:
    """
    Compute the integral time scale of the input signal.
    """
    n = data.shape[0]
    autocorr = windowed_autocorrelation(data, nperseg=n // 4, noverlap=n // 8)

    if corr_threshold is None:
        idx = n
    else:
        idx = np.nonzero(autocorr < corr_threshold)[0][0] - 1
        if idx < 0:
            raise ValueError(
                "No value of the autocorrelation is above the correlation threshold."
            )

    integral_time_scale = integrate.simpson(autocorr[:idx], dx=dt)
    return integral_time_scale


def compute_diagnostics(
    data: np.ndarray, dt: float = 1.0, corr_threshold: float | None = 0.0
) -> dict:
    avg = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    skew = stats.skew(data, axis=0, bias=False)
    kurt = stats.kurtosis(data, axis=0, bias=False)
    t_int = compute_integral_time_scale(data, corr_threshold=corr_threshold, dt=dt)
    n_eff = round(data.shape[0] * dt / 2 / t_int)
    return {
        "mean": avg,
        "variance": var,
        "skewness": skew,
        "kurtosis": kurt,
        "integral_time_scale": t_int,
        "effective_samples": n_eff,
    }


def display_diagnostics(
    data: np.ndarray, dt: float = 1.0, corr_threshold: float | None = 0.0
) -> None:
    diagnostic = compute_diagnostics(data, dt=dt, corr_threshold=corr_threshold)

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Mean", f"{diagnostic['mean']:.4f}")
    table.add_row("Variance", f"{diagnostic['variance']:.4f}")
    table.add_row("Skewness", f"{diagnostic['skewness']:.4f}")
    table.add_row("Kurtosis", f"{diagnostic['kurtosis']:.4f}")
    table.add_row("Time step", f"{dt:.2e}")
    table.add_row("Int. Time Scale", f"{diagnostic['integral_time_scale']:.2e}")
    table.add_row("Eff. Samples", f"{diagnostic['effective_samples']:d}")

    console.print(table)


def group_rv(x: np.ndarray, b: int) -> np.ndarray:
    """Group the input data into blocks of size b and return the sum of each block."""
    if b <= 0:
        raise ValueError("Block size must be a positive integer.")
    if b == 1:
        return x
    indices = np.arange(b, len(x) - b, b, dtype=int)
    blocks = np.array_split(x, indices)
    return np.array([block.sum() for block in blocks])


def empirical_scgf(x: np.ndarray, k: np.ndarray, b: int = 100, derivative: bool = False):
    """Compute the empirical scaled cumulant generating function (SCGF) of the input data."""
    blocks = group_rv(x, b)
    scgf = np.zeros(k.shape)
    scgf_prime = np.zeros(k.shape)
    for i, k_val in enumerate(k):  # potentially parallelize
        avg = np.mean(np.exp(k_val * blocks))
        scgf[i] = np.log(avg) / b
        scgf_prime[i] = np.mean(blocks * np.exp(k_val * blocks)) / avg / b

    if derivative:
        return scgf_prime
    return scgf


def empirical_rate_func(x: np.ndarray, k: np.ndarray, b: int = 100):
    """Compute the empirical rate function of the input data."""
    blocks = group_rv(x, b)
    scgf = np.zeros_like(k)
    scgf_prime = np.zeros_like(k)
    for i, k_val in enumerate(k):  # potentially parallelize
        avg = np.mean(np.exp(k_val * blocks))
        scgf[i] = np.log(avg) / b
        scgf_prime[i] = np.mean(blocks * np.exp(k_val * blocks)) / avg / b
    I_k = k * scgf_prime - scgf
    sort_ind = np.argsort(scgf_prime)
    return scgf_prime, scgf_prime[sort_ind], I_k[sort_ind]
