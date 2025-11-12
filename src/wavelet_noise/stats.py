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


def compute_integral_time_scale(
    data: np.ndarray, dt: float = 1.0, corr_threshold: float | None = 0.0
) -> float:
    """
    Compute the integral time scale of the input signal.
    """
    n = data.shape[0]
    autocorr = (
        sg.correlate(data, data, mode="full")[n - 1 :] / n / data.var()
    )  # biased autocorrelation

    if corr_threshold is None:
        idx = n
    else:
        idx = np.nonzero(autocorr < corr_threshold)[0][0]
        if idx < 1:
            raise ValueError(
                "No value of the autocorrelation is above the correlation threshold."
            )

    integral_time_scale = integrate.trapezoid(autocorr[:idx], dx=dt)
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
