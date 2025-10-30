import numpy as np
import scipy.signal as sg

from typing import Callable


def _butter_bandpass(lowcut, highcut, fs, order=5, output="sos"):
    # Butterworth bandpass filter design
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    out = sg.butter(order, [low, high], btype="band", output=output)
    return out


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, form="ba"):
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
    filter: Callable[[np.ndarray], np.ndarray] = None,
):
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
