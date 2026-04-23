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
    

    data = conditioning(data, detrend=True, detrend_degree=1)
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
        "samples": data.shape[0],
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
    table.add_row("Total Samples", f"{diagnostic['samples']:d}")
    console.print(table)


def group_rv(x: np.ndarray, b: int) -> np.ndarray:
    """Group the input data into blocks of size b and return the sum of each block.

    Arguments
    ---------
    x: np.ndarray
        input data of size ``n``
    b: int
        block size

    Returns
    -------
    np.ndarray
        Blocks array of size ``n//b``
    """
    if b <= 0:
        raise ValueError("Block size must be a positive integer.")
    if b == 1:
        return x
    indices = np.arange(b, len(x) - b, b, dtype=int)
    blocks = np.array_split(x, indices)
    return np.array([block.sum() for block in blocks])


def empirical_scgf(x: np.ndarray, k: np.ndarray, b: int = 1, derivative: bool = False):
    r"""Compute the empirical scaled cumulant generating function (SCGF) of the input data.

    Compute cumulant generating function of the input data:

    .. math::
        \lambda(k) = \log \mathbb{E} \left(e^{kX} \right)

    For large deviation analysis of time series, it is sometimes necessary to perform a block-sum of the data to enforce independence. For examples a series :math:`X_1,X_2,\dots X_n` might not be composed of independent samples, but the series :math:`Y_1,Y_2,\dots,Y_m` where :math:`Y_i = \sum_{j=bi}^{b(i+1)}X_j` is, provided that :math:`b` is large enough.

    Arguments
    ---------
    x: np.ndarray
        input data
    k: np.ndarray
        input :math:`k` values where the SCGF will be computed
    b: int, optional
        Block size. Default is 1.
    derivative: bool, optional
        If True, return the derivative of the SCGF. Default False

    Returns
    -------
    np.ndarray
        Value of the SCGF for every value of k.
    """
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


def empirical_rate_func(x: np.ndarray, k: np.ndarray, b: int = 1):
    """Compute the empirical rate function of the input data.

    The algorithm starts by computing the cumulant generating function of the data, :math:`\lambda(k)`, using :func:`empirical_scgf`. Then, the rate function :math:`I(s)` is computed using the Legendre-Flechet transform:

    .. math::
        I(s(k)) = k s(k) - \lambda(k)

    where :math:`s(k) = \lambda'(k)`.

    .. caution::
        This algorithm only works under the hypothesis of the Gärtner-Ellis theorem, that is that the SCGF is everywhere differentiable, so that the rate function can be calculated as the Legendre transform of :math:`\lambda`. If that is not the case, the algorithm will, at best, return the convex envelope of :math:`I(s)`

    Arguments
    ---------
    x: np.ndarray
        input data
    k: np.ndarray
        input :math:`k` values where the SCGF will be computed
    b: int, optional
        Block size. Default is 1.

    Returns
    -------
    np.ndarray
        Value of the rate function for every value of k.
    """
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


def estimate_kc(num_samples: int, exponent: float) -> tuple[float, float]:
    r"""Compute an estimation of the convergence bounds :math:`k_c^{\pm}` for the SCGF estimator, based on the hypothesis that the dominant exponential term of the probability density function (pdf) of the data is of the form:

    .. math::
        p(x) \approx e^{-\vert x \vert^\rho}

    If the exponent is exactly 1 (similar to a logistic distribution), then the value :math:`k_c^{\pm}=\pm \pi/\sqrt{3}` is returned.

    For more information see :ref:`Rohwer et al. [1] <rowler2015>`

    Arguments
    ---------
    num_samples: int
        number of samples
    exponent: float
        exponent :math:`\rho` of the pdf

    Returns
    -------
    tuple[float,float]
        lower and upper bounds of the estimator

    References
    ^^^^^^^^^^

    .. _rowler2015:

    [1] Rohwer, C. M., Angeletti, F., & Touchette, H. (2015). Convergence of large-deviation estimators. Physical Review E, 92(5), 052104.
    """
    assert exponent >= 1, "The exponent should be bigger than one"
    if exponent == 1:
        return -np.pi / np.sqrt(3), np.pi / np.sqrt(3)
    else:
        return -((np.log(num_samples)) ** (1 - 1 / exponent)), (
            np.log(num_samples)
        ) ** (1 - 1 / exponent)


def structure_function(u: np.ndarray, n: int, tau: int = 1) -> float:
    r"""
    Estimate :math:`S_n(\tau) = \langle\vert\Delta u\vert^n\rangle` from a 1D time series :math:`u`, where :math:`\Delta u = u(t+\tau) - u(t)`.

    Arguments
    ---------
    u: np.ndarray
        data to analyse
    n: int
        degree of the structure function
    tau: int, optional
        spacing between samples. Default 1

    Returns
    -------
    float
        Structure function of order n.
    """
    u = np.asarray(u)
    du = u[tau:] - u[:-tau]
    return stats.moment(du, order=n)


def generalized_flatness(u: np.ndarray, n: int, tau: int = 1) -> float:
    r"""
    :math:`\sigma(n) = S_n / S_2^{n/2}`. See :func:`structure_function` for more details
    """
    S_n = structure_function(u, n, tau=tau)
    S_2 = structure_function(u, 2, tau=tau)
    return S_n / (S_2 ** (n / 2))


def wavelet_intermittency(u: np.ndarray, wavelet="coif8"):
    import pywt

    coef = pywt.wavedec(u, wavelet=wavelet)
    I = []
    for cd in coef[-1:0:-1]:
        I.append(stats.moment(cd, 4) / stats.moment(cd, 2) ** 2)

    return np.array(I)
