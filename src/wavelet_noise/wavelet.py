from typing import Tuple

from warnings import warn

import numpy as np

import pywt as pw


def dwt(
    data: np.ndarray,
    wavelet: str = "db4",
    mode: str = "constant",
    level: int = None,
    fs: float = 1.0,
    axis: int = -1,
    type: str = "list",
):
    """
    Discrete Wavelet Transform (DWT) using PyWavelets. Also, returns the
     frequencies corresponding to the wavelet scales.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.
    wavelet : str, optional
        Wavelet type to use (default 'db4').
    mode : str, optional
        Signal extension mode (default 'constant').
    level : int, optional
        Level of decomposition. If None, the maximum level is computed based
        on the length of the data and the wavelet filter length.
    fs : float, optional
        Sampling frequency of the data (default 1.0).
    axis : int, optional
        Axis along which to compute the DWT (default -1, which is the last
         axis).

    Returns
    -------
    tuple
        Frequencies corresponding to the wavelet scales and the DWT
         coefficients.
    """
    if level is None:
        level = pw.dwt_max_level(len(data), filter_len=wavelet)

    freq = pw.scale2frequency(wavelet=wavelet,
                              scale=2.0 ** np.arange(1, level - 1))
    dw = pw.wavedec(data, wavelet=wavelet, mode=mode, level=level, axis=axis)

    if type == "numpy":
        dw = _dwt2numpy(dw)
    return freq * fs, dw


def idwt(
    coeffs: tuple,
):
    pass


def cwt(
    data: np.ndarray,
    freq: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    fs: float = 1.0,
    **kwargs,
):
    """
    Continuous Wavelet Transform (CWT) using PyWavelets. Automatically convert
    frequencies to scales based on the specified wavelet type.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.
    freq : np.ndarray
        Frequencies at which to compute the CWT.
    wavelet : str, optional
        Wavelet type to use (default 'cmor1.5-1.0').
    fs : float, optional
        Sampling frequency of the data (default 1.0).
    **kwargs : dict, optional
        Additional keyword arguments passed to the PyWavelets CWT function.

    Returns
    -------
    np.ndarray
        CWT coefficients of the input signal at the specified frequencies.
    """
    scales = pw.frequency2scale(wavelet=wavelet, freq=freq / fs, precision=8)
    cw, _ = pw.cwt(data, scales, wavelet, **kwargs)
    return cw


def coherent_vortex_extraction(
    data: np.array, wavelet: str, max_iter=20, tol: int = 1
) -> Tuple[np.array, np.array]:

    x = data - np.mean(data, axis=0)
    _, coef = dwt(x, wavelet=wavelet, mode="constant", axis=0, type="numpy")
    N, Ni = len(data), len(data)
    T = (2.0 * np.var(coef) * np.log(N)) ** 0.5

    Ni_new = 0
    it = 0
    while (Ni_new <= Ni - tol) and it < max_iter:
        coef_i = coef[coef < T]
        T = (2.0 * np.var(coef_i) * np.log(N)) ** 0.5

        it += 1
        Ni = Ni_new
        Ni_new = sum(coef_i < T)
        if iter == max_iter:
            warn(
                f"Maximum iterations reached: {max_iter}.\n"
                f"Delta N = {Ni - Ni_new} > {tol}.\n"
                "Consider increasing the number of iterations or decreasing"
                " the tolerance."
            )

        if Ni_new == 0:
            warn(
                "No coherent vortices found. Consider adjusting the "
                "tolerance or wavelet parameters."
            )
            return np.array([]), np.array([])

    _, coef_i = dwt(x, wavelet=wavelet, mode="constant", axis=0, type="list")
    coef_i[1:] = [np.where(np.abs(c) < T, 0.0, c) for c in coef_i[1:]]
    noise = pw.waverec(coef_i, wavelet=wavelet, mode="constant", axis=0)
    signal = data - noise
    return signal, noise


def _count_coef(coef: list, T: float = 0.0, details_only: bool = True) -> int:
    """Compute the number of coefficients bigger than T in a list of wavelet
    coefficients returned by wavedec."""

    if details_only:
        coef_ = coef[1:]
    else:
        coef_ = coef
    return sum([sum(np.abs(c) > T) for c in coef_])


def _dwt2numpy(dwt_coef: list):
    """Convert a list of wavelet coefficients returned by wavedec to a numpy
    array."""
    return np.concatenate([c.flatten() for c in dwt_coef])
