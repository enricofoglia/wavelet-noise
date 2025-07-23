from typing import Tuple

from warnings import warn

import numpy as np

from dataclasses import dataclass

import pywt as pw


@dataclass
class CVEResults:
    """Results of the Coherent Vortex Extraction convergence."""

    iterations: int
    final_threshold: float
    num_coherent_coeffs: int
    num_incoherent_coeffs: int
    signal: np.ndarray
    noise: np.ndarray
    incoherent_coeffs_history: list


def dwt(
    data: np.ndarray,
    wavelet: str = "db4",
    mode: str = "constant",
    level: int = None,
    fs: float = 1.0,
    axis: int = -1,
    type: str = "list",
    return_approx: bool = True
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
    type : str, optional
        Type of output ('list' for list of coefficients, 'numpy' for a
        concatenated numpy array) (default 'list').
    return_approx : bool, optional
        If True, the approximation coefficients are included in the output
        (default True).

    Returns
    -------
    tuple
        Frequencies corresponding to the wavelet scales and the DWT
         coefficients.
    """
    if level is None:
        level = pw.dwt_max_level(len(data), filter_len=wavelet)

    freq = pw.scale2frequency(wavelet=wavelet, scale=2.0 ** np.arange(1, level - 1))
    dw = pw.wavedec(data, wavelet=wavelet, mode=mode, level=level, axis=axis)

    if not return_approx:
        dw = dw[1:]

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
    data: np.array, wavelet: str, max_iter=20, tol: int = 1, use_approx: bool=True,  **kwargs
) -> Tuple[np.array, np.array]:
    """Separate the coherent and incoherent parts of a signal.

    This function uses the discrete wavelet tranform and the adaptive
     thresholding presented in Azzalini, A., Farge, M., & Schneider, K. (2005).
     Appl. Comput. Harmon. Anal., 18(2), 177-185. It is based on the hypothesis
     that the coherent part of the signal  can be accurately represented by a
     small number of large wavelet coefficients, while the incoherent part is
     represented by a large number of small wavelet coefficients.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.
    wavelet : str
        Wavelet type to use for the DWT.
    max_iter : int, optional
        Maximum number of iterations for adaptive thresholding (default 20).
    tol : int, optional
        Tolerance for the number of coefficients to consider as coherent
        (default 1).
    use_approx : bool, optional
        If True, the approximation coefficients are included in the
        thresholding process (default True).
    **kwargs : dict, optional
        Additional keyword arguments passed to the DWT function.

    Returns
    -------
    CVEResults
        A dataclass containing the results of the coherent vortex extraction,
        including the number of iterations, final threshold, number of
        coherent and incoherent coefficients, the extracted signal, noise,
        and history of incoherent coefficients.

        
    .. warning::
        This implementation of the Coherent Vortex Extraction thresholds both the detail and the approximation coefficients of the wavelet transform. In our tests, including the approximation or not did not make much of a difference in terms of the performance of the algorithm, but discretion is advised if unexpected results are obtained.
    """
    x = data - np.mean(data, axis=0)
    _, coef = dwt(x, wavelet=wavelet, mode="periodic", axis=0, type="numpy",
                  return_approx=use_approx)
    N, Ni = len(coef), len(coef)
    T = (2.0 * np.var(coef) * np.log(N)) ** 0.5

    Ni_new = Ni_new = sum(coef < T)
    it = 0
    Ni_history = [Ni]
    while (Ni_new <= Ni - tol) and it < max_iter:
        Ni = Ni_new
        coef_i = coef[np.abs(coef) < T]
        T = (2.0 * np.var(coef_i) * np.log(N)) ** 0.5

        it += 1
        Ni_new = sum(coef_i < T)
        Ni_history.append(Ni_new)
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

    _, coef_i = dwt(x, wavelet=wavelet, mode="periodic", axis=0, type="list")
    coef_i[1:] = [np.where(np.abs(c) < T, c, 0.0) for c in coef_i[1:]]
    coef_i[0] = np.zeros_like(coef_i[0])
    noise = pw.waverec(coef_i, wavelet=wavelet, mode="constant", axis=0)
    signal = data - noise

    results = CVEResults(
        iterations=it,
        final_threshold=T,
        num_coherent_coeffs=N - Ni_new,
        num_incoherent_coeffs=Ni,
        signal=signal,
        noise=noise,
        incoherent_coeffs_history=Ni_history,
    )
    return results


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
