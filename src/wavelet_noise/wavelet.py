import numpy as np

import pywt as pw


def dwt(
        data: np.ndarray,
        wavelet: str = 'db4',
        mode: str = 'constant',
        level: int = None,
        fs: float = 1.0,
        axis: int = -1,
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

    freq = pw.scale2frequency(wavelet=wavelet, scale=2**np.arange(1, level-1))
    dw = pw.wavedec(data, wavelet=wavelet, mode=mode, level=level, axis=axis)
    return freq * fs, dw


def idwt(
        coeffs: tuple,
):
    pass


def cwt(
        data: np.ndarray,
        freq: np.ndarray,
        wavelet: str = 'cmor1.5-1.0',
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
    scales = pw.frequency2scale(wavelet=wavelet, freq=freq/fs, precision=8)
    cw, _ = pw.cwt(data, scales, wavelet, **kwargs)
    return cw
