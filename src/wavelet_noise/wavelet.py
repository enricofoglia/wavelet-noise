import numpy as np

import pywt as pw


def dwt(
        data: np.ndarray,
):
    pass


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
    Continuous Wavelet Transform (CWT) using PyWavelets.
    """
    scales = pw.frequency2scale(wavelet=wavelet, freq=freq/fs, precision=8)
    cw, _ = pw.cwt(data, scales, wavelet, **kwargs)
    return cw
