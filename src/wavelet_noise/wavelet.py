import numpy as np

import pywavelets as pw


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
        scales: np.ndarray,
        **kwargs,

):

    return pw.cwt(data, scales, **kwargs)
