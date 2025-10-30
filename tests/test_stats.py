import numpy as np

from wavelet_noise import stats


def test_conditioning():
    t = np.linspace(0, 2*np.pi, 100000)

    assert np.allclose(
        stats.conditioning(t, standardize=False, detrend=False, filter=None), t
    )
    assert np.allclose(
        stats.conditioning(
            t, standardize=False, detrend=True, detrend_degree=1, filter=None
        ),
        np.zeros_like(t),
    )
    assert np.allclose(
        stats.conditioning(
            t, standardize=False, detrend=True, detrend_degree=0, filter=None
        ),
        t - np.pi
    )
    def identity_filter(x):
        return x
    assert np.allclose(
        stats.conditioning(
            t, standardize=False, detrend=False, filter=identity_filter
        ),
        t
    )
    assert np.allclose(
        stats.conditioning(
            t, standardize=True, detrend=False, filter=None
        ),
        (t-np.pi)/(1/np.sqrt(3)*np.pi)
    )
