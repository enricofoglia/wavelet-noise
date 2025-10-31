import numpy as np

from wavelet_noise import stats

np.random.seed(42)


def test_conditioning():
    t = np.linspace(0, 2 * np.pi, 100000)

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
        t - np.pi,
    )

    def identity_filter(x):
        return x

    assert np.allclose(
        stats.conditioning(t, standardize=False, detrend=False, filter=identity_filter),
        t,
    )
    assert np.allclose(
        stats.conditioning(t, standardize=True, detrend=False, filter=None),
        (t - np.pi) / (1 / np.sqrt(3) * np.pi),
    )


def test_compute_diagnostic():
    # should use more studied convergence of the estimators of the moments
    size = 10000
    data1D = np.random.normal(loc=2.0, scale=np.sqrt(2.0), size=size)
    diagnostics1D = stats.compute_diagnostics(data1D, corr_threshold=None)
    assert np.isclose(diagnostics1D["mean"], 2.0, atol=10 / np.sqrt(size))
    assert np.isclose(diagnostics1D["variance"], 2.0, atol=10 / np.sqrt(size))

    assert np.isclose(diagnostics1D["skewness"], 0.0, atol=10 / np.sqrt(size))
    assert np.isclose(diagnostics1D["kurtosis"], 0.0, atol=10.0 / np.sqrt(size))


def test_windowed_autocorrelation():
    size = 10000
    data1D = np.random.normal(loc=0.0, scale=1.0, size=size)
    autocorr = stats.windowed_autocorrelation(data1D, nperseg=1000, noverlap=500)
    assert autocorr.shape[0] == 1000
    assert np.isclose(autocorr[0], 1.0, atol=1e-2)
    assert np.all(autocorr[1:] < 0.2)
