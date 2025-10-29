import numpy as np
import matplotlib.pyplot as plt

from wavelet_noise import wavelet


def test_coherent_vortex_extraction():
    t = np.linspace(0, 1, 1024, endpoint=False)
    data_clean = np.sin(t * 50)
    data = data_clean + 0.01 * np.random.normal(size=t.shape)
    cve = wavelet.coherent_vortex_extraction(data, wavelet="db4", max_iter=10, tol=1e-3)

    assert hasattr(cve, "signal")
    assert hasattr(cve, "noise")
    assert cve.signal.shape == data.shape
    assert cve.noise.shape == data.shape
    assert np.allclose(data, cve.signal + cve.noise)
    assert np.allclose(data_clean, cve.signal, atol=5e-2)
