import numpy as np
import matplotlib.pyplot as plt

import wavelet_noise.stats as stats

plt.style.use("../style.mplstyle")


def test_spectrum():
    t = np.linspace(0, 10, 10000) 
    signal = np.sin(2 * np.pi * 50 * t) + 0.02* np.random.randn(10000)  

    fs = 1 / (t[1] - t[0])  # Sampling frequency
    f, spp = stats.spectrum(signal[:, np.newaxis], fs=fs, nperseg=256, filter=False, flims=(0.1, 10.0),
                            avg=1, axis=0)
    assert f.shape[0] > 0, "Frequency array should not be empty."
    assert spp.shape[0] > 0, "Power spectral density array should not be empty."

    fig, ax = plt.subplots()
    ax.loglog(f, spp)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$S(f)$ [V$^2$/Hz]")
    ax.grid()

    plt.show()




if __name__ == "__main__":
    test_spectrum()