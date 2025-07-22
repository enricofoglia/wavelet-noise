import numpy as np
import matplotlib.pyplot as plt
import wavelet_noise as wn


def test_chirp():
    def gaussian(x, x0, sigma):
        return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

    def make_chirp(t, t0, a):
        frequency = (a * (t + t0)) ** 2
        chirp = np.sin(2 * np.pi * frequency * t)
        return chirp, frequency

    # generate signal
    t = np.linspace(0, 1, 1000)
    fs = 1/(t[1] - t[0])  # Sampling frequency
    chirp1, frequency1 = make_chirp(t, 0.2, 9)
    chirp2, frequency2 = make_chirp(t, 0.1, 5)
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(t, 0.5, 0.2)

    freq = np.geomspace(2, 500, 150)
    cw = wn.wavelet.cwt(chirp, freq, wavelet='cmor1.5-1.0', fs=fs,
                        method='fft')
    # ==============================================
    #    Plot Scaleogram
    # ==============================================
    t_grid, f_grid = np.meshgrid(t, freq)

    fig, ax = plt.subplots()
    sg = ax.pcolormesh(t_grid, f_grid, np.abs(cw), shading='nearest',
                       cmap='bone')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$f$ [Hz]')
    fig.colorbar(sg, ax=ax, label=r'$\vert \mathcal{W}\{f\} \vert$')
    plt.show()


def test_coherent_vortex_extraction():
    # Generate a sample signal with noise
    t = np.linspace(0, 1, 1000)
    fs = 1000  # Sampling frequency
    signal = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.5, len(t))

    # Apply coherent vortex extraction
    wavelet = 'db4'
    signal_extracted, noise = wn.wavelet.coherent_vortex_extraction(
        signal, wavelet=wavelet, max_iter=20, tol=1)

    # Plot the original signal, extracted signal, and noise
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='Original Signal', alpha=0.5)
    plt.plot(t, signal_extracted, label='Extracted Signal', color='orange')
    plt.plot(t, noise, label='Noise', color='red')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Coherent Vortex Extraction')
    plt.show()

if __name__ == "__main__":
    test_chirp()
    test_coherent_vortex_extraction()