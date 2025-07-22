import os

import numpy as np
import matplotlib.pyplot as plt
import wavelet_noise as wn

plt.style.use("../style.mplstyle")


def test_chirp():
    def gaussian(x, x0, sigma):
        return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

    def make_chirp(t, t0, a):
        frequency = (a * (t + t0)) ** 2
        chirp = np.sin(2 * np.pi * frequency * t)
        return chirp, frequency

    # generate signal
    t = np.linspace(0, 1, 1000)
    fs = 1 / (t[1] - t[0])  # Sampling frequency
    chirp1, frequency1 = make_chirp(t, 0.2, 9)
    chirp2, frequency2 = make_chirp(t, 0.1, 5)
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(t, 0.5, 0.2)

    freq = np.geomspace(2, 500, 150)
    cw = wn.wavelet.cwt(chirp, freq, wavelet="cmor1.5-1.0", fs=fs, method="fft")
    # ==============================================
    #    Plot Scaleogram
    # ==============================================
    t_grid, f_grid = np.meshgrid(t, freq)

    fig, ax = plt.subplots()
    sg = ax.pcolormesh(t_grid, f_grid, np.abs(cw), shading="nearest", cmap="bone")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$f$ [Hz]")
    fig.colorbar(sg, ax=ax, label=r"$\vert \mathcal{W}\{f\} \vert$")
    plt.show()


def test_coherent_vortex_extraction():
    # Generate a sample signal with noise
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t)
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = signal + noise

    # Apply coherent vortex extraction
    wavelet = "coif8"
    signal_extracted, noise_extracted = wn.wavelet.coherent_vortex_extraction(
        noisy_signal, wavelet=wavelet, max_iter=20, tol=1
    )

    err = np.abs(signal - signal_extracted)
    print(f"Mean error in signal extraction: {err.mean():.4f}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, noisy_signal, label="Noisy Signal")
    axs[1].plot(t, signal, label=r"Original Signal $f(t)$")
    axs[1].plot(t, signal_extracted, "k--", label=r"Denoised Signal $\tilde f(t)$")
    axs[1].plot(t, err, label=r"Error $\vert f(t)-\tilde f(t)\vert$", color="tab:red")
    axs[2].plot(t, noise, label=r"Noise $w(t)$")
    axs[2].plot(t, noise_extracted, "k--", label=r"Extracted Noise $\tilde w(t)$")

    axs[2].set_xlabel(r"$t$")
    axs[0].set_ylabel(r"$f(t)+w(t)$")
    axs[1].set_ylabel(r"$f(t),\;\tilde f(t)$")
    axs[2].set_ylabel(r"$w(t),\;\tilde w(t)$")
    for ax in axs:
        ax.legend()
        ax.grid()
        ax.set_xlim(t[0], t[-1])

    plt.tight_layout()
    plt.show()


def test_cve_trailing_edge():
    data_dir = "/home/daep/e.foglia/Documents/2A/13_gibbs/data"
    data_file = os.path.join(data_dir, "SherFWHsolid1_p_raw_data_250.h5")

    info_dict = wn.utils.get_data_info(data_file, verbose=False)
    p_te = wn.utils.extract_pressure_te(data_file, 50, info_dict["N"], False)

    # ==============================================
    #    De-normalize the data
    # ==============================================
    rho_ref = 1.225  # experiments density [kg/m^3]
    U_ref = 16  # experiments velocity [m/s]
    cref = 0.1356  # airfoil chord [m]
    p_dyn = rho_ref * U_ref**2  # dynamic pressure [Pa]
    # p_ref = 2e-5        # Reference pressure in Pa

    p_te *= p_dyn
    info_dict["T_s"] *= cref / U_ref
    info_dict["f_s"] *= U_ref / cref

    # ==============================================
    #    Basic information
    # ==============================================
    T = info_dict["T_s"]  # sample spacing
    n_sens = p_te.shape[1]  # Number of sensors

    signal = p_te[:, n_sens // 2]

    denoised_signal, noise = wn.wavelet.coherent_vortex_extraction(
        signal, wavelet="coif8", max_iter=20, tol=1
    )
    t = T * np.arange(len(signal))
    fig, ax = plt.subplots()
    ax.plot(t, signal, label="Signal")
    ax.plot(t, denoised_signal, "--k", label="Denoised signal")
    ax.plot(t, noise, color="tab:red", label="Noise")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("signal [Pa]")
    ax.set_xlim([t[0], t[-1]])
    ax.grid()
    ax.legend()

    plt.show()


if __name__ == "__main__":
    test_chirp()
    test_coherent_vortex_extraction()
    test_cve_trailing_edge()
