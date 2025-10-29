import os

import yaml

import numpy as np

import scipy.signal as sg

import matplotlib.pyplot as plt
import wavelet_noise as wn

# plt.style.use("style.mplstyle")
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "lines.linewidth": 2,
        "savefig.dpi": 300,
    }
)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    data = wn.utils.read_beamforming_case(
        os.path.join(config["data_dir"], config["case_name"])
    )

    signal = data.rmp[:, 0]

    welch_kwargs = {
        "fs":data.fs,
        "nperseg": signal.shape[0] // 2**6,
        "noverlap": signal.shape[0] // 2**7,
        "window": "hamming"
    }
    
    f, psd = sg.welch(
        signal,
        **welch_kwargs
    )

    cve = wn.wavelet.coherent_vortex_extraction(
        signal,
        wavelet="coif8",
        max_iter=100,
        tol=1,
        use_approx=False
    )

    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold:.4e}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")
    print(f"Fraction of coherent coefficients: {cve.num_coherent_coeffs / (cve.num_coherent_coeffs + cve.num_incoherent_coeffs):.2%}")

    psd_noise = sg.welch(
        cve.noise,
        **welch_kwargs
    )[1]

    psd_hydro = sg.welch(
        cve.signal,
        **welch_kwargs
    )[1]

   


    p_ref = config.get("p_ref", 20e-6)

    fig, ax = plt.subplots()
    ax.plot(cve.incoherent_coeffs_history, "-o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of incoherent coefficients")
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(config["out_dir"], "cve_convergence.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10*np.log10(psd/p_ref), label="Original signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_original.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10*np.log10(psd_hydro/p_ref), label="Original signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_denoised.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10*np.log10(psd_noise/p_ref), label="Original signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_noise.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10*np.log10(psd/p_ref), label="Original signal")
    ax.semilogx(f, 10*np.log10(psd_hydro/p_ref), label="Hydrodynamic component")
    ax.semilogx(f, 10*np.log10(psd_noise/p_ref), label="Noise component")


    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    ax.legend()
    plt.savefig(os.path.join(config["out_dir"], "psd_comparison.png"))

    nsamples = 1000
    fig, ax = plt.subplots()
    ax.plot(data.time[:nsamples], signal[:nsamples], label="Original signal")
    ax.plot(data.time[:nsamples], cve.signal[:nsamples], "--", label="Hydrodynamic component")
    ax.plot(data.time[:nsamples], cve.noise[:nsamples], label="Noise component")
    ax.plot(data.time[:nsamples], cve.signal[:nsamples] + cve.noise[:nsamples], "--", label="Reconstructed signal")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure fluctuation [Pa]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()
    plt.savefig(os.path.join(config["out_dir"], "time_signal_comparison.png"))

    def standard_gaussian(x):
        return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x**2)

    def standard_logistic(x):
        return np.pi/np.sqrt(3)*np.exp(-x*np.pi/np.sqrt(3)) / (1.0 + np.exp(-x*np.pi/np.sqrt(3)))**2

    fig, ax = plt.subplots()
    ax.hist(signal/signal.std(), bins=100, alpha=1.0, label="Original signal", density=True, histtype="stepfilled", zorder=1)
    ax.hist(cve.signal/cve.signal.std(), bins=100, alpha=1.0, label="Hydrodynamic component", density=True, histtype="step", linewidth=2.0, zorder=2)
    ax.hist(cve.noise/cve.noise.std(), bins=100, alpha=1.0, label="Noise component", density=True, zorder=3)
    ax.plot(np.linspace(-10.0,10.0,250), standard_logistic(np.linspace(-10.0,10.0,250)), "--", color="tomato", label="Standard Logistic", zorder=4)
    ax.set_xlabel(r"Pressure fluctuation / $\sigma$ [-]")
    ax.set_ylabel("Probability density")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(config["out_dir"], "pdf_comparison.png"))

    fig, ax = plt.subplots()
    ax.hist(cve.noise, bins=100, alpha=1.0, label="Noise component", density=True)
    ax.set_xlabel("Pressure fluctuation [Pa]")
    ax.set_ylabel("Probability density")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(config["out_dir"], "pdf_noise.png"))



    

if __name__ == "__main__":
    main()
