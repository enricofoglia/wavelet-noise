import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from wavelet_noise.utils import read_beamforming_case
from wavelet_noise.wavelet import coherent_vortex_extraction
from wavelet_noise.stats import empirical_rate_func, empirical_scgf, compute_diagnostics

from rich import print

plt.style.use("style.mplstyle")

parser = argparse.ArgumentParser()  
parser.add_argument("--file_path", type=str, help="Path to the beamforming case file.")
parser.add_argument("--fig_dir", type=str, default=".", help="Directory to save the figure.")

def _main():
    args = parser.parse_args()
    case = read_beamforming_case(args.file_path)
    mic = case.microphones[:, 29]
    signal = case.rmp[:, 0]

    diagnostics = compute_diagnostics(signal, dt=case.time[1] - case.time[0], corr_threshold=0.0)
    b = len(signal) // diagnostics["effective_samples"]
    print(f"Using block size of {b} for large deviation analysis based on effective samples")
    
    # CVE
    cve = coherent_vortex_extraction(signal, "coif8", max_iter=100, tol=1, use_approx=False)

    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")

    hydro, noise = cve.signal, cve.noise
    noise_st = (noise - np.mean(noise)) / np.std(noise)

    def gaussian_rate_func(s, mu:float=0.0,var:float=1.0):
        return (s-mu)**2 * var / 2
    def gaussian_scgf(k, var:float=1.0):
        return  var * k**2 / 2
    def gaussian_s_k(k, var:float=1.0):
        return var * k
    
    gauss_noise = np.random.normal(loc=0.0, scale=1.0, size=noise_st.shape)

    k = np.linspace(-4, 4, 250)
    scgf = empirical_scgf(noise_st, k, b=b)
    s_k, s, I = empirical_rate_func(noise_st, k, b=b)

    scgf_gauss = empirical_scgf(gauss_noise, k, b=b)
    s_k_gauss, s_gauss, I_gauss = empirical_rate_func(gauss_noise, k, b=b)

    fig, ax = plt.subplots(1,3, figsize=(12, 4), layout='tight')
    ax[0].plot(k, scgf, color="k")
    ax[0].plot(k, gaussian_scgf(k), ls='--', color="k", alpha=1.0, label="Gaussian SCGF")
    ax[0].plot(k, scgf_gauss, ls='--', color="k", alpha=0.5, label="Gaussian SCGF (emp)")
    ax[0].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$\widehat{\lambda}_L(k)$")
    ax[0].set_xlim(min(k), max(k))
    ax[0].grid()

    ax[1].plot(k, s_k, color="k")
    ax[1].plot(k, gaussian_s_k(k), ls='--', color="k", alpha=1.0, label="Gaussian s(k)")
    ax[1].plot(k, s_k_gauss, ls='--', color="k", alpha=0.5, label="Gaussian s(k) (emp)")
    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel(r"$s(k) = \widehat{\lambda}_L'(k)$")
    ax[1].set_xlim(min(k), max(k))
    ax[1].grid()

    emp = ax[2].plot(s, I, color="k", label="Empirical rate function")
    gaus = ax[2].plot(s, gaussian_rate_func(s), ls='--', label='Gaussian rate function', color="k", alpha=1.0)
    gaus_emp = ax[2].plot(s_gauss, I_gauss, ls='--', color="k", alpha=0.5, label="Gaussian rate function (emp)")
    ax[2].set_xlim(min(s), max(s))
    ax[2].set_ylim(-0.1, max(I)*1.1)
    # ax[2].legend()
    ax[2].set_xlabel(r"$s$")
    ax[2].set_ylabel(r"$\widehat{I}_L(s)$")
    ax[2].grid()
    fig.legend([emp[0], gaus[0], gaus_emp[0]], ["Data", "Gaussian (analytical)", "Gaussian (empirical)"], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncols=3, frameon=True, fontsize=12)
    plt.savefig(os.path.join(args.fig_dir, "large_deviations.pdf"), dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    _main()