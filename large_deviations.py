import argparse
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


from wavelet_noise.utils import read_beamforming_case
from wavelet_noise.wavelet import coherent_vortex_extraction
from wavelet_noise.stats import empirical_rate_func, empirical_scgf, compute_diagnostics, estimate_kc, generalized_flatness, wavelet_intermittency

from rich import print

plt.style.use("style.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, help="Path to the beamforming case file.")
parser.add_argument(
    "--fig_dir", type=str, default=".", help="Directory to save the figure."
)
parser.add_argument("--hydro", action="store_true", help="Whether to analyze the hydrodynamic signal instead of the noise.")
parser.add_argument("--b", type=int, help="Block size for large deviation analysis.")
parser.add_argument("--rmp", type=int, help="RMP index. Can be 0, 1, 2, or 3", default=0)
# colors
my_colors = {"biennale_red": "#D21C2D"}


def _main():
    args = parser.parse_args()
    print(args.rmp)
    case = read_beamforming_case(args.file_path)
    mic = case.microphones[:, 29]
    signal = case.rmp[:, args.rmp]

   
    # CVE
    cve = coherent_vortex_extraction(
        signal, "coif8", max_iter=100, tol=1, use_approx=False
    )

    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")

    hydro, noise = cve.signal, cve.noise
    noise_st = (noise - np.mean(noise)) / np.std(noise)
    if args.hydro:
        hydro_st = (hydro - np.mean(hydro)) / np.std(hydro)
        noise_st = hydro_st

    diagnostics = compute_diagnostics(
        noise_st, dt=case.time[1] - case.time[0], corr_threshold=0.0
    )
    b = len(signal) // diagnostics["effective_samples"]
    if args.b is not None:
        b = args.b
    print(
        f"Using block size of {b} for large deviation analysis based on effective samples"
    )
    def gaussian_rate_func(s, mu: float = 0.0, var: float = 1.0):
        return (s - mu) ** 2 * var / 2

    def gaussian_scgf(k, var: float = 1.0):
        return var * k**2 / 2

    def gaussian_s_k(k, var: float = 1.0):
        return var * k

    def logi_scgf(k, s: float = np.sqrt(3) / np.pi):
        return np.log(np.pi * s * k / np.sin(np.pi * s * k))

    def logi_s_k(k, s: float = np.sqrt(3) / np.pi):
        return -s * np.pi * (1 / np.tan(s * np.pi * k) - 1 / (s * np.pi * k))

    def logi_rate_func(k, s: float = np.sqrt(3) / np.pi):
        return logi_s_k(k, s) * k - logi_scgf(k, s)

    gauss_noise = np.random.normal(loc=0.0, scale=1.0, size=noise_st.shape)
    logi_noise = np.random.logistic(loc=0.0, scale=np.sqrt(3) / np.pi, size=noise_st.shape)

    k_c = np.pi / np.sqrt(3)
    k_c_pm = estimate_kc(len(noise_st), 2.0)
    print(f"Estimated limits k_c_pm = {k_c_pm[0]:.2f}, {k_c_pm[1]:.2f}")
    k = np.linspace(k_c_pm[0], k_c_pm[1], 250)
    s_c = empirical_scgf(noise_st, np.array([k_c]), b=b, derivative=True)
    k_logi = np.linspace(-k_c, k_c, 250)[1:-1]  # avoid singularities at the endpoints

    scgf = empirical_scgf(noise_st, k, b=b)
    s_k, s, I = empirical_rate_func(noise_st, k, b=b)

    scgf_gauss = empirical_scgf(gauss_noise, k, b=b)
    s_k_gauss, s_gauss, I_gauss = empirical_rate_func(gauss_noise, k, b=b)
    scgf_logi = empirical_scgf(logi_noise, k_logi, b=b)
    s_k_logi, s_logi, I_logi = empirical_rate_func(logi_noise, k_logi, b=b)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="tight")
    ax[0].plot(k, scgf, color="k")
    ax[0].plot(
        k, gaussian_scgf(k), ls="--", color="k", alpha=1.0, label="Gaussian SCGF"
    )
    ax[0].plot(
        k, scgf_gauss, ls="--", color="k", alpha=0.5, label="Gaussian SCGF (emp)"
    )
    ax[0].plot(
        k_logi, scgf_logi, label="Logistic SCGF (emp)"
    )
    ax[0].plot(
        k_logi, logi_scgf(k_logi), ls=":", color="k", alpha=1.0, label="Logistic SCGF"
    )
    ax[0].axvline(-k_c, color="grey", linewidth=1, zorder=0)
    ax[0].axvline(k_c, color="grey", linewidth=1, zorder=0)
    ax[0].fill_between(
        k,
        min(scgf) * 1.1,
        max(scgf) * 1.1,
        where=(k <= -k_c) | (k >= k_c),
        color="grey",
        alpha=0.1,
    )
    ax[0].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$\widehat{\lambda}_L(k)$")
    ax[0].set_xlim(min(k), max(k))
    ax[0].set_ylim(min(scgf) * 1.1, max(scgf) * 1.1)
    ax[0].grid()

    ax[1].plot(k, s_k, color="k")
    ax[1].plot(k, gaussian_s_k(k), ls="--", color="k", alpha=1.0, label="Gaussian s(k)")
    ax[1].plot(k, s_k_gauss, ls="--", color="k", alpha=0.5, label="Gaussian s(k) (emp)")
    ax[1].plot(k_logi, s_k_logi, label="Logistic s(k) (emp)")
    ax[1].plot(
        k_logi, logi_s_k(k_logi), ls=":", color="k", alpha=1.0, label="Logistic s(k)"
    )
    ax[1].axvline(-k_c, color="grey", linewidth=1, zorder=0)
    ax[1].axvline(k_c, color="grey", linewidth=1, zorder=0)
    ax[1].fill_between(
        k,
        min(s_k) * 1.1,
        max(s_k) * 1.1,
        where=(k <= -k_c) | (k >= k_c),
        color="grey",
        alpha=0.1,
    )

    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel(r"$s(k) = \widehat{\lambda}_L'(k)$")
    ax[1].set_xlim(min(k), max(k))
    ax[1].set_ylim(min(s_k) * 1.1, max(s_k) * 1.1)
    ax[1].grid()

    emp = ax[2].plot(s, I, color="k", label="Empirical rate function")
    gaus = ax[2].plot(
        s,
        gaussian_rate_func(s),
        ls="--",
        label="Gaussian rate function",
        color="k",
        alpha=1.0,
    )
    gaus_emp = ax[2].plot(
        s_gauss,
        I_gauss,
        ls="--",
        color="k",
        alpha=0.5,
        label="Gaussian rate function (emp)",
    )
    logi_emp = ax[2].plot(
        s_logi,
        I_logi,
        label="Logistic rate function (emp)",
    )
    sort_ind = np.argsort(logi_s_k(k_logi))
    s_logi = logi_s_k(k_logi)[sort_ind]
    I_logi = logi_rate_func(k_logi)[sort_ind]
    logi = ax[2].plot(
        s_logi, I_logi, ls=":", color="k", alpha=1.0, label="Logistic rate function"
    )
    ax[2].axvline(-s_c, color="grey", linewidth=1, zorder=0)
    ax[2].axvline(s_c, color="grey", linewidth=1, zorder=0)
    ax[2].fill_between(
        s,
        min(I) * 1.1,
        max(I) * 1.1,
        where=(s <= -s_c) | (s >= s_c), 
        color="grey",
        alpha=0.1,
    )
    ax[2].set_xlim(min(s), max(s))
    ax[2].set_ylim(-0.1, max(I) * 1.1)
    # ax[2].legend()
    ax[2].set_xlabel(r"$s$")
    ax[2].set_ylabel(r"$\widehat{I}_L(s)$")
    ax[2].grid()
    fig.legend(
        [emp[0], gaus[0], gaus_emp[0], logi[0]],
        [
            "Data",
            "Gaussian (analytical)",
            "Gaussian (empirical)",
            "Logistic (analytical)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncols=4,
        frameon=False,
        fontsize=12,
        facecolor="white",
        fancybox=False,
    )
    plt.savefig(
        os.path.join(args.fig_dir, "large_deviations.pdf"), dpi=300, bbox_inches="tight"
    )

    bins = np.linspace(-5, 5, 100)
    hist, _ = np.histogram(noise_st, bins=bins, density=True)
    I_hist = -np.log(hist)
    I_hist -= I_hist.min()  # shift so that the minimum is at zero

    fig, ax = plt.subplots(figsize=(6, 6), layout="tight")
    ax.plot(s, gaussian_rate_func(s), ls="--", color="k", alpha=0.5, linewidth=6, label="Gaussian rate function")
    ax.plot(bins[:-1], I_hist, ls="--",color="k", label="Empirical rate function (hist)")
    ax.plot(s_logi, I_logi, ls="-", linewidth=6, color="k", alpha=0.5, label="Logistic rate function")
    ax.plot(s, I, color="k", ls="-", label="Empirical rate function (SCGF)")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\widehat{I}_L(s)$")
    ax.set_xlim(min(s), max(s))
    # ax.set_xlim(-6,6)
    ax.set_ylim(-0.1, max(I) * 1.1)
    ax.legend()
    ax.grid()
    plt.savefig(
        os.path.join(args.fig_dir, "large_deviations_hist.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="tight")
    n_steps = 10
    # Sample gray_r from 0.1 to 1.0 to avoid pure white at the low end
    colors = cm.gray_r(np.linspace(0.1, 1.0, n_steps)) 
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(0.5, n_steps + 1.5, 1)  # boundaries between discrete levels
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for m in range(n_steps, 0, -1):
        bm = b  * m
        scgf = empirical_scgf(noise_st, k, b=bm)
        s_k, s, I = empirical_rate_func(noise_st, k, b=bm)

        ax[0].plot(k, scgf, color="k", alpha=0.1 * m)
        ax[1].plot(k, s_k, color="k", alpha=0.1 * m)
        ax[2].plot(s, I, color="k", alpha=0.1 * m)

    ax[0].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$\widehat{\lambda}_L(k)$")
    ax[0].set_xlim(min(k), max(k))
    ax[0].grid()

    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel(r"$s(k) = \widehat{\lambda}_L'(k)$")
    ax[1].set_xlim(min(k), max(k))
    ax[1].grid()

    ax[2].set_xlim(min(s), max(s))
    ax[2].set_ylim(-0.1, max(I) * 1.1)
    ax[2].set_xlabel(r"$s$")
    ax[2].set_ylabel(r"$\widehat{I}_L(s)$")
    ax[2].grid()

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        sm, cax=cax, ticks=np.arange(1, n_steps + 1), label=r"$\Delta t / \tau_c$"
    )
    plt.savefig(
        os.path.join(args.fig_dir, "large_deviations_b_convergence.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    fig, ax = plt.subplots()
    ax.hist(noise_st, bins=bins, density=False, color="k", alpha = 0.1, histtype="stepfilled")
    ax.hist(noise_st, bins=bins, density=False, color="k", alpha = .8, histtype="step", linewidth=2)
    ax.set_ylabel("Count")
    ax.set_xlabel(r"$p_{\mathrm{noise}}$ (standardized)")
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")


    fig, ax = plt.subplots()
    tau = np.logspace(1,4,100)
    tau = np.unique(np.round(tau).astype(int))
    dt = 1/ case.fs 
    intermittency = []
    for _tau in tau:
        intermittency.append(generalized_flatness(noise_st, n=4, tau=_tau))

    intermittency = np.array(intermittency)
    ax.plot(tau * dt ,intermittency, color="k")
    ax.axhline(3.0, linestyle="--", color=my_colors["biennale_red"])
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"$\sigma(4)$ [-]")
    ax.grid(which="major")
    ax.set_xscale("log")


    fig, ax = plt.subplots()
    wavelet = ["db"+str(2*i) for i in range(1,11)]
    for i ,wav in enumerate(wavelet):
        intermittency = wavelet_intermittency(noise_st , wavelet = wav)

        ax.plot(intermittency, color="k", alpha=i*0.1)
    ax.axhline(3.0, linestyle="--", color=my_colors["biennale_red"])
    ax.set_xlabel(r"Scale index")
    ax.set_ylabel(r"$\sigma_w(4)$ [-]")
    ax.grid(which="major")
    plt.show()


if __name__ == "__main__":
    _main()
