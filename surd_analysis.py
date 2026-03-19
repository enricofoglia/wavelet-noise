import argparse

import numpy as np
import matplotlib.pyplot as plt

from wavelet_noise.utils import read_beamforming_case
from wavelet_noise.wavelet import coherent_vortex_extraction

from surd_states import surd
from surd_states import it_tools as it

plt.style.use("style.mplstyle")
my_colors ={
    "coherent": "#1A5B5B",
    "incoherent": "#F4AB5C",
    "synergistic": "#ACC8BE",

}

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, help="Path to the beamforming case file.")
parser.add_argument("--rmp", type=int, help="RMP index. Can be 0, 1, 2, or 3", default=0)


def _main():
    args = parser.parse_args()
    case = read_beamforming_case(args.file_path)
    mic = case.microphones[:, 29]
    signal = case.rmp[:, args.rmp]
    time = case.time

    # CVE
    cve = coherent_vortex_extraction(
        signal, "coif8", max_iter=100, tol=1, use_approx=False
    )

    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")

    hydro, noise = cve.signal, cve.noise

    X = np.stack([mic, hydro, noise], axis=0)

    # compute bins for histogram
    nbins = 25
    max_abs = np.percentile(X, 99.99)
    max_abs = np.floor(max_abs)
    bin_width = 2 * max_abs / (nbins - 1)
    bins_list = []
    for _i in range(X.shape[0]):
        bins = np.linspace(-max_abs, max_abs + bin_width, nbins + 1)
        bins_list.append(bins)

       
# === Causality analysis across lags ===
    # apply time delay to microphone signal
    sound_speed = np.sqrt(1.4 * 287.05 * (273.15 + 22))  # Speed of sound at 22 C
    delay = 1.45 / sound_speed
    sound_lag = int(delay / (time[1] - time[0]))
    print(
        f"Propagation delay based on microphone distance and speed of sound: {delay:.4f} seconds."
    )

    # Select delta T
    max_lag = sound_lag * 2
    nlags_range = range(1, max_lag, 1)
    num_lags = len(nlags_range)
    unique_lag = np.zeros((num_lags,2), dtype=np.float64)
    syn_lag = np.zeros(num_lags, dtype=np.float64)

    for n_idx, nlag in enumerate(nlags_range):
        _Y = np.vstack([X[0, nlag:], X[1:, :-nlag]])
        _hist, _ = np.histogramdd(
            _Y.T, bins=[bins_list[0], bins_list[1], bins_list[2]]
        )
        I_R, I_S, MI, _info_leak, *_ = surd.surd_states(_hist)  # Prepare lagged joint data
        H = it.entropy_nvars(_hist, (0,))
        unique_lag[n_idx, 0] = I_R[(1,)] / H
        unique_lag[n_idx, 1] = I_R[(2,)] / H
        syn_lag[n_idx] = I_S[(1,2)] / H
    _fig, ax = plt.subplots()
    ax.plot(time[nlags_range], unique_lag[:, 0], label="Unique coherent", color=my_colors["coherent"])
    ax.plot(time[nlags_range], unique_lag[:, 1], label="Unique incoherent", color=my_colors["incoherent"])
    ax.plot(time[nlags_range], syn_lag, label="Synergistic", color=my_colors["synergistic"])
    ax.grid()
    ax.set_xlabel(r"$\Delta t$ (s)")
    ax.set_ylabel("Normalized Information")
    ax.set_xlim(0, time[max_lag])
    ax.legend()
    plt.tight_layout()

    # get best time lag based on unique causality to hydrodynamic signal
    best_idx = np.argmax(unique_lag[:, 0])
    nlag = nlags_range[best_idx]
    print(f"Best time lag for unique causality to hydrophone signal: {nlag} samples, corresponding to {time[nlag]:.4f} seconds.")

    _fig, ax = plt.subplots(figsize=(10, 4))
    print(f"INFORMATION FLUX FOR MICROPHONE SIGNAL")
    Y = np.vstack([X[0, nlag:], X[1:, :-nlag]])
    hist, _ = np.histogramdd(
        Y.T, bins=[bins_list[0], bins_list[1], bins_list[2]]
    )
    Rd, Sy, mi, info_leak, rd_states, u_states, sy_states = surd.surd_states(hist)
    surd.nice_print(Rd, Sy, mi, info_leak)

    heights = np.array([Rd[(1,2)], Rd[(1,)], Rd[(2,)], Sy[(1,2)]])
    heights /= sum(heights)  
    labels = ["Redundant", "Unique coherent", "Unique incoherent", "Synergistic"]
    ax.bar(labels, heights, 
           color=["gray", my_colors["coherent"], my_colors["incoherent"], my_colors["synergistic"]],
            lw=2,  edgecolor="black")
    ax.set_ylabel("Information Fraction")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    _main()
