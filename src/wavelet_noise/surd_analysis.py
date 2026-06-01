import argparse

import numpy as np
import matplotlib.pyplot as plt

from surd_states import surd
from surd_states import it_tools as it

DEFAULT_COLORS = {
    "coherent": "#1A5B5B",
    "incoherent": "#F4AB5C",
    "synergistic": "#ACC8BE",
    "redundant": "gray",
}


def _compute_bins(X, nbins):
    max_abs = np.floor(np.percentile(X, 99.99))
    bin_width = 2 * max_abs / (nbins - 1)
    return [
        np.linspace(-max_abs, max_abs + bin_width, nbins + 1)
        for _ in range(X.shape[0])
    ]


def lag_sweep(X, sound_lag, nbins=25):
    """Compute SURD components across a range of time lags.

    Parameters
    ----------
    X : ndarray, shape (3, N)
        Stacked signals as [target, source1, source2] (e.g. [mic, hydro, noise]).
    sound_lag : int
        Number of samples for the acoustic propagation delay. The sweep runs
        from lag 1 to 2 * sound_lag.
    nbins : int
        Number of histogram bins (default 25).

    Returns
    -------
    lags : range
        Lag indices used (pass directly to surd_states_at_lag / plot_surd_results).
    unique : ndarray, shape (num_lags, 2)
        Unique information fractions [source1, source2], normalised by H(target).
    redundant : ndarray, shape (num_lags,)
        Redundant information fraction.
    synergistic : ndarray, shape (num_lags,)
        Synergistic information fraction.
    bins_list : list of ndarray
        Bin edges for each variable; pass to surd_states_at_lag.
    """
    bins_list = _compute_bins(X, nbins)
    max_lag = sound_lag * 2
    lags = range(1, max_lag)
    num_lags = len(lags)

    unique = np.zeros((num_lags, 2))
    redundant = np.zeros(num_lags)
    synergistic = np.zeros(num_lags)

    for n_idx, nlag in enumerate(lags):
        _Y = np.vstack([X[0, nlag:], X[1:, :-nlag]])
        _hist, _ = np.histogramdd(_Y.T, bins=[bins_list[0], bins_list[1], bins_list[2]])
        I_R, I_S, _MI, _info_leak, *_ = surd.surd_states(_hist)
        H = it.entropy_nvars(_hist, (0,))
        unique[n_idx, 0] = I_R[(1,)] / H
        unique[n_idx, 1] = I_R[(2,)] / H
        redundant[n_idx] = I_R[(1, 2)] / H
        synergistic[n_idx] = I_S[(1, 2)] / H

    return lags, unique, redundant, synergistic, bins_list


def surd_states_at_lag(X, bins_list, nlag, verbose=True):
    """Compute the full SURD decomposition for a single time lag.

    Parameters
    ----------
    X : ndarray, shape (3, N)
    bins_list : list of ndarray
        Bin edges from lag_sweep (or _compute_bins).
    nlag : int
        Time lag in samples.
    verbose : bool
        If True, print the SURD decomposition table.

    Returns
    -------
    Rd, Sy : dict
        Redundant and synergistic information atoms.
    mi : float
        Mutual information.
    info_leak : float
    rd_states, u_states, sy_states : ndarray
        State-level decompositions.
    """
    Y = np.vstack([X[0, nlag:], X[1:, :-nlag]])
    hist, _ = np.histogramdd(Y.T, bins=[bins_list[0], bins_list[1], bins_list[2]])
    Rd, Sy, mi, info_leak, rd_states, u_states, sy_states = surd.surd_states(hist)
    if verbose:
        surd.nice_print(Rd, Sy, mi, info_leak)
    return Rd, Sy, mi, info_leak, rd_states, u_states, sy_states


def plot_surd_results(time, lags, unique, redundant, synergistic, Rd, Sy, colors=None):
    """Plot the lag sweep and the per-component bar chart.

    Parameters
    ----------
    time : ndarray
    lags : range
        Lag indices from lag_sweep.
    unique : ndarray, shape (num_lags, 2)
    redundant : ndarray, shape (num_lags,)
    synergistic : ndarray, shape (num_lags,)
    Rd, Sy : dict
        Atoms from surd_states_at_lag, used for the bar chart.
    colors : dict, optional
        Override any key in DEFAULT_COLORS.

    Returns
    -------
    fig_sweep, fig_bar : matplotlib Figure
    """
    c = {**DEFAULT_COLORS, **(colors or {})}

    fig_sweep, ax = plt.subplots()
    ax.plot(time[lags], unique[:, 0], label="Unique coherent", color=c["coherent"])
    ax.plot(time[lags], unique[:, 1], label="Unique incoherent", color=c["incoherent"])
    ax.plot(time[lags], synergistic, label="Synergistic", color=c["synergistic"])
    ax.plot(time[lags], redundant, label="Redundant", color=c["redundant"])
    ax.grid()
    ax.set_xlabel(r"$\Delta t$ (s)")
    ax.set_ylabel("Normalized Information")
    ax.set_xlim(0, time[lags.stop])
    ax.legend()
    fig_sweep.tight_layout()

    heights = np.array([Rd[(1, 2)], Rd[(1,)], Rd[(2,)], Sy[(1, 2)]])
    heights /= sum(heights)
    labels = ["Redundant", "Unique coherent", "Unique incoherent", "Synergistic"]

    fig_bar, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        labels,
        heights,
        color=[c["redundant"], c["coherent"], c["incoherent"], c["synergistic"]],
        lw=2,
        edgecolor="black",
    )
    ax.set_ylabel("Information Fraction")
    fig_bar.tight_layout()

    return fig_sweep, fig_bar


def _main():
    from wavelet_noise.utils import read_beamforming_case
    from wavelet_noise.wavelet import coherent_vortex_extraction

    plt.style.use("style.mplstyle")

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the beamforming case file.")
    parser.add_argument("--rmp", type=int, default=0, help="RMP index (0-3).")
    args = parser.parse_args()

    case = read_beamforming_case(args.file_path)
    mic = case.microphones[:, 29]
    signal = case.rmp[:, args.rmp]
    time = case.time

    cve = coherent_vortex_extraction(signal, "coif8", max_iter=100, tol=1, use_approx=False)
    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")

    X = np.stack([mic, cve.signal, cve.noise], axis=0)

    sound_speed = np.sqrt(1.4 * 287.05 * (273.15 + 22))
    delay = 1.45 / sound_speed
    sound_lag = int(delay / (time[1] - time[0]))
    print(f"Propagation delay: {delay:.4f} seconds.")

    lags, unique, redundant, synergistic, bins_list = lag_sweep(X, sound_lag)

    best_idx = np.argmax(unique[:, 0])
    nlag = lags[best_idx]
    print(f"Best time lag: {nlag} samples, {time[nlag]:.4f} seconds.")

    print("INFORMATION FLUX FOR MICROPHONE SIGNAL")
    Rd, Sy, mi, info_leak, *_ = surd_states_at_lag(X, bins_list, nlag)

    plot_surd_results(time, lags, unique, redundant, synergistic, Rd, Sy)
    plt.show()


if __name__ == "__main__":
    _main()
