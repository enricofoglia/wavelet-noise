import argparse

import numpy as np
import matplotlib.pyplot as plt

from wavelet_noise.utils import read_beamforming_case
from wavelet_noise.wavelet import coherent_vortex_extraction

from surd_states import surd 
from surd_states import it_tools as it


parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, help="Path to the beamforming case file.")

def _main():
    args = parser.parse_args()
    case = read_beamforming_case(args.file_path)
    mic = case.microphones[:, 29]
    signal = case.rmp[:, -1]
    
    # CVE
    cve = coherent_vortex_extraction(signal, "coif8", max_iter=100, tol=1, use_approx=False)

    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")

    hydro, noise = cve.signal, cve.noise

    X = np.stack([mic, hydro, noise], axis=0)
    nvars = X.shape[0]

    # compute bins for histogram
    nbins = 25
    max_abs = np.percentile(X, 99.99)
    max_abs = np.floor(max_abs)
    bin_width = 2 * max_abs / (nbins - 1)
    bins_list = []
    for _i in range(X.shape[0]):
        bins = np.linspace(-max_abs, max_abs + bin_width, nbins + 1)
        bins_list.append(bins)

    # apply time delay to microphone signal
    delay = 1.45 / 343.0
    nlag = np.argmin(np.abs(case.time - delay))
    print(f"Applied time delay of {delay:.4f} seconds, corresponding to {nlag} samples at fs={case.fs[0]:.2f} Hz.")

    Rd_results, Sy_results, mi_results, info_leak_results = ({}, {}, {}, {})
    rd_states_results, u_states_results, sy_states_results = ({}, {}, {})
    _fig, axs = plt.subplots(nvars, 2, figsize=(10, 2.6 * nvars), gridspec_kw={'width_ratios': [nvars * 20, 1]})
    for _i in range(nvars):
        print(f'INFORMATION FLUX FOR SIGNAL {_i + 1}')
        Y = np.vstack([X[_i, nlag:], X[:, :-nlag]])
        hist, bins_1 = np.histogramdd(Y.T, bins=[bins_list[_i], bins_list[0], bins_list[1], bins_list[2]])
        Rd, Sy, mi, info_leak, rd_states, u_states, sy_states = surd.surd_states(hist)
        surd.nice_print(Rd, Sy, mi, info_leak)
        _ = surd.plot(Rd, Sy, info_leak, axs[_i, :], nvars, threshold=-0.01)
        axs[_i, 0].set_title(f'${{\\Delta I}}_{{(\\cdot) \\rightarrow {_i + 1}}} / I \\left(Q_{_i + 1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$', pad=12)
        axs[_i, 1].set_title(f'$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {_i + 1}}}}}{{H \\left(Q_{_i + 1}^+ \\right)}}$', pad=20)
        axs[_i, 0].set_xticklabels(axs[_i, 0].get_xticklabels(), fontsize=18, rotation=60, ha='right', rotation_mode='anchor')
        axs[_i, 0].set_yticks([0, 0.5])
        axs[_i, 0].set_ylim([0, 0.5])
        Rd_results[_i + 1] = Rd
        Sy_results[_i + 1] = Sy
        mi_results[_i + 1] = mi
        info_leak_results[_i + 1] = info_leak
        rd_states_results[_i + 1] = rd_states
        u_states_results[_i + 1] = u_states
        sy_states_results[_i + 1] = sy_states
    plt.tight_layout(w_pad=-12, h_pad=1)
    plt.show()


if __name__ == "__main__":
    _main()