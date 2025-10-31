import os

import yaml

from functools import partial

import numpy as np

import scipy.signal as sg
import scipy.stats as stats
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import wavelet_noise as wn

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


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
    if config["conditioning"]["bandpass_filter"]["apply"]:
        bandpass_filter = partial(
            wn.stats.butter_bandpass_filter,
            lowcut=config["conditioning"]["bandpass_filter"]["lowcut"],
            highcut=config["conditioning"]["bandpass_filter"]["highcut"],
            fs=data.fs[0],
            order=2,
            form="ba",
        )
    else:
        bandpass_filter = None

    conditioning = partial(
        wn.stats.conditioning,
        standardize=config["conditioning"]["standardize"],
        detrend=config["conditioning"]["detrend"],
        detrend_degree=config["conditioning"]["detrend_degree"],
        filter=bandpass_filter,
    )

    signal = conditioning(signal)
    wn.stats.display_diagnostics(signal, dt=1.0 / data.fs[0], corr_threshold=0.0)

    # display autocorrelation of the rmp signal
    n = signal.shape[0]
    nperseg = n // 32
    autocorr, autocorr_std = wn.stats.windowed_autocorrelation(
        signal, nperseg=nperseg, noverlap=nperseg // 2, return_std=True
    )
    time_lags = sg.correlation_lags(nperseg, nperseg, mode="full") / data.fs
    fig, ax = plt.subplots()
    ax.plot(time_lags[nperseg - 1 :], autocorr)
    ax.fill_between(
        time_lags[nperseg - 1 :],
        autocorr - autocorr_std,
        autocorr + autocorr_std,
        alpha=0.5,
    )
    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel("Autocorrelation [-]")
    ax.set_title("Autocorrelation of RMP signal")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(0.0, 0.025)
    plt.savefig(os.path.join(config["out_dir"], "autocorrelation_rmp.png"))
    plt.close()

    threshold_corr = np.linspace(0, 1, 25, endpoint=False)
    integral_time_scale = []
    t_lag = []
    for thresh in threshold_corr:
        idx = np.nonzero(autocorr < thresh)[0][0] - 1
        if idx < 0:
            idx = 0
        integral_time_scale.append(integrate.simpson(autocorr[:idx], dx=1.0 / data.fs))
        t_lag.append(time_lags[n // 8 - 1 + idx])
    fig, ax = plt.subplots()
    ax.plot(threshold_corr, integral_time_scale, "-o")

    ax.set_xlabel("Autocorrelation threshold [-]")
    ax.set_ylabel("Integral time scale $\Lambda_t$ [s]")
    ax.set_title("Integral time scale vs autocorrelation threshold")
    ax.xaxis.set_inverted(True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))  # Force 10^-3 notation
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(
        os.path.join(config["out_dir"], "itc_vs_threshold.png"), bbox_inches="tight"
    )
    plt.close("all")

    welch_kwargs = {
        "fs": data.fs,
        "nperseg": signal.shape[0] // 2**6,
        "noverlap": signal.shape[0] // 2**7,
        "window": "hamming",
    }

    f, psd = sg.welch(signal, **welch_kwargs)

    cve = wn.wavelet.coherent_vortex_extraction(
        signal, wavelet="coif8", max_iter=100, tol=1, use_approx=False
    )

    print(f"Number of iterations: {cve.iterations}")
    print(f"Final threshold: {cve.final_threshold:.4e}")
    print(f"Number of coherent coefficients: {cve.num_coherent_coeffs}")
    print(f"Number of incoherent coefficients: {cve.num_incoherent_coeffs}")
    print(
        f"Fraction of coherent coefficients: {cve.num_coherent_coeffs / (cve.num_coherent_coeffs + cve.num_incoherent_coeffs):.2%}"
    )

    psd_noise = sg.welch(cve.noise, **welch_kwargs)[1]

    psd_hydro = sg.welch(cve.signal, **welch_kwargs)[1]

    signal_micro = conditioning(data.microphones[:, 29])

    f, coherence_hydro = sg.coherence(signal_micro, cve.signal, **welch_kwargs)
    _, coherence_noise = sg.coherence(signal_micro, cve.noise, **welch_kwargs)
    _, coherence_signal = sg.coherence(signal_micro, cve.signal, **welch_kwargs)

    correlation_hydro = (
        sg.correlate(signal_micro, cve.signal, mode="full") / len(signal_micro) ** 2
    )
    correlation_signal = (
        sg.correlate(signal_micro, signal, mode="full") / len(signal_micro) ** 2
    )
    correlation_noise = (
        sg.correlate(signal_micro, cve.noise, mode="full") / len(signal_micro) ** 2
    )

    time_lags = (
        sg.correlation_lags(len(signal_micro), len(cve.signal), mode="full") / data.fs
    )
    c0 = config["sound_speed"]
    L = config["microphone_distance"]
    error_L = 0.05  # 1 cm error in distance measurement
    p_ref = config.get("p_ref", 20e-6)

    fig, ax = plt.subplots()
    ax.fill_betweenx(
        [0, max(np.abs(correlation_hydro))],
        (L + error_L) / c0,
        (L - error_L) / c0,
        color="tomato",
        alpha=0.3,
        label=r"Error on $t^*=L/c_0$",
    )
    ax.axvline(L / c0, color="tomato", ls="--", label=r"$t^*=L/c_0$")
    for i in range(2, 11):
        ax.axvline(i * L / c0, color="tomato", ls="--")
    ax.plot(time_lags, np.abs(correlation_hydro) / p_ref**2)

    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel(r"Cross-correlation / $p_{ref}^2$ [-]")
    ax.set_title("Cross-correlation between microphone and hydrodynamic component")
    ax.grid(True, which="both", ls="--", lw=0.5)
    # ax.set_xscale("log")
    ax.set_xlim(-0.025, 0.025)

    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "correlation_hydro.png"))

    fig, ax = plt.subplots()
    ax.plot(time_lags, np.abs(correlation_signal) / p_ref**2, label="Original signal")
    ax.plot(
        time_lags,
        np.abs(correlation_hydro) / p_ref**2,
        "--",
        label="Hydrodynamic component",
    )
    ax.plot(
        time_lags,
        np.abs(correlation_noise) / p_ref**2,
        label="Noise component",
        linewidth=3,
    )
    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel(r"Cross-correlation / $p_{ref}^2$ [-]")
    ax.set_title("Cross-correlation between microphone and hydrodynamic component")
    ax.grid(True, which="both", ls="--", lw=0.5)
    # ax.set_xscale("log")
    ax.set_xlim(-0.025, 0.025)

    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "correlation_comparison.png"))
    plt.close("all")

    correlation = []
    with progress_bar as pb:
        for micro in pb.track(data.microphones.T):
            correlation.append(
                np.abs(sg.correlate(micro, cve.signal, mode="full"))
                / len(signal_micro) ** 2
            )
    correlation = np.array(correlation)
    avg_correlation = np.mean(correlation, axis=0)
    std_correlation = np.std(correlation, axis=0)
    fig, ax = plt.subplots()

    ax.plot(time_lags, avg_correlation / p_ref**2, label="Average cross-correlation")

    ax.fill_between(
        time_lags,
        (avg_correlation - 1 * std_correlation) / p_ref**2,
        (avg_correlation + 1 * std_correlation) / p_ref**2,
        alpha=0.5,
        label=r"$\pm \sigma$",
    )
    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel(r"Cross-correlation / $p_{ref}^2$ [-]")
    ax.set_title(
        "Average cross-correlation between microphones and hydrodynamic component"
    )
    ax.grid(True, which="both", ls="--", lw=0.5)
    # ax.set_xscale("log")
    ax.set_xlim(-0.025, 0.025)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    plt.savefig(
        os.path.join(config["out_dir"], "correlation_hydro_avg.png"),
        bbox_inches="tight",
    )

    fig, ax = plt.subplots()
    ax.plot(cve.incoherent_coeffs_history, "-o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of incoherent coefficients")
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(config["out_dir"], "cve_convergence.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd / p_ref), label="Original signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_original.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd_hydro / p_ref), label="Original signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_denoised.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd_noise / p_ref), label="Original signal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_noise.png"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd / p_ref), label="Original signal")
    ax.semilogx(f, 10 * np.log10(psd_hydro / p_ref), label="Hydrodynamic component")
    ax.semilogx(f, 10 * np.log10(psd_noise / p_ref), label="Noise component")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "psd_comparison.png"))
    plt.close("all")

    nsamples = 1000
    fig, ax = plt.subplots()
    ax.plot(data.time[:nsamples], signal[:nsamples], label="Original signal")
    ax.plot(
        data.time[:nsamples],
        cve.signal[:nsamples],
        "--",
        label="Hydrodynamic component",
    )
    ax.plot(data.time[:nsamples], cve.noise[:nsamples], label="Noise component")
    # ax.plot(
    #     data.time[:nsamples],
    #     cve.signal[:nsamples] + cve.noise[:nsamples],
    #     "--",
    #     label="Reconstructed signal",
    # ) # useful for debugging
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure fluctuation [Pa]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "time_signal_comparison.png"))

    fig, ax = plt.subplots()
    ax.hist(
        signal / signal.std(),
        bins=100,
        alpha=1.0,
        label="Original signal",
        density=True,
        histtype="stepfilled",
        zorder=1,
    )
    ax.hist(
        cve.signal / cve.signal.std(),
        bins=100,
        alpha=1.0,
        label="Hydrodynamic component",
        density=True,
        histtype="step",
        linewidth=2.0,
        zorder=2,
    )
    ax.hist(
        cve.noise / cve.noise.std(),
        bins=100,
        alpha=1.0,
        label="Noise component",
        density=True,
        zorder=3,
    )
    ax.plot(
        np.linspace(-10.0, 10.0, 250),
        stats.logistic.pdf(np.linspace(-10.0, 10.0, 250), scale=np.sqrt(3) / np.pi),
        "--",
        color="tomato",
        label="Standard Logistic",
        zorder=4,
    )
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

    fig, ax = plt.subplots()
    ax.semilogx(f, coherence_signal, label="Original signal")
    ax.semilogx(f, coherence_hydro, "--", label="Hydrodynamic component")
    ax.semilogx(f, coherence_noise, label="Noise component")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence with microphone signal")
    ax.set_xlim(20, 20e3)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "coherence_comparison.png"))
    plt.close("all")


if __name__ == "__main__":
    main()
