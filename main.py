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

from rich import print
from rich.progress import track

from cycler import cycler

from pypalettes import load_cmap
# cmap = load_cmap("CafeTerrace")
# cmap = load_cmap("Dark")
# cmap = load_cmap("Antique")
# cmap = load_cmap("Lively")
# cmap = load_cmap("Tableau_10")
cmap = load_cmap("alger", shuffle=2)
main_color = cmap(4)

colors = [cmap(i) for i in range(cmap.N)]
def_colors = cycler(color=colors)

plt.style.use("style.mplstyle")
# plt.style.use("dark_background")
plt.rcParams.update(
    {
        # "lines.linewidth": 2,
        # "savefig.dpi": 300,
        "axes.prop_cycle": def_colors
    }
)


def perform_analysis(data: wn.utils.Case, config: dict):
    signal = data.rmp[:, config["rmp_index"]]
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
    # wn.stats.display_diagnostics(signal, dt=1.0 / data.fs[0], corr_threshold=0.0)

    # display autocorrelation of the rmp signal
    n = signal.shape[0]
    autocorr = sg.correlate(signal, signal, mode="full")[n - 1 :] / n / signal.var()
    time_lags = sg.correlation_lags(n, n, mode="full")[n - 1 :] / data.fs
    fig, ax = plt.subplots()
    ax.plot(time_lags, autocorr, color=main_color)
    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel("Autocorrelation [-]")
    ax.set_title("Autocorrelation of RMP signal")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(0.0, 0.025)
    plt.savefig(os.path.join(config["out_dir"], "autocorrelation_rmp.svg"))
    plt.close()

    threshold_corr = np.linspace(0, 1, 25, endpoint=False)
    integral_time_scale = []
    t_lag = []
    for thresh in threshold_corr:
        idx = np.nonzero(autocorr < thresh)[0][0]
        if idx < 1:
            idx = 1
        integral_time_scale.append(
            integrate.trapezoid(autocorr[:idx], dx=1.0 / data.fs)
        )
        t_lag.append(time_lags[1 + idx])
    fig, ax = plt.subplots()
    ax.plot(threshold_corr, integral_time_scale, "-o", color=main_color)

    ax.set_xlabel("Autocorrelation threshold [-]")
    ax.set_ylabel("Integral time scale $\Lambda_t$ [s]")
    ax.set_title("Integral time scale vs autocorrelation threshold")
    ax.xaxis.set_inverted(True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))  # Force 10^-3 notation
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(
        os.path.join(config["out_dir"], "itc_vs_threshold.svg"), bbox_inches="tight"
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
        signal, wavelet=config["wavelet"], max_iter=100, tol=1, use_approx=False
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

    signal_micro = conditioning(data.microphones[:, config["micro_index"]])

    f, coherence_hydro = sg.coherence(signal_micro, cve.signal, **welch_kwargs)
    _, coherence_noise = sg.coherence(signal_micro, cve.noise, **welch_kwargs)
    _, coherence_signal = sg.coherence(signal_micro, signal, **welch_kwargs)

    psd_micro = sg.welch(signal_micro, **welch_kwargs)[1]

    correlation_hydro = (
        sg.correlate(signal_micro, cve.signal, mode="full")
        / len(signal_micro)
        / signal_micro.std()
        / cve.signal.std()
    )
    correlation_signal = (
        sg.correlate(signal_micro, signal, mode="full")
        / len(signal_micro)
        / signal_micro.std()
        / signal.std()
    )
    correlation_noise = (
        sg.correlate(signal_micro, cve.noise, mode="full")
        / len(signal_micro)
        / signal_micro.std()
        / cve.noise.std()
    )

    time_lags = (
        sg.correlation_lags(len(signal_micro), len(cve.signal), mode="full") / data.fs
    )
    c0 = config["sound_speed"]
    L = config["microphone_distance"]
    error_L = 0.05  # 1 cm error in distance measurement
    p_ref = config.get("p_ref", 20e-6)

    # print
    print(f"Theoretical time lag: {L / c0:.2e} s")
    peak_lag = time_lags[np.argmax(correlation_hydro)]
    print(f"Peak time lag: {peak_lag:.2e} s")
    print(f"Relative error: {np.abs(peak_lag - L / c0) * c0 / L:.1%}")
    fig, ax = plt.subplots()

    ax.axvline(L / c0, color="tomato", ls="--", label=r"$t^*=L/c_0$")
    ax.plot(time_lags, correlation_hydro, color=main_color)
    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel(r"Cross-correlation [-]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    # ax.set_xscale("log")
    ax.set_xlim(-0.015, 0.025)

    # ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "correlation_hydro.svg"))

    fig, ax = plt.subplots()
    ax.plot(time_lags, correlation_signal, label="Original signal")
    ax.plot(
        time_lags,
        correlation_hydro,
        "--",
        label="Coherent component",
    )
    ax.plot(
        time_lags,
        correlation_noise,
        label="Incoherent component",
        linewidth=3,
    )
    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel(r"Cross-correlation [-]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    # ax.set_xscale("log")
    ax.set_xlim(-0.025, 0.025)

    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "correlation_comparison.svg"))
    plt.close("all")

    correlation = []
    for micro in track(
        data.microphones.T[:60], transient=True, description="Microphones"
    ):
        correlation.append(
            np.abs(sg.correlate(micro, cve.signal, mode="full"))
            / len(signal_micro)
            / micro.std()
            / cve.signal.std()
        )
    correlation = np.array(correlation)
    avg_correlation = np.mean(correlation, axis=0)
    std_correlation = np.std(correlation, axis=0)
    fig, ax = plt.subplots()

    ax.plot(time_lags, avg_correlation, color=main_color,label="Average cross-correlation")

    ax.fill_between(
        time_lags,
        (avg_correlation - 1 * std_correlation),
        (avg_correlation + 1 * std_correlation),
        alpha=0.5,
        label=r"$\pm \sigma$",
    )
    ax.axvline(L / c0, color="tomato", ls="--", label=r"$t^*=L/c_0$")

    ax.set_xlabel("Time lag [s]")
    ax.set_ylabel(r"Cross-correlation [-]")

    ax.grid(True, which="both", ls="--", lw=0.5)
    # ax.set_xscale("log")
    ax.set_xlim(-0.025, 0.025)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    plt.savefig(
        os.path.join(config["out_dir"], "correlation_hydro_avg.svg"),
        bbox_inches="tight",
    )

    fig, ax = plt.subplots()
    ax.plot(np.array(cve.incoherent_coeffs_history) / n, "-o", color=main_color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of incoherent coefficients")
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(config["out_dir"], "cve_convergence.svg"))

    lowcut = config["conditioning"]["bandpass_filter"]["lowcut"]
    highcut = config["conditioning"]["bandpass_filter"]["highcut"]

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd / p_ref**2), label="Original signal")
    ax.axvspan(20, lowcut, color="0.9")
    ax.axvspan(highcut, 20e3, color="0.9")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-60, 80)
    plt.savefig(os.path.join(config["out_dir"], "psd_original.svg"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd_hydro / p_ref**2), label="Original signal")
    ax.axvspan(20, lowcut, color="0.9")
    ax.axvspan(highcut, 20e3, color="0.9")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_denoised.svg"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd_noise / p_ref**2), label="Original signal")
    ax.axvspan(20, lowcut, color="0.9")
    ax.axvspan(highcut, 20e3, color="0.9")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    plt.savefig(os.path.join(config["out_dir"], "psd_noise.svg"))

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd / p_ref**2), label="Original signal")
    ax.semilogx(
        f, 10 * np.log10(psd_hydro / p_ref**2), "--", label="Coherent component"
    )
    ax.semilogx(f, 10 * np.log10(psd_noise / p_ref**2), label="Incoherent component")
    ax.axvspan(20, lowcut, color="0.9")
    ax.axvspan(highcut, 20e3, color="0.9")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-60, 80)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "psd_comparison.svg"))
    plt.close("all")

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd_micro / p_ref**2), label="Microphone signal")
    ax.axvspan(20, lowcut, color="0.9")
    ax.axvspan(highcut, 20e3, color="0.9")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-60, 40)
    plt.savefig(
        os.path.join(config["out_dir"], f"psd_farfield_{config['micro_index'] + 1}.svg")
    )

    nsamples = 1000
    fig, ax = plt.subplots()
    ax.plot(data.time[:nsamples], signal[:nsamples], label="Original signal")
    ax.plot(
        data.time[:nsamples],
        cve.signal[:nsamples],
        "--",
        label="Coherent component",
    )
    ax.plot(data.time[:nsamples], cve.noise[:nsamples], label="Incoherent component")
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
    plt.savefig(os.path.join(config["out_dir"], "time_signal_comparison.svg"))

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
        label="Coherent component",
        density=True,
        histtype="step",
        linewidth=2.0,
        zorder=2,
    )
    ax.hist(
        cve.noise / cve.noise.std(),
        bins=100,
        alpha=1.0,
        label="Incoherent component",
        histtype="step",
        linewidth=2.0,
        density=True,
        zorder=3,
    )
    ax.plot(
        np.linspace(-10.0, 10.0, 250),
        stats.logistic.pdf(np.linspace(-10.0, 10.0, 250), scale=np.sqrt(3) / np.pi),
        "--",
        label="Standard Logistic",
        zorder=4,
    )
    ax.plot(
        np.linspace(-10.0, 10.0, 250),
        stats.norm.pdf(np.linspace(-10.0, 10.0, 250)),
        "-.",
        label="Standard Gaussian",
        zorder=5,
    )
    ax.set_xlabel(r"Pressure fluctuation / $\sigma$ [-]")
    ax.set_ylabel("Probability density")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6)
    ax.grid(True, which="both", ls="--", lw=0.5, zorder=0)
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(config["out_dir"], "pdf_comparison.svg"))

    fig, ax = plt.subplots()
    ax.hist(cve.noise, bins=100, alpha=1.0, label="Incoherent component", density=True)
    ax.set_xlabel("Pressure fluctuation [Pa]")
    ax.set_ylabel("Probability density")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(config["out_dir"], "pdf_noise.svg"))

    fig, ax = plt.subplots()
    ax.semilogx(f, coherence_signal, label="Original signal")
    ax.semilogx(f, coherence_hydro, "--", label="Coherent component")
    ax.semilogx(f, coherence_noise, label="Incoherent component")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence with microphone signal")
    ax.set_xlim(20, 20e3)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "coherence_comparison.svg"))
    plt.close("all")

    weiner_analysis(signal, signal_micro, config, data.fs)


def main():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if config["compute_all"]:
        cases = wn.utils.list_beamforming_cases(config["data_dir"])

        for case in track(cases, description="Analysing cases"):
            try:
                data = wn.utils.read_beamforming_case(case)
                for rmp in range(4):
                    config["rmp_index"] = rmp
                    config["out_dir"] = wn.utils.create_out_directory(
                        config["out_dir_root"], case, data.rmp_idx[rmp]
                    )
                    print(f"Output directory : {config['out_dir']}")
                    perform_analysis(data, config)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                continue

    else:
        data = wn.utils.read_beamforming_case(
            os.path.join(config["data_dir"], config["case_name"])
        )
        config["out_dir"] = wn.utils.create_out_directory(
            config["out_dir_root"],
            os.path.join(config["data_dir"], config["case_name"]),
            data.rmp_idx[config["rmp_index"]],
        )

        print(f"[bold]Output directory[/bold] : {config['out_dir']}")
        wn.stats.display_diagnostics(
            data.rmp[:, config["rmp_index"]], dt=1.0 / data.fs[0], corr_threshold=0.0
        )
        perform_analysis(data, config)

def weiner_analysis(signal, micro, config, fs=1.0):
    hydro = sg.wiener(signal)
    noise = signal - hydro

    welch_kwargs = {
        "fs": fs,
        "nperseg": signal.shape[0] // 2**6,
        "noverlap": signal.shape[0] // 2**7,
        "window": "hamming",
    }

    f, psd = sg.welch(signal, **welch_kwargs)
    psd_hydro = sg.welch(hydro, **welch_kwargs)[1]
    psd_noise = sg.welch(noise, **welch_kwargs)[1]

    coherence_hydro = sg.coherence(micro, hydro, **welch_kwargs)[1]
    coherence_noise = sg.coherence(micro, noise, **welch_kwargs)[1]
    coherence_signal = sg.coherence(micro, signal, **welch_kwargs)[1]


    p_ref = config["p_ref"]
    lowcut = config["conditioning"]["bandpass_filter"]["lowcut"]
    highcut = config["conditioning"]["bandpass_filter"]["highcut"]

    fig, ax = plt.subplots()
    ax.semilogx(f, 10 * np.log10(psd / p_ref**2), label="Original signal")
    ax.semilogx(
        f, 10 * np.log10(psd_hydro / p_ref**2), "--", label="Coherent component"
    )
    ax.semilogx(f, 10 * np.log10(psd_noise / p_ref**2), label="Incoherent component")
    ax.axvspan(20, lowcut, color="0.9")
    ax.axvspan(highcut, 20e3, color="0.9")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density [dB/Hz]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-60, 80)
    ax.set_facecolor("0.9")
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(config["out_dir"], "wiener_psd_comparison.svg"))
    
    fig, ax = plt.subplots()
    ax.semilogx(f, coherence_signal, label="Original signal")
    ax.semilogx(f, coherence_hydro, "--", label="Coherent component")
    ax.semilogx(f, coherence_noise, label="Incoherent component")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence with microphone signal")
    ax.set_xlim(20, 20e3)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="upper right")
    ax.set_facecolor("0.9")
    plt.savefig(os.path.join(config["out_dir"], "wiener_coherence_comparison.svg"))
    plt.close("all")





if __name__ == "__main__":
    main()
