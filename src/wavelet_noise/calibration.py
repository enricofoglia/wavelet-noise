from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
import os


@dataclass
class CalibrationResult:
    """
    Dataclass to store the results of the RMP signal calibration process.

    Attributes:
    frequencies (np.ndarray): The frequencies at which the transfer function and coherence were computed.
    transfer_function (np.ndarray): The estimated transfer function between the reference and RMP signals.
    coherence (np.ndarray): The coherence between the reference and RMP signals.
    delay (float): The estimated propagation time (delay) between the reference and RMP signals.
    group_delay (np.ndarray): The group delay computed from the phase of the transfer function.
    calibrated_signal (np.ndarray): The calibrated RMP signal in the time domain.
    """

    frequencies: np.ndarray
    transfer_function: np.ndarray
    coherence: np.ndarray
    delay: float
    group_delay: np.ndarray
    calibrated_signal: np.ndarray


def load_calibration_data(
    file_path: str, file_name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load calibration data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the calibration data.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded calibration data.
    """
    try:
        data = pd.read_csv(
            os.path.join(file_path, file_name), delimiter="\t", header=None
        )
        return np.asarray(data[0]), np.asarray(data[1] // 0.0215)
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None


def load_experimental_data(
    file_path: str, file_name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load experimental data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the experimental data.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded experimental data.
    """
    try:
        data = pd.read_csv(
            os.path.join(file_path, file_name), delimiter="\t", header=None
        )
        return np.asarray(data[0]), np.asarray(data[1])
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None


def calibrate_rmp_signal(
    p_ref: np.ndarray,
    p_rmp: np.ndarray,
    p_target: np.ndarray,
    time: np.ndarray,
    nperseg: int = 1024,
) -> CalibrationResult:
    """
    Calibrate the RMP signal using the reference pressure, RMP pressure, and target pressure.

    Parameters:
    p_ref (np.array): The reference pressure signal.
    p_rmp (np.array): The RMP pressure signal.
    p_target (np.array): The pressure signal to calibrate.
    time (np.array): The time signal.

    Returns:
    np.array: The calibrated RMP signal.
    """

    fs = 1 / (time[1] - time[0])  # Sampling frequency

    # Compute transfer function and coherence
    f, Gxx = signal.welch(p_ref, fs=fs, nperseg=nperseg)
    _, Gyy = signal.welch(p_rmp, fs=fs, nperseg=nperseg)
    _, Gxy = signal.csd(p_ref, p_rmp, fs=fs, nperseg=nperseg)

    H = Gxy / Gxx  # Transfer function
    Cxy = np.abs(Gxy) ** 2 / (Gxx * Gyy)  # Coherence

    # Estimate propagation time (delay) using the phase of the transfer function
    phase = np.unwrap(np.angle(H))
    slope, _ = np.polyfit(f[1 : nperseg // 4], phase[1 : nperseg // 4], 1)
    delay = -slope / (2 * np.pi)
    group_delay = -np.diff(phase) / np.diff(2 * np.pi * f)

    # Signal correction using the estimated transfer function and delay
    N = len(p_target)
    freq_target = np.fft.rfftfreq(N, d=1 / fs)
    p_target_fft = np.fft.rfft(p_target)

    # Interpolate H to match the frequencies of the target signal
    H_interp = np.interp(freq_target, f, H)

    # Apply the transfer function and delay correction
    epsilon = 1e-3 * np.max(
        np.abs(H_interp)
    )  # Regularization term to avoid division by zero
    H_inv = np.conj(H_interp) / (
        np.abs(H_interp) ** 2 + epsilon
    )  # Inverse of the transfer function with regularization

    p_calibrated_fft = p_target_fft * H_inv
    p_calibrated_t = np.fft.irfft(p_calibrated_fft, n=N)

    return CalibrationResult(f, H, Cxy, delay, group_delay, p_calibrated_t)
