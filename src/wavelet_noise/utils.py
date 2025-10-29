from dataclasses import dataclass, field

import os
import re

import h5py

import numpy as np


@dataclass(frozen=True)
class Case:
    """Class to store the experimental results. Ensures that synchronized microphones and pressure fluctuations data stay together. Also, stores metadata about the case."""

    speed: float  #: wind-speed in m/s
    aoa: float  #: angle of attack in degrees
    rmp_idx: list  #: list of the indices of the remote microphone probes (see scheme)
    microphones: np.ndarray  #: synchronized acoustics data
    rmp: (
        np.ndarray
    )  #: synchronized remote microphone probes data (wall pressure fluctuations)
    time: np.ndarray  #: time vector in seconds
    notape: bool = False  #: flag to indicate if the case is a no-tape case


def _check_file_exists(file_path: os.PathLike) -> os.PathLike:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return file_path


def rotate_airfoil(x, y, alpha_deg):
    """Rotate airfoil by :obj:`alpha_deg` degrees"""
    alpha = alpha_deg * np.pi / 180
    x_rot = x * np.cos(alpha) - y * np.sin(alpha)
    y_rot = x * np.sin(alpha) + y * np.cos(alpha)
    return x_rot, y_rot


def find_le(mesh_file):
    """Find leading edge of geometry passed as a path to the mesh file."""
    with h5py.File(mesh_file, "r") as f:
        x = f["x"]
        y = f["y"]

        # rotate airfoil
        aoa = 15.0
        x, y = rotate_airfoil(x, y, aoa)

        # separate suction and pressure sides
        idx = np.argmin(x, axis=0)[0]
    return idx


def extract_pressure_te(
    data_file: str, dx: int = 100, n_steps: int = 1500, verbose: bool = False
):
    """
    Extracts the pressure fluctuations at the trailing edge from the data file.

    Parameters
    ----------
    data_file : str
        Path to the data file.
    dx : int, optional
        Distance from the trailing edge to extract the pressure. Default is
         100.
    n_steps : int, optional
        Number of time steps to extract. Default is 1500.
    verbose : bool, optional
        If True, prints additional information. Default is False.

    Returns
    -------
    p_te : np.ndarray
        Pressure at the trailing edge.
    """
    with h5py.File(_check_file_exists(data_file), "r") as f:
        if verbose:
            print(f"+{'-' * 20}+{'-' * 20}+")
            print(f"| {'Name':<18} | {'Value':>18} |")
            print(f"+{'-' * 20}+{'-' * 20}+")
            print(f"| {'Timesteps':<18} | {f['N'][()]:>18} |")
            print(f"| {'Δt sampling':<18} | {f['T_s'][()]:>18.2e} |")
            print(f"| {'Sampling frequency':<18} | {f['f_s'][()]:18.3f} |")
            print(f"| {'Total time':<18} | {f['N'][()] * f['T_s'][()]:>18.3f} |")
            print(f"+{'-' * 20}+{'-' * 20}+")

        p_avg = f["pressure_mean"]
        p = f["pressure"]

        p_te = p[:n_steps, -dx, :] - p_avg[-dx, :]
    return p_te


def get_data_info(data_file: str, verbose: bool = False):
    """
    Prints information about the data file.

    Parameters
    ----------
    data_file : str
        Path to the data file.
    verbose : bool, optional
        If True, prints additional information. Default is False.

    Returns
    -------
    dict
        Dictionary containing the number of timesteps, sampling time, sampling
         frequency, and total time.
    """
    with h5py.File(_check_file_exists(data_file), "r") as f:
        if verbose:
            print(f"+{'-' * 20}+{'-' * 20}+")
            print(f"| {'Name':<18} | {'Value':>18} |")
            print(f"+{'-' * 20}+{'-' * 20}+")
            print(f"| {'Timesteps':<18} | {f['N'][()]:>18} |")
            print(f"| {'Δt sampling':<18} | {f['T_s'][()]:>18.2e} |")
            print(f"| {'Sampling frequency':<18} | {f['f_s'][()]:18.3f} |")
            print(f"| {'Total time':<18} | {f['N'][()] * f['T_s'][()]:>18.3f} |")
            print(f"+{'-' * 20}+{'-' * 20}+")

        dt = f["T_s"][()]  # adimensional time step
        return {
            "N": f["N"][()],  # number of time steps
            "T_s": dt,  # adimensional time step
            "f_s": 1 / dt,  # sampling frequency
            "total_time": f["N"][()] * dt,  # total time
        }


RMP_CONVERT = {
    "2-6": [2, 3, 5, 6],
    "4-12": [4, 8, 10, 12],
    "22-26": [22, 23, 24, 26],
    "25-29": [25, 26, 27, 29],
}

VELOCITY_FACTOR = 1.14  # factor to convert from percentage to m/s


def parse_beamforming_name(file_name):
    """
    Parse a beamforming data filename and extract its parameters.

    Parameters
    ----------
    file_name : str
        Filename in format: CD-ISAE-{n}Xdeg-Ypr-rmpA-B-{90s}-{notape}-1.h5

    Returns
    -------
    dict
        Dictionary containing:
        - 'wind_speed': float, wind speed in m/s
        - 'angle_of_attack': float, angle of attack in degrees
        - 'rmp_numbers': list of int, list of RMP indices
        - 'notape': bool, whether gap was not sealed

    Examples
    --------
    >>> _parse_beamforming_name("CD-ISAE-5deg-50pr-rmp1-30-90s-notape-1.h5")
    {'wind_speed': 57.0, 'angle_of_attack': 5.0, 'rmp_numbers': [1, ..., 30], 'notape': True}

    >>> _parse_beamforming_name("CD-ISAE-n2deg-75pr-rmp10-50-1.h5")
    {'wind_speed': 85.5, 'angle_of_attack': -2.0, 'rmp_numbers': [10, ..., 50], 'notape': False}
    """
    # pattern = r'CD-ISAE-(n)?(\d+)deg-(\d+)pr-rmp(\d+)-(\d+)'
    pattern = r"CD-ISAE-(n)?(\d+)deg-(\d+)pr-rmp(\d+-\d+)"

    match = re.search(pattern, file_name)
    if not match:
        raise ValueError(f"Filename '{file_name}' does not match expected format")

    is_negative, angle, wind_percent, rmp_code = match.groups()

    angle_of_attack = float(angle)
    if is_negative:
        angle_of_attack = -angle_of_attack

    # Convert wind speed: percentage to m/s (multiply by 1.14)
    wind_speed = float(wind_percent) * VELOCITY_FACTOR

    rmp_numbers = RMP_CONVERT.get(rmp_code.strip())
    if rmp_numbers is None:
        raise ValueError(
            f"RMP code '{rmp_code}' not recognized in filename '{file_name}'"
        )

    notape = "notape" in file_name

    return {
        "wind_speed": wind_speed,
        "angle_of_attack": angle_of_attack,
        "rmp_numbers": rmp_numbers,
        "notape": notape,
    }


def read_beamforming_case(file_path: os.PathLike) -> Case:
    """
    Reads a beamforming case from an HDF5 file and returns a Case object.
    """
    file_path = _check_file_exists(file_path)
    metadata = parse_beamforming_name(os.path.basename(file_path))

    with h5py.File(file_path, "r") as f:
        group = f["Table1"]
        dataset_names = list(group.keys())
        time = group[dataset_names[0]][:]
        microphones = [
            group[data_name][:]
            for data_name in dataset_names[1 : -len(metadata["rmp_numbers"])]
        ]
        rmp = [
            group[data_name][:]
            for data_name in dataset_names[-len(metadata["rmp_numbers"]) :]
        ]

    return Case(
        speed=metadata["wind_speed"],
        aoa=metadata["angle_of_attack"],
        rmp_idx=metadata["rmp_numbers"],
        microphones=np.array(microphones).T,
        rmp=np.array(rmp).T,
        time=np.array(time),
        notape=metadata["notape"],
    )
