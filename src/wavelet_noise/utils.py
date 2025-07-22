import h5py

import numpy as np


def _print_name(name):
    print(name)


def _print_info(name, object):
    print(name)
    print(type(object))
    try:
        print(f'... of shape {object.shape}')
    except AttributeError:
        pass


def rotate_airfoil(x, y, alpha_deg):
    '''Rotate airfoil by :obj:`alpha_deg` degrees'''
    alpha = alpha_deg * np.pi / 180
    x_rot = x * np.cos(alpha) - y * np.sin(alpha)
    y_rot = x * np.sin(alpha) + y * np.cos(alpha)
    return x_rot, y_rot


def find_le(mesh_file):
    '''Find leading edge of geometry passed as a path to the mesh file.'''
    with h5py.File(mesh_file, "r") as f:
        x = f['x']
        y = f['y']

        # rotate airfoil
        aoa = 15.0
        x, y = rotate_airfoil(x, y, aoa)

        # separate suction and pressure sides
        idx = np.argmin(x, axis=0)[0]
    return idx


def extract_pressure_te(
        data_file: str,
        dx: int = 100,
        n_steps: int = 1500,
        verbose: bool = False
):
    '''
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
    '''
    with h5py.File(data_file, "r") as f:
        if verbose:
            print(f'+{"-"*20}+{"-"*20}+')
            print(f'| {"Name":<18} | {"Value":>18} |')
            print(f'+{"-"*20}+{"-"*20}+')
            print(f'| {"Timesteps":<18} | {f["N"][()]:>18} |')
            print(f'| {"Δt sampling":<18} | {f["T_s"][()]:>18.2e} |')
            print(f'| {"Sampling frequency":<18} | {f["f_s"][()]:18.3f} |')
            print(
                f'| {"Total time":<18} | {f["N"][()] * f["T_s"][()]:>18.3f} |')
            print(f'+{"-"*20}+{"-"*20}+')

        p_avg = f['pressure_mean']
        p = f['pressure']

        p_te = p[:n_steps, -dx, :] - p_avg[-dx, :]
    return p_te


def get_data_info(data_file: str, verbose: bool = False):
    '''
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
    '''
    with h5py.File(data_file, "r") as f:
        if verbose:
            print(f'+{"-"*20}+{"-"*20}+')
            print(f'| {"Name":<18} | {"Value":>18} |')
            print(f'+{"-"*20}+{"-"*20}+')
            print(f'| {"Timesteps":<18} | {f["N"][()]:>18} |')
            print(f'| {"Δt sampling":<18} | {f["T_s"][()]:>18.2e} |')
            print(f'| {"Sampling frequency":<18} | {f["f_s"][()]:18.3f} |')
            print(
                f'| {"Total time":<18} | {f["N"][()] * f["T_s"][()]:>18.3f} |')
            print(f'+{"-"*20}+{"-"*20}+')

        dt = f['T_s'][()]  # adimensional time step
        return {
            'N': f['N'][()],  # number of time steps
            'T_s': dt,  # adimensional time step
            'f_s': 1 / dt,  # sampling frequency
            'total_time': f['N'][()] * dt  # total time
        }
