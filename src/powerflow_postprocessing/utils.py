import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import os

plt.rcParams.update({
    "text.usetex": True,           # Usa LaTeX real (requiere instalación)
    "font.family": "serif",        # Usa fuentes serif (estilo LaTeX)
    "font.serif": ["Computer Modern"], 
    "axes.labelsize": 12,          # Tamaño de etiquetas de ejes
    "xtick.labelsize": 10,         # Tamaño de los números en X
    "ytick.labelsize": 10          # Tamaño de los números en Y
})

def next_greater_power_of_2(x):
    return round(2**(x-1).bit_length())

def rmp_timetrace(rmp_location:int, data_dir:str, *file_names):
    
    '''
    This function generates the time trace of the RMP signal at a given location.\n 
    It takes as input the location of the RMP signal, the directory where the data is stored,\n 
    and the names of the files to be read. It returns the time trace of the RMP signal at the given location.\n
    \n
    Parameters
    ----------
    rmp_location : int
        The location of the RMP signal.
    data_dir : str
        The directory where the data is stored.
    file_names : list
        The names of the files to be read.

    Returns
    -------
    rmp_time_trace : np.ndarray
        The time trace of the RMP signal at the given location.
    '''
    rmp_time_trace = []
    rmp_pressure_trace = []

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with h5.File(file_path, 'r') as f:
            rmp_time = f[f'{rmp_location:02d}']['instants']['z0']['variables']['time_node'][:]
            rmp_time_trace.append(rmp_time)
            rmp_pressure = f[f'{rmp_location:02d}']['instants']['z0']['variables']['static_pressure_node'][:]
            rmp_pressure_trace.append(rmp_pressure)
            
    rmp_time_trace = np.concatenate(rmp_time_trace)
    rmp_pressure_trace = np.concatenate(rmp_pressure_trace)

    return np.array((rmp_time_trace,rmp_pressure_trace))

def plot_rmp_time_trace(data_dir:str,rmp_location:int,rmp_time_trace:np.ndarray):
    
    '''
    This function generates the plot of the RMP time trace.\n
    It takes as input the time trace of the RMP signal and generates a plot of it.\n
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    rmp_location : int
        The location of the RMP signal.
    rmp_time_trace : np.ndarray
        The time trace of the RMP signal.
    '''
    
    os.makedirs(os.path.join(data_dir,'rmp/timetrace'), exist_ok=True)
    image_path = os.path.join(data_dir,'rmp/timetrace')
    
    plt.figure(figsize=(10, 6))
    plt.plot(rmp_time_trace[0], rmp_time_trace[1], label=f'RMP {rmp_location}', color='k')
    plt.grid(True, which='both', ls='--')
    plt.xlabel('$Time~[s]$')
    plt.ylabel('$Static~Pressure~[Pa]$')
    plt.legend()
    plt.savefig(os.path.join(image_path,f'rmp{rmp_location}_timetrace.pdf'), dpi=600)
    
def plot_psd_rmp(data_dir:str,rmp_location:int,rmp_time_trace:np.ndarray):
    
    '''
    This function generates the plot of the RMP power spectral density.\n
    It takes as input the time trace of the RMP signal and generates a plot of its power spectral density.\n
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    rmp_time_trace : np.ndarray
        The time trace of the RMP signal.
    rmp_location : int
        The location of the RMP signal.
    '''
    
    os.makedirs(os.path.join(data_dir,'rmp/psd'), exist_ok=True)
    image_path = os.path.join(data_dir,'rmp/psd')
    
    from scipy.signal import welch
    
    dt = rmp_time_trace[0][1] - rmp_time_trace[0][0]
    n_chunk = 20
    lensg = len(rmp_time_trace[0])
    nperseg = lensg/n_chunk
    nfft = next_greater_power_of_2(int(nperseg))
    
    if nperseg > lensg:
        raise RuntimeError(f'Wrong value for nperseg={nperseg}')
    
    fs = 1 / dt
    f, Pxx = welch(rmp_time_trace[1], fs=fs, window='hann', nperseg=nperseg, nfft=nfft, scaling='density')

    plt.figure(figsize=(10, 6))
    plt.plot(f,10*np.log10(Pxx/4.0e-10), label=f'RMP {rmp_location} PSD', color='k')
    plt.grid(True, which='both', ls='--')
    plt.xlabel('$Frequency~[Hz]$')
    plt.ylabel('$PSD~[dB/Hz]$')
    ax=plt.gca()
    ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_path,f'rmp{rmp_location}_psd.pdf'), dpi=600)