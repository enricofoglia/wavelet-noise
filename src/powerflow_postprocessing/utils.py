import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import pandas as pd
import pdb
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
    
    os.makedirs(os.path.join(data_dir,'images/rmp/timetrace'), exist_ok=True)
    image_path = os.path.join(data_dir,'images/rmp/timetrace')

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
    
    os.makedirs(os.path.join(data_dir,'images/rmp/psd'), exist_ok=True)
    image_path = os.path.join(data_dir,'images/rmp/psd')

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

def sherfwh_timetrace(data_dir:str, file_name:str):
    
    '''
    This function generates the time trace of the Sherfwh signal.\n 
    It takes as input the directory where the data is stored and the names of the files to be read.\n 
    It returns the time trace of the Sherfwh signal.\n
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    file_name : str
        The names of the files to be read.

    Returns
    -------
    sherfwh_time_trace : np.ndarray
        The time trace of the Sherfwh signal.
    '''
    
    data = pd.read_csv(os.path.join(data_dir, file_name), delimiter='\s+', skiprows=1, names=['iter', 'time', 'pressure', 'nbcontrib'])
    time = data['time']
    pressure = data['pressure']
    contribution = data['nbcontrib']
    filtered = np.where(contribution!=0)[0]

    time = time[filtered[0]:filtered[-1]+1]
    pressure = pressure[filtered[0]:filtered[-1]+1]
    
    return np.array((time, pressure))
        
def plot_sherfwh_time_trace(data_dir:str, mic_num:int, sherfwh_time_trace:np.ndarray):
    
    '''
    This function generates the plot of the Sherfwh time trace.\n
    It takes as input the time trace of the Sherfwh signal and generates a plot of it.\n
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    mic_num : int
        The number of the microphone where the Sherfwh signal was measured.
    sherfwh_time_trace : np.ndarray
        The time trace of the Sherfwh signal.
    '''
    
    os.makedirs(os.path.join(data_dir,'images/sherfwh/timetrace'), exist_ok=True)
    image_path = os.path.join(data_dir,'images/sherfwh/timetrace')

    plt.figure(figsize=(10, 6))
    plt.plot(sherfwh_time_trace[0], sherfwh_time_trace[1], label=f'Sherfwh', color='k')
    plt.grid(True, which='both', ls='--')
    plt.xlabel('$Time~[s]$')
    plt.ylabel('$Static~Pressure~[Pa]$')
    plt.legend()
    plt.savefig(os.path.join(image_path,f'mic{mic_num}_sherfwh_timetrace.pdf'), dpi=600)

def plot_sherfwh_psd(data_dir:str, mic_num:int, sherfwh_time_trace:np.ndarray):
    
    '''
    This function generates the plot of the Sherfwh power spectral density.\n
    It takes as input the time trace of the Sherfwh signal and generates a plot of its power spectral density.\n
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    mic_num : int
        The number of the microphone where the Sherfwh signal was measured.
    sherfwh_time_trace : np.ndarray
        The time trace of the Sherfwh signal.
    '''
    
    os.makedirs(os.path.join(data_dir,'images/sherfwh/psd'), exist_ok=True)
    image_path = os.path.join(data_dir,'images/sherfwh/psd')

    from scipy.signal import welch
    
    dt = sherfwh_time_trace[0][1] - sherfwh_time_trace[0][0]
    n_chunk = 4
    lensg = len(sherfwh_time_trace[0])
    nperseg = lensg/n_chunk
    nfft = next_greater_power_of_2(int(nperseg))
    
    if nperseg > lensg:
        raise RuntimeError(f'Wrong value for nperseg={nperseg}')
    
    fs = 1 / dt
    f, Pxx = welch(sherfwh_time_trace[1], fs=fs, window='hann', nperseg=nperseg, nfft=nfft, detrend='constant')

    plt.figure(figsize=(10, 6))
    plt.plot(f,10*np.log10(Pxx/4.0e-10), label=f'Sherfwh mic {mic_num} PSD', color='k')
    plt.grid(True, which='both', ls='--')
    plt.xlabel('$Frequency~[Hz]$')
    plt.ylabel('$PSD~[dB/Hz]$')
    ax=plt.gca()
    ax.set_xscale('log')
    plt.legend()
    plt.xlim([100,40000])
    plt.tight_layout()
    plt.savefig(os.path.join(image_path,f'mic{mic_num}_sherfwh_psd.pdf'), dpi=600)

def pf_plane_plot(data_dir:str, lx:float, ly:float, x_lim:list, y_lim:list, v_min:float, v_max:float, label:str, file:str):

    '''This function generates the plot of the power flow in the plane.\n
    It takes as input the directory where the data is stored, the dimensions of the plane,\n
    the limits of the plot, and the names of the files to be read. It generates a plot
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    lx : float
        The length of the plane in the x direction.
    ly : float
        The length of the plane in the y direction.
    x_lim : list
        The limits of the plot in the x direction.
    y_lim : list
        The limits of the plot in the y direction.
    v_min : float
        The minimum value of the colorbar.
    v_max : float
        The maximum value of the colorbar.
    label : str
        The label of the colorbar.
    files : str
        The names of the file to be read.
    '''

    import matplotlib.ticker as ticker
    
    os.makedirs(os.path.join(data_dir,'images/plane/psd'), exist_ok=True)
    image_path = os.path.join(data_dir,'images/plane/psd')

    u_tip = 1#(np.pi*(diameter/2)*speed)/60
    levels = np.linspace(v_min, v_max, 101)
    
    print(f'Reading file: {file}')
    data=np.genfromtxt(os.path.join(data_dir,file),skip_header=15,filling_values=0)
    ny, nx = data.shape
    x = np.linspace(-lx/2,lx/2,nx)
    y = np.linspace(-ly/2,ly/2,ny)
    X, Y = np.meshgrid(x, y)
    Z = data
    print('Plotting')
    cp=plt.contourf(X,Y,Z/u_tip,levels=levels,cmap='turbo',vmin=v_min,vmax=v_max)
    cbar = plt.colorbar(cp)
    cbar.set_label(label, fontsize=14)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.xlabel('$x~[m]$')
    plt.ylabel('$y~[m]$')
    plt.xlim(x_lim)
    plt.xticks(fontsize=12)
    plt.ylim(y_lim)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    print('Saving plot')
    plt.savefig(os.path.join(data_dir,'images',file.split('.')[0] +'.png'), dpi=600)
    plt.show()

def cp_plot(data_dir:str, file:str, p_char:float, rho_char:float, u_inf:float):
    
    '''
    This function generates the plot of the pressure coefficient in the plane.\n
    It takes as input the directory where the data is stored and the name of the file to be read.\n
    It generates a plot of the pressure coefficient in the plane.\n
    \n
    Parameters
    ----------
    data_dir : str
        The directory where the data is stored.
    file : str
        The name of the file to be read.
    '''
    
    os.makedirs(os.path.join(data_dir,'images/plane/cp'), exist_ok=True)
    image_path = os.path.join(data_dir,'images/plane/cp')

    data = pd.read_csv(os.path.join(data_dir, file))
    positions = data[data.columns.tolist()[0]]
    positions = positions - np.min(positions)
    cp_data = data[data.columns.tolist()[1]]
    
    position_ss = positions[:np.where(positions==np.min(positions))[0][-1]]
    variable_ss = cp_data[:np.where(positions==np.min(positions))[0][-1]]

    position_ps = positions[np.where(positions==np.min(positions))[0][-1]:]
    variable_ps = cp_data[np.where(positions==np.min(positions))[0][-1]:]

    cp_suction = -((variable_ss - p_char) / (0.5 * rho_char * u_inf ** 2))
    cp_pressure = -((variable_ps - p_char) / (0.5 * rho_char * u_inf ** 2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(position_ss/np.max(positions), cp_suction,'k')
    plt.plot(position_ps/np.max(positions), cp_pressure,'k')
    plt.grid(True, which='both', ls='--')
    plt.xlabel('$x/c$')
    plt.ylabel('$-C_p$')
    plt.tight_layout()
    plt.savefig(os.path.join(image_path,f'{file.split(".")[0]}.pdf'), dpi=600)
