import os

import numpy as np
import matplotlib.pyplot as plt
import wavelet_noise as wn

plt.style.use('style.mplstyle')


def main():
    data_dir = "/home/daep/e.foglia/Documents/2A/13_gibbs/data"
    data_file = os.path.join(data_dir, "SherFWHsolid1_p_raw_data_250.h5")

    info_dict = wn.utils.get_data_info(data_file, verbose=False)
    p_te = wn.utils.extract_pressure_te(data_file, 50, info_dict['N'], False)

    # ==============================================
    #    De-normalize the data
    # ==============================================
    rho_ref = 1.225     # experiments density [kg/m^3]
    U_ref = 16          # experiments velocity [m/s]
    cref = 0.1356       # airfoil chord [m]
    p_dyn = rho_ref*U_ref**2  # dynamic pressure [Pa]
    # p_ref = 2e-5        # Reference pressure in Pa

    p_te *= p_dyn
    info_dict['T_s'] *= cref / U_ref
    info_dict['f_s'] *= U_ref / cref

    # ==============================================
    #    Basic information
    # ==============================================
    N = p_te.shape[0]       # Number of sample points
    T = info_dict['T_s']    # sample spacing
    fs = info_dict['f_s']   # Sampling frequency
    n_sens = p_te.shape[1]  # Number of sensors

    # ==============================================
    #    Wavelet Transform
    # ==============================================
    idx = n_sens // 2
    freq = np.logspace(np.log10(20), np.log10(20e3), 150)
    cw = wn.wavelet.cwt(p_te[:, idx], freq, wavelet='cmor2.5-1.5', fs=fs,
                        method='fft')

    # ==============================================
    #    Plot Scaleogram
    # ==============================================

    t = np.arange(N) * T
    t_grid, f_grid = np.meshgrid(t, freq)

    fig, ax = plt.subplots()
    sg = ax.pcolormesh(t_grid, f_grid, np.abs(cw), shading='nearest',
                       cmap='bone')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$f$ [Hz]')
    fig.colorbar(sg, ax=ax, label=r'$\vert \mathcal{W}[f] \vert$')
    plt.show()

    # ==============================================
    #  DWT
    # ==============================================
    wavelet = 'coif4'
    f, coeffs = wn.wavelet.dwt(p_te, wavelet=wavelet, mode='constant', axis=0,
                               fs=fs)
    print(len(coeffs))
    print(coeffs[2].shape)
    print(f)


if __name__ == "__main__":
    main()
