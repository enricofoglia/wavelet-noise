from utils import *
import numpy as np

#rmp_location = [1,2,3,5,6,7,9,11,21,22,23,24,25]
mic_numbers = np.arange(25,36)
frames = np.arange(0,28,1)
data_dir = '/storage/renj3003/cd-airfoil/5deg-dns/data/mics'
file_names = ['probes_data_p1.h5', 'probes_data_p2.h5']

#------------- RMP post-processing ---------------#

#for rmp in rmp_location:
    #timetrace = rmp_timetrace(rmp, data_dir, *file_names)
    #plot_rmp_time_trace(data_dir, rmp, timetrace)
    #plot_psd_rmp(data_dir, rmp, timetrace)

#------------- RMP post-processing ---------------#

#for mic in mic_numbers:
    #timetrace = sherfwh_timetrace(data_dir, f'Mic_{mic:06d}')
    #plot_sherfwh_time_trace(data_dir, mic, timetrace)
    #plot_sherfwh_psd(data_dir, mic, timetrace)

#---------- Inst plane post processing ----------#

#for frame in frames:
    #pf_plane_plot(data_dir, 0.5, 0.2, [-0.22,-0.02], [-0.01,0.03], -5000, 5000, '$\omega_z~[m/s]$',f'{frame:02d}t_z-vor.txt')

#--------- Pressure Coefficient plotting ---------#

cp_plot('/storage/renj3003/cd-airfoil/5deg-dns/data/snc','cp_midspan.csv',101325,1.25,16)
