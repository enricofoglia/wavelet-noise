from utils import *

rmp_location = 0
data_dir = 'data'
file_names = ['probe_data_p1.h5', 'probe_data_p2.h5']

timetrace = rmp_timetrace(rmp_location, data_dir, *file_names)
plot_rmp_time_trace(data_dir, rmp_location, timetrace)
plot_psd_rmp(data_dir, rmp_location, timetrace)