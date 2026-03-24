import matplotlib.pyplot as plt
from utils import *

file_path_calibration = ''
file_name_calibration = ''

file_path_experiments = ''
file_name_experiments = ''

p_ref, p_rmp = load_calibration_data(file_path_calibration, file_name_calibration)
time, p_target = load_experimental_data(file_path_experiments, file_name_experiments)
f, H, Cxy, delay, group_delay, p_calibrated_t = calibrate_rmp_signal(p_ref, p_rmp, p_target, time)

print(40 * '-')
print(f'Estimated delay: {delay:.4f} ms')
print(40 * '-')

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.semilogx(f, 20 * np.log10(np.abs(H)))
plt.title('Transfer Function (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.subplot(3, 1, 2)
plt.semilogx(f, np.angle(H, deg=True))
plt.title('Transfer Function Phase (degrees)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.semilogx(f, Cxy)
plt.title('Coherence Quality Check')
plt.ylim(0, 1.1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()