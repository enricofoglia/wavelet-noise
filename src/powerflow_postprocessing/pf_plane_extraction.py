# Direct Recorded Script
# PowerVIZ 6-2025-R1 ( 6.5.3 )
# Date: Wed Mar 11 08:47:40 2026

import numpy as np

ts_i = 2022912
ts_s = 2042368
ts_n = 2548224

timesteps = np.arange(ts_i, ts_n + 1, ts_s - ts_i)
variables = ['X-Velocity','Y-Velocity','Z-Velocity','X-Vorticity','Y-Vorticity','Z-Vorticity'] # z coordinate is the out of plane one
variables_label = ['x-vel','y-vel','z-vel','x-vor','y-vor','z-vor']

project1=app.currentProject
slice1=project1.get(name="Slice1", type="Slice")

for var,label in zip(variables, variables_label):
    scalarPropertySet11=project1.get(name=var, type="ScalarPropertySet")
    
    for frame,ts in enumerate(timesteps):
        project1.timeStep=ts
        slice1.imageSPS=scalarPropertySet11
        slice1.saveAsciiData(filename="/storage/renj3003/cd-airfoil/5deg-dns/powerflow_postprocessing/{:02d}t_{}.txt".format(frame,label), format="Matrix", dataSamplesPerCellWidth=0.25, significantDigits=7)




