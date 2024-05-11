import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
import numpy as np


def plot_measurements_out_of_file():
    with open('../../Datalogs/IMU_0.txt', 'r') as IMU_0:
        # TODO plot the raw measurements with matplotlib
        print('this')



def plot_measurements_out_of_data(measurements, static_detector_values, calibrated=False, port_number=0):
    
    title ="" 
    fig = plot.figure(tight_layout=True)
    gs = gridspec.GridSpec(3,1)

    time = np.array(measurements[:,0])
    time /= 1e3
    time -= time[0]
    

    if static_detector_values is not None:
        gs = gridspec.GridSpec(4,1)
        ax4 = fig.add_subplot(gs[3,0])
        ax4.plot(time, static_detector_values)
        ax4.set_xlabel("t [s]")
        ax4.set_title("Static Detector Values")
        title = f'Calibration Measurements of IMU {port_number}'
        fig.suptitle(title, fontsize=14)

    if calibrated:
        display_String = "Calibrated "
    else:
        display_String = "Raw "
    
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(time, measurements[:,1])
    ax1.plot(time, measurements[:,2])
    ax1.plot(time, measurements[:,3])
    ax1.set_ylabel("mg")
    ax1.set_title(display_String+"accelerometer measurements")

    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(time, measurements[:,4])
    ax1.plot(time, measurements[:,5])
    ax1.plot(time, measurements[:,6])
    ax1.set_ylabel("rad/s")
    ax1.set_title(display_String+"gyroscope measurements")

    ax1 = fig.add_subplot(gs[2,0])
    ax1.plot(time, measurements[:,7])
    ax1.plot(time, measurements[:,8])
    ax1.plot(time, measurements[:,9])
    ax1.set_ylabel("uT")
    ax1.set_title(display_String+"magnetometer measurements")

    fig.align_labels()

    plot.show()

    fig.savefig(f'../../Figures/{title}', bbox_inches='tight')



