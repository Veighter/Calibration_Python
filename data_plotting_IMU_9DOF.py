import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
import numpy as np


def plot_measurements_out_of_file():
    with open('../../Datalogs/IMU_0.txt', 'r') as IMU_0:
        # TODO plot the raw measurements with matplotlib
        print('this')

def plot_measurements_out_of_data(measurements, quasi_static_coefficients=None, shw_t=False, calibrated=False):
    
    fig = plot.figure(tight_layout=True)
    gs = gridspec.GridSpec(3,1)

    time = np.array(measurements[:,0])
    time -= time[0]
    

    if quasi_static_coefficients is not None and shw_t==True:
        gs = gridspec.GridSpec(5,1)
        ax4 = fig.add_subplot(gs[3,0])
        ax4.plot(time, quasi_static_coefficients)
        ax4.set_xlabel("t [s]")
        ax4.set_title("Quasi-static-coefficients")
        ax5 = fig.add_subplot(gs[4,0])
        ax5.plot(time, quasi_static_coefficients>0.98)
        ax5.set_title("Quasi-static-states")
    if quasi_static_coefficients is not None and shw_t==False:
        gs = gridspec.GridSpec(4,1)
        ax4 = fig.add_subplot(gs[3,0])
        ax4.plot(time, quasi_static_coefficients)
        ax4.set_xlabel("t [s]")
        ax4.set_title("Quasi-static-coefficients")

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
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("uT")
    ax1.set_title(display_String+"magnetometer measurements")

    fig.align_labels()

    plot.show()
    #ax = fig.add_subplot(gs[1,:]) # second row slicing of the grid spec


