""" Program for a quasi-static-state detector out of a dataset of IMU raw measurements.

    Based on "Automatic Calibration for Inertial Measurement Unit" form Cheuck, Lau, et. al.
"""
import numpy as np
import scipy.signal as sig

filter_order = 5


def rectification(dataset):
    rectification = []
    for data in dataset:
        rectification.append(np.linalg.norm(data)) # vector norm
        # rectification.append(np.abs(data))

    return rectification


def determine_static_coefficients(dataset):
    time = dataset[:, 0]
    acc_x = dataset[:, 1]
    acc_y = dataset[:, 2]
    acc_z = dataset[:, 3]
    mag_x =  dataset[:, 7]
    mag_y =  dataset[:, 8]
    mag_z =  dataset[:, 9]
    gyro_x = dataset[:,4] 
    gyro_y = dataset[:, 5]
    gyro_z = dataset[:, 6]


    # Low and Highfilter of the Accelerometer
    lowpass_filter_acc = sig.butter(filter_order, 0.25, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_acc = sig.butter(filter_order, 1, btype="highpass", output="sos", fs = 100.0)
    filtered_acc_x = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, acc_x))) # ACC_X
    filtered_acc_y = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, acc_y))) # ACC_Y
    filtered_acc_z = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, acc_z))) # ACC_Z

    # acc_x_hp = np.diff(acc_x,prepend=acc_x[0])
    # acc_y_hp = np.diff(acc_y,prepend=acc_y[0])
    # acc_z_hp = np.diff(acc_z, prepend=acc_z[0])

    # fig, ax1= plot.subplots()
    # ax1.plot(time, acc_x_hp)
    # ax1.plot(time, acc_y_hp)
    # ax1.plot(time, acc_z_hp)
    # ax1.set_title("Highpass")
    # plot.show()

    filtered_acc = np.array([(filtered_acc_x), (filtered_acc_y), (filtered_acc_z)])

    quasi_static_acc_coefficient = []
    
    for acc in filtered_acc.T:
        quasi_static_acc_coefficient.append(np.linalg.norm(acc)/1000) # be careful with the units


    # Magnetometer quasi-static detector
    lowpass_filter_mag = sig.butter(filter_order, 0.25, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_mag = sig.butter(filter_order, 1, btype="highpass", output="sos", fs=100.0) # Cutoff frequenz so niedrig gewaehlt da die Aenderungen in den Achsenwerte gering ist, Bewegungserkennung < 1 Hz

    filtered_mag_x = rectification(sig.sosfilt(highpass_filter_mag, mag_x)) # MAG_X
    # filtered_mag_y = sig.sosfilt(highpass_filter_mag, mag_y) # MAG_Y
    # filtered_mag_z = sig.sosfilt(highpass_filter_mag, mag_z) # MAG_Z

    
    mag_x_hp = np.diff(mag_x,prepend=mag_x[0])
    mag_y_hp = np.diff(mag_y,prepend=mag_y[0])
    mag_z_hp = np.diff(mag_z,prepend=mag_z[0])


    filtered_mag_x = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_mag, mag_x))) # MAG_X
    filtered_mag_y = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_mag, mag_y))) # MAG_Y
    filtered_mag_z = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_mag, mag_z))) # MAG_Z

    filtered_mag = np.array([(filtered_mag_x), (filtered_mag_y), (filtered_mag_z)])

    quasi_static_mag_coefficient = []
#
    for mag in filtered_mag.T:
        quasi_static_mag_coefficient.append(np.linalg.norm(mag)/49.4006)


    # Gyroscope quasi-static detector
    lowpass_filter_gyro = sig.butter(filter_order, 0.9, btype="lowpass", output="sos", fs=100.0)

    filtered_gyro_x = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_x)) # GYRO_X
    filtered_gyro_y = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_y)) # GYRO_Y
    filtered_gyro_z = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_z)) # GYRO_Z

    filtered_gyro = np.array([(filtered_gyro_x), (filtered_gyro_y), (filtered_gyro_z)])

    quasi_static_gyro_coefficient = []

    for gyro in filtered_gyro.T:
        quasi_static_gyro_coefficient.append(np.linalg.norm(gyro))

    static_coefficients = np.array(1./(1.+np.array(quasi_static_acc_coefficient)+np.array(quasi_static_mag_coefficient)+np.array(quasi_static_gyro_coefficient)))


    return static_coefficients