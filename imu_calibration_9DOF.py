
import ga_utils_IMU_Calibration as ga
import data_plotting as dp
import random as rd 
import numpy as np
import math
import pandas as pd
import scipy.signal as sig
from scipy.optimize import least_squares
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec

ORDER = 5
POPULATION_SIZE = 10e0 # typical size for differential evolution is 10*(number of inputs)
SEARCH_SPACE_ACC = [(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000)] # milli gs
SEARCH_SPACE_GYRO= [(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)] # degrees per seconde
SEARCH_SPACE_MAG = [(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5), (-100, 100),(-100,100), (-100,100)] # micro Tesla
CROSSOVER_PROBABILITY = 0.9
DIFFERENTIAL_WEIGHT = 0.8 # inital values guessed by wikipedia

quasi_static_states = []


def rectification(dataset):
    rectification = []
    for data in dataset:
        rectification.append(np.linalg.norm(data)) # vector norm
        # rectification.append(np.abs(data))

    return rectification

def get_measurements(filepath):
    """Get Method of the Raw Measurements of an IMU
    Args:
        filepath (String): Filepath of the Raw Measurements file

    Returns:
        ndarray: Raw Measurements (Row to Row)
    """
    raw_measurements_df = pd.read_csv(filepath)
    return raw_measurements_df.to_numpy() 

# TODO Bestimmen des quasi static coefficient der Raw Measurements, zur Auswahl der Messwerte der Kalibrierung
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
    lowpass_filter_acc = sig.butter(ORDER, 0.25, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_acc = sig.butter(ORDER, 1, btype="highpass", output="sos", fs = 100.0)
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
    lowpass_filter_mag = sig.butter(ORDER, 0.25, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_mag = sig.butter(ORDER, 1, btype="highpass", output="sos", fs=100.0) # Cutoff frequenz so niedrig gewaehlt da die Aenderungen in den Achsenwerte gering ist, Bewegungserkennung < 1 Hz

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
    lowpass_filter_gyro = sig.butter(ORDER, 0.9, btype="lowpass", output="sos", fs=100.0)

    filtered_gyro_x = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_x)) # GYRO_X
    filtered_gyro_y = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_y)) # GYRO_Y
    filtered_gyro_z = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_z)) # GYRO_Z

    filtered_gyro = np.array([(filtered_gyro_x), (filtered_gyro_y), (filtered_gyro_z)])

    quasi_static_gyro_coefficient = []

    for gyro in filtered_gyro.T:
        quasi_static_gyro_coefficient.append(np.linalg.norm(gyro))

    static_coefficients = np.array(1./(1.+np.array(quasi_static_acc_coefficient)+np.array(quasi_static_mag_coefficient)+np.array(quasi_static_gyro_coefficient)))


    return static_coefficients

def calibrate_sensor_ga(quasi_static_measurements, sensor):
    if sensor == "acc":
        return ga.algorithm(quasi_static_measurements, sensor, SEARCH_SPACE_ACC, 1000, POPULATION_SIZE, CROSSOVER_PROBABILITY, DIFFERENTIAL_WEIGHT)
    if sensor == "mag":
        return ga.algorithm(quasi_static_measurements, sensor, SEARCH_SPACE_MAG, 15, POPULATION_SIZE, CROSSOVER_PROBABILITY, DIFFERENTIAL_WEIGHT)
    if sensor == "gyro":
        return ga.algorithm(quasi_static_measurements, sensor, SEARCH_SPACE_GYRO, 1, POPULATION_SIZE, CROSSOVER_PROBABILITY, DIFFERENTIAL_WEIGHT)

def calibrate_sensor_lm(sensor, initial_parameter_vector):
    if sensor == "acc":
        return least_squares(acc_fitness, initial_parameter_vector)

def acc_fitness(parameter_vector):
    cost = 0
    print(len(quasi_static_states))
    for state in quasi_static_states:
        cost += ((1000)-np.linalg.norm(np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]]) @ state.T-np.array([parameter_vector[9], parameter_vector[10], parameter_vector[11]])))**2
    print(f"Cost: {cost}")
    return np.array(cost)

def get_calibrated_measurement(raw_measurements, calibration_params, sensor):


    calibrated_measurements = []
    if sensor == "acc":
        theta = np.array([calibration_params[0:3], calibration_params[3:6], calibration_params[6:9]])
        bias = np.array([calibration_params[9], calibration_params[10], calibration_params[11]])
        for raw_measurement in raw_measurements:
                calibrated_measurements.append(theta@raw_measurement.T - bias)

    return np.array(calibrated_measurements)



def main ():
    raw_measurements = get_measurements('../../Datalogs/IMU_0.txt') # Format of Raw Measurements is that as in the datalogs

    raw_measurements[:,4] = raw_measurements[:,4]*math.pi/180 # degree to radians
    raw_measurements[:,5] = raw_measurements[:,5]*math.pi/180
    raw_measurements[:,6] = raw_measurements[:,6]*math.pi/180

    print(f"Anzahl Samples: {len(raw_measurements)}")    

    quasi_static_coefficients = determine_static_coefficients(raw_measurements)

   # dp.plot_measurements_out_of_data(raw_measurements, quasi_static_coefficients)

    indixes = quasi_static_coefficients > 0.98

    quasi_static_measurements = np.array([raw_measurements[i,:] for i in range(len(raw_measurements)) if indixes[i]])
    quasi_static_states = quasi_static_measurements
    print(f"Potenzielle statische Zustaende: {len(quasi_static_measurements)}")
    #calibration_parameters = calibrate_sensor_ga(quasi_static_measurements[:, 1:4], "acc")
    calibration_parameters = calibrate_sensor_lm("acc", [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    
    calibrated_measurements = get_calibrated_measurement(raw_measurements[:, 1:4], calibration_parameters, sensor="acc")

    fig, [ax1,ax2] = plot.subplots(2,1)
    ax1.plot(raw_measurements[:,0], calibrated_measurements)
    ax2.plot(raw_measurements[:,0], raw_measurements[:,1:4])
    plot.show()

if __name__ == "__main__":
    main()
