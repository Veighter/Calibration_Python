import matplotlib.pyplot as plt
import ga_utils_IMU_Calibration as ga
import data_plotting_IMU_9DOF as dp
import quasi_static_state_detector as qssd
import calibration_wo_external_equipment as cwee
import numpy as np
import math
import pandas as pd
from scipy.optimize import least_squares




# get local magnitudes of magnetic field and acceleration at "https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm" and "https://www.ptb.de/cms/en/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html" 
# https://www.mapcoordinates.net/de
# 49,402.4 nT, 9.81158 m/s**2



POPULATION_SIZE = 10e0 # typical size for differential evolution is 10*(number of inputs)
SEARCH_SPACE_ACC = [(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000)] # milli gs
SEARCH_SPACE_GYRO= [(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)] # degrees per seconde
SEARCH_SPACE_MAG = [(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5), (-100, 100),(-100,100), (-100,100)] # micro Tesla
CROSSOVER_PROBABILITY = 0.9
DIFFERENTIAL_WEIGHT = 0.8 # inital values guessed by wikipedia

def get_measurements(filepath):
    """Get Method of the Raw Measurements of an IMU
    Args:
        filepath (String): Filepath of the Raw Measurements file

    Returns:
        ndarray: Raw Measurements (Row to Row)
    """
    raw_measurements_df = pd.read_csv(filepath)
    return raw_measurements_df.to_numpy() 

def calibrate_sensor_ga(quasi_static_measurements, sensor):
    if sensor == "acc":
        return ga.algorithm(quasi_static_measurements, sensor, SEARCH_SPACE_ACC, 1000, POPULATION_SIZE, CROSSOVER_PROBABILITY, DIFFERENTIAL_WEIGHT)
    if sensor == "mag":
        return ga.algorithm(quasi_static_measurements, sensor, SEARCH_SPACE_MAG, 15, POPULATION_SIZE, CROSSOVER_PROBABILITY, DIFFERENTIAL_WEIGHT)
    if sensor == "gyro":
        return ga.algorithm(quasi_static_measurements, sensor, SEARCH_SPACE_GYRO, 1, POPULATION_SIZE, CROSSOVER_PROBABILITY, DIFFERENTIAL_WEIGHT)

def calibrate_sensor_lm(sensor, quasi_static_measurements):
    if sensor == "acc":
        initial_parameter_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        return least_squares(acc_fitness, initial_parameter_vector, args=(quasi_static_measurements, []), verbose=1, max_nfev=100000000, method='lm')

def acc_fitness(parameter_vector, *args):
    quasi_static_states, _ = args
    cost = 0
    for state in quasi_static_states:
        cost += ((1000)-np.linalg.norm(np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]]) @ state.T-np.array([parameter_vector[9], parameter_vector[10], parameter_vector[11]])))**2
    print(f"Cost: {cost}")
    return np.array(cost)

def mag_fitness(parameter_vector, *args):
    pass

def gyro_fitness(parameter_vector, *args):
    pass

def get_calibrated_measurements(raw_measurements, calibration_params, sensor):
    calibrated_measurements = []
    if sensor == 'acc':
        theta = np.array([calibration_params[0:3], calibration_params[3:6], calibration_params[6:9]])
        bias = np.array([calibration_params[9], calibration_params[10], calibration_params[11]])
        for raw_measurement in raw_measurements:
                calibrated_measurements.append((theta@raw_measurement.T) - bias.T)
    # if sensor == 'mag':
    #     axis_misalignment_matrix = np.array([calibration_params[0:3], calibration_params[3:6], calibration_params[6:9]])
    #     scaling_matrix = np.eye(3)
    #     scaling_matrix[0,0]= calibration_params[9]
    #     scaling_matrix[1,1]= calibration_params[10]
    #     scaling_matrix[2,2]=calibration_params[11]
    #     bias = np.array([calibration_params[12], calibration_params[13], calibration_params[14]])
    #     for raw_measurement in raw_measurements:
    #             calibrated_measurements.append((np.linalg.inv(axis_misalignment_matrix)@scaling_matrix@axis_misalignment_matrix@raw_measurement.T)-bias.T)
    if sensor == 'mag':    
      #  Define calibration parameters
        A = np.array(   [[0.941657, -0.005734, -0.004382],
                        [-0.005734, 0.989327, 0.011440],
                        [-0.004382, 0.011440, 0.949412]])
        b = np.array([-21.348454, 15.020109, 60.648129])
        
        for raw_measurement in raw_measurements:
            calibrated_measurements.append(A@(raw_measurement-b).T)
    return np.array(calibrated_measurements)

def main ():

    # Allan Variance
    # raw_measurements = get_measurements('../../Datalogs/Allan Variance/IMU_0.txt') # Format of Raw Measurements is that as in the datalogs for the Allan Variance

    # allan_variance_x = cwee.allan_variance(cwee.parse_allan_variance(raw_measurements[:,4]))
    # allan_variance_y = cwee.allan_variance(cwee.parse_allan_variance(raw_measurements[:,5]))
    # allan_variance_z = cwee.allan_variance(cwee.parse_allan_variance(raw_measurements[:,6]))

    # time = raw_measurements[:,0]
    # fig, ax = plt.subplots(1,1)
    # ax.plot(allan_variance_x[0], allan_variance_x[1])
    # ax.plot(allan_variance_y[0], allan_variance_y[1])
    # ax.plot(allan_variance_z[0], allan_variance_z[1])
    # plt.show()


    # Static Detection
    raw_measurements = get_measurements('../../Datalogs/IMU_0.txt')

    
    raw_measurements[:,4] = raw_measurements[:,4]*math.pi/180 # degree to radians
    raw_measurements[:,5] = raw_measurements[:,5]*math.pi/180
    raw_measurements[:,6] = raw_measurements[:,6]*math.pi/180

    time = (raw_measurements[:,0]-raw_measurements[0,0])/1000 # in sek
    acc_measurements = raw_measurements[:,1:4]
    gyro_measurements = raw_measurements[:,4:7]
    mag_measurements = raw_measurements[:,7:10]

    static_detector_values_list = cwee.static_detector(acc_measurements, 150)

    static_intervals = [cwee.static_interval_detector(static_detector_values_list[i][0]) for i in range(len(static_detector_values_list))]
    thresholds = [static_detector_values_list[i][1] for i in range(len(static_detector_values_list))]

    

    opt_param_acc, opt_threshold = cwee.optimize_acc_lm(acc_measurements, static_intervals_list=static_intervals, thresholds=thresholds)
   # opt_param_mag = cwee.optimize_mag_diff_ev(mag_measurements, static_intervals[thresholds.index(opt_threshold)])
   # opt_param_mag = cwee.optimize_mag_lm(mag_measurements, static_intervals[thresholds.index(opt_threshold)])
    avg_measurements_opt_static_interval = np.array([cwee.avg_measurements_static_interval(mag_measurements, static_interval) for static_interval in static_intervals[thresholds.index(opt_threshold)]])


    print(f"Opt_Params Accelerometer: {opt_param_acc}")
  #  print(f"Opt_Params Magnetometer: {opt_param_mag}")
    #print(f"Avg_measurements in Interval: {avg_measurements_opt_static_interval}")

    calibrated_acc_measurements = get_calibrated_measurements(acc_measurements, opt_param_acc, "acc")

    opt_param_mag = None
    calibrated_avg_mag_measurements = get_calibrated_measurements(avg_measurements_opt_static_interval, opt_param_mag,"mag")

    # fig, [ax1, ax2] = plt.subplots(2,1)
    # ax1.plot(time, calibrated_acc_measurements)
    # ax2.plot(time, acc_measurements)
    # plt.show()


    #calibrated_mag_measurements = get_calibrated_measurements(mag_measurements, opt_param_mag, "mag")

    
    # fig, [ax1, ax2] = plt.subplots(2,1)
    # ax1.plot(time,calibrated_mag_measurements)
    # ax2.plot(time, mag_measurements)
    # plt.show()


    # Plot XZ data
    plt.figure()
    plt.plot(avg_measurements_opt_static_interval[:,0], avg_measurements_opt_static_interval[:,2], 'b*', label='Raw Meas.')
    plt.plot(calibrated_avg_mag_measurements[:,0], calibrated_avg_mag_measurements[:, 2], 'r*', label='Calibrated Meas.')
    plt.title('XZ Magnetometer Data')
    plt.xlabel('X [uT]')
    plt.ylabel('Z [uT]')
    plt.legend()
    plt.grid()
    plt.axis('equal')

    # Plot XY data
    plt.figure()
    plt.plot(avg_measurements_opt_static_interval[:,0], avg_measurements_opt_static_interval[:,1], 'b*', label='Raw Meas.')
    plt.plot(calibrated_avg_mag_measurements[:,0], calibrated_avg_mag_measurements[:, 1], 'r*', label='Calibrated Meas.')
    plt.title('XY Magnetometer Data')
    plt.xlabel('X [uT]')
    plt.ylabel('Y [uT]')
    plt.legend()
    plt.grid()
    plt.axis('equal')

    # Plot YZ data
    plt.figure()
    plt.plot(avg_measurements_opt_static_interval[:,1], avg_measurements_opt_static_interval[:,2], 'b*', label='Raw Meas.')
    plt.plot(calibrated_avg_mag_measurements[:,1], calibrated_avg_mag_measurements[:, 2], 'r*', label='Calibrated Meas.')
    plt.title('YZ Magnetometer Data')
    plt.xlabel('Y [uT]')
    plt.ylabel('Z [uT]')
    plt.legend()
    plt.grid()
    plt.axis('equal')

    ## Plot 3D Scatter
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    magnitude_mag_local = 49.4006 #T

    ## Natuerlich nur die Punkte nutzen, welche geaveraget wurden!!!
    # Das hier ist sinnlos
    for avg_measurement in avg_measurements_opt_static_interval:

        mag_x = avg_measurement[0]
        mag_y = avg_measurement[1]
        mag_z = avg_measurement[2]
        ax.scatter(mag_x, mag_y, mag_z, color='r')

    for calibrated_avg_mag_measurement in calibrated_avg_mag_measurements:

        cal_mag_x = calibrated_avg_mag_measurement[0]
        cal_mag_y = calibrated_avg_mag_measurement[1]
        cal_mag_z = calibrated_avg_mag_measurement[2]

        ax.scatter(cal_mag_x, cal_mag_y, cal_mag_z, color='b')

    # with open("cal_mag_avg_measurements.txt", "w") as file:
    #     file.write(str(calibrated_avg_mag_measurements))
        
    # with open("mag_avg_measurements", "w") as file:
    #     file.write(str(avg_measurements_opt_static_interval))

    ax.set_title('3D Scatter Plot of Magnetometer Data')
    ax.set_xlabel('X [uT]')
    ax.set_ylabel('Y [uT]')
    ax.set_ylabel('Z [uT]')

    plt.show()



# # Randfall rechts fehlt!!
# def static_interval_detector(static_detector_values):
#     static_intervals = [] 
#     static_samples_interval = t_wait*sample_rate

#     left_bound = 0
#     right_bound = 0
#     flag = 0

#     i = 0
#     while i<(len(static_detector_values)):
#         if static_detector_values[i] == True and flag == 0:
#             flag = 1
#             left_bound = i
#         if static_detector_values[i] == False and flag ==1:
#             flag = 0
#             right_bound = i
#             if right_bound-left_bound+1>=static_samples_interval:
#                 static_intervals.append((left_bound, right_bound))
#         i+=1

#     return static_intervals













    #print(f"Anzahl Samples: {len(raw_measurements)}")    

    #quasi_static_coefficients = qssd.determine_static_coefficients(raw_measurements)

    #indixes = quasi_static_coefficients > 0.98
    
    #quasi_static_measurements = np.array([raw_measurements[i,:] for i in range(len(raw_measurements))  if indixes[i]] )
    #print(f"Anzahl Quasi-statischer-Zustaende: {len(quasi_static_measurements)}")

    #dp.plot_measurements_out_of_data(raw_measurements, quasi_static_coefficients, shw_t=True)

    # quasi_static_states = quasi_static_measurements
    # #print(f"Potenzielle statische Zustaende: {quasi_static_states}")
    # #calibration_parameters = calibrate_sensor_ga(quasi_static_measurements[:, 1:4], "acc")
    # calibration_parameters = calibrate_sensor_lm("acc", quasi_static_measurements[:, 1:4])["x"]
    
    # calibrated_measurements = get_calibrated_measurement(acc_measurements, calibration_parameters, sensor="acc")

    # fig, [ax1,ax2] = plot.subplots(2,1)
    # ax1.plot(raw_measurements[:,0], calibrated_measurements)
    # ax2.plot(raw_measurements[:,0], raw_measurements[:,1:4])
    # plot.show()

if __name__ == "__main__":
    main()
