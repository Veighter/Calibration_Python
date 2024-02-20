# halbe stunde daten sammeln, entspricht bei einer sample-Frequenz von 100 Hz 180000 samplen
# for statistic reasoning - damit das Intervall nicht statistisch zu viele Daten enthaelt - ist die Grenze 225s

import numpy as np
from scipy.optimize import least_squares

magnitude_acc_local = 1000.16106 # mg
(magnitude_mag_local) = 49.4006 # uTesla

sample_rate = 100   # Hz
t_wait = 2          # s


def parse_allan_variance(gyro_measurements):
    collection_time = 3600 # time of static data collection in seconds
    cutoff_sample = collection_time*sample_rate # data we would face if we sample at the sample rate for 1h
    
    return gyro_measurements[0:cutoff_sample]

# [x] Allan Variance von Gyroskop Daten bestimmen -> gleich grosse Intervall wichtig
def allan_variance(gyro_measurements):
    # one axis allan variance computing
    time_begin = 1 # begin time in s
    time_end = 400 # end time in s
    sample_time = 1/sample_rate
    measurement_time = int(len(gyro_measurements)/100)


    num_elements = int((time_end - time_begin) / sample_time) + 1
    times = [time for time in np.linspace(start=time_begin,stop=time_end,num=num_elements) if measurement_time%time==0]

    allan_variance = np.zeros(shape=(len(times),))

    for i, time in enumerate(times): # evenly space 0.1 Zeitsteps
        K = int(measurement_time/float(time))
        sample_number_intervall = (int)(time*sample_rate)
        
        for k in range(K-1):
            left_bound = (int)(k*sample_number_intervall)
            right_bound = (int)((k+1)*sample_number_intervall) # goes through the equal length intervalls
            
            average_1 = np.mean(gyro_measurements[left_bound: right_bound])
            average_2 = np.mean(gyro_measurements[right_bound: right_bound+sample_number_intervall])
            
            allan_variance[i]+=(average_2-average_1)**2
        allan_variance[i] /= (2*K)

    return (times, allan_variance)

# [x] implement the static Detector
def static_detector(acc_dataset, T_init):
    detector_threshold_tuples = [] # tuple holding [threshold, s_detector_values]

    sample_number_init_intervall = T_init*sample_rate

    roh_init = static_detector_functional(acc_dataset[0:sample_number_init_intervall, :])
    
    sample_number_intervall = t_wait*sample_rate
    
    for k in range(1,10): # Anomalien, wenn das k>1 ist
        size_acc = len(acc_dataset)
        threshold = k*roh_init**2
        static_detector_values = np.zeros(shape = (size_acc,) )
        left_bound = (int)(sample_number_init_intervall-sample_number_intervall/2)
        right_bound = (int)(sample_number_init_intervall+sample_number_intervall/2)
        for t, _ in enumerate(acc_dataset[left_bound: ,:], sample_number_init_intervall):

            variance_magnitude = static_detector_functional(acc_dataset[left_bound:right_bound,:])
            if variance_magnitude< threshold:
                static_detector_values[t] = 1
            if variance_magnitude >= threshold:
                static_detector_values[t] = 0
            
            left_bound += 1
            right_bound += 1

            if right_bound>=size_acc:
                break
        
        static_intervals = static_interval_detector(static_detector_values)
        detector_threshold_tuples.append((static_detector_values, threshold))      
    

    return detector_threshold_tuples

def static_detector_functional(acc_dataset):
    acc_x = acc_dataset[:,0]
    acc_y = acc_dataset[:,1]
    acc_z = acc_dataset[:,2]
    return np.linalg.norm([np.var(acc_x), np.var(acc_y), np.var(acc_z)])

def static_interval_detector(static_detector_values):
    static_intervals = [] # indixes of the M distinct intervalls of the window size t_wait
    static_samples_interval = t_wait*sample_rate

    i = 0
    while i<(len(static_detector_values)):
        if static_detector_values[i] != False:
            left_bound = i
            right_bound = i+static_samples_interval # intervall length has to be t_wait
            while(static_detector_values[i]==True):
                i+=1
            if (i-1) >= right_bound:
                static_intervals.append((left_bound, right_bound))
        i+=1

    return static_intervals

# TODO implement the optimizer for the parameters for the sensor error model with the levenberg marquard algorithnm
# TODO use the same threshold, default is min cost threshold accelerometer optimum
def optimize_acc_lm(dataset, static_intervals_list, thresholds):
    opt_param = []
    max_nfev = 100000
    ftol=1e-10

    for static_intervals in static_intervals_list:
        avg_measurements = []
        calibration_params = None
        for static_interval in static_intervals:
            avg_measurement = np.mean(dataset[static_interval[0]:static_interval[1]+1, :], axis=0)
            avg_measurements.append(avg_measurement)
        initial_parameter_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] # 12 Params for the Matrix and the bias
        calibration_params = least_squares(acc_residuals, initial_parameter_vector, args=(avg_measurements, []), max_nfev=max_nfev, ftol=ftol)
        opt_param.append((calibration_params['x'], calibration_params['cost']))
        
    costs = [cost for _, cost in opt_param]
    min_cost = costs.index(min(costs))
    
    return opt_param[min_cost][0], thresholds[min_cost]

def calibration_algorithm(raw_measurements, sample_rate):
    time = raw_measurements[:, 0]
    acc_x = raw_measurements[:, 1]
    acc_y = raw_measurements[:, 2]
    acc_z = raw_measurements[:, 3]
    mag_x =  raw_measurements[:, 7]
    mag_y =  raw_measurements[:, 8]
    mag_z =  raw_measurements[:, 9]
    gyro_x = raw_measurements[:,4] 
    gyro_y = raw_measurements[:, 5]
    gyro_z = raw_measurements[:, 6]

    t_init = allan_variance(np.array([time, gyro_x, gyro_y, gyro_z]).T, sample_rate)
    pass

def sensor_error_model_acc_transformation(parameter_vector, static_measurement):
    axis_misalignment_and_scaling_matrix = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    bias = np.array([parameter_vector[9], parameter_vector[10], parameter_vector[11]])
    return axis_misalignment_and_scaling_matrix @ (static_measurement-bias).T

# axis misalignment, without cross axis scaling!!
def sensor_error_model_mag_transformation(parameter_vector, static_measurement):
    axis_misalignment_matrix = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    scaling_matrix = np.eye(3)
    scaling_matrix[0,0]=parameter_vector[9]
    scaling_matrix[1,1]=parameter_vector[10]
    scaling_matrix[2,2]=parameter_vector[11]

    bias = np.array([parameter_vector[12], parameter_vector[13], parameter_vector[14]])
    
    return np.linalg.inv(axis_misalignment_matrix)@scaling_matrix@axis_misalignment_matrix@(static_measurement-bias).T
# TODO implement fitness function (look Equaiton 9 and 10 "Robust and Easy Implementation for IMU Calibration")
def acc_residuals(parameter_vector, *args):
    static_measurements, _ = args
    residuals = np.zeros(len(static_measurements))
    for i, measurement in enumerate(static_measurements):
        residuals[i]= ((magnitude_acc_local)**2-np.linalg.norm(sensor_error_model_acc_transformation(parameter_vector, measurement))**2)**2
    return residuals
    

def gyro_fitness():
    pass

def mag_residuals(parameter_vector, *args):
    static_measurements = args
    residuals = np.zeros(len(static_measurements))
    for i, measurement in enumerate(static_measurements):
        residuals[i] = ((magnitude_mag_local)-np.linalg.norm(sensor_error_model_mag_transformation(parameter_vector, measurement)))**2
    return residuals

def optimize_mag_lm(dataset, static_intervals, threshold):
    opt_param = []
    max_nfev = 100000
    ftol=1e-10

    avg_measurements = []
    calibration_params = None

    for static_interval in static_intervals:
        avg_measurement = np.mean(dataset[static_interval[0]:static_interval[1]+1, :], axis=0)
        avg_measurements.append(avg_measurement)
    initial_parameter_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1,1,1 ,0, 0, 0] # 15 Params for the Matrix and the bias
    calibration_params = least_squares(mag_residuals, initial_parameter_vector, args=(avg_measurements, []), max_nfev=max_nfev, ftol=ftol)
    
    return calibration_params['x']