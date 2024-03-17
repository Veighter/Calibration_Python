# halbe stunde daten sammeln, entspricht bei einer sample-Frequenz von 100 Hz 180000 samplen
# for statistic reasoning - damit das Intervall nicht statistisch zu viele Daten enthaelt - ist die Grenze 225s -> ueberworfen, siehe parse allan variance

import numpy as np
import quaternion
from scipy.optimize import least_squares, differential_evolution

magnitude_acc_local = 1000.16106 # mg
magnitude_mag_local = 49.4006 # uTesla 18.2

sample_rate = 100   # Hz
t_wait = 1          # s


def parse_allan_variance(w):
    collection_time = 3600 # time of static data collection in seconds
    cutoff_sample = collection_time*sample_rate # data we would face if we sample at the sample rate for 1h
    
    return w[0:cutoff_sample]

# [x] Allan Variance von Gyroskop Daten bestimmen -> gleich grosse Intervall wichtig
def allan_variance(w):
    # one axis allan variance computing
    time_begin = 1 # begin time in s
    time_end = 400 # end time in s
    sample_time = 1/sample_rate
    measurement_time = int(len(w)/100)


    num_elements = int((time_end - time_begin) / sample_time) + 1
    times = [time for time in np.linspace(start=time_begin,stop=time_end,num=num_elements) if measurement_time%time==0]

    allan_variance = np.zeros(shape=(len(times),))

    for i, time in enumerate(times): 
        K = int(measurement_time/float(time))
        sample_number_intervall = (int)(time*sample_rate)
        
        for k in range(K-1):
            left_bound = (int)(k*sample_number_intervall)
            right_bound = (int)((k+1)*sample_number_intervall) # goes through the equal length intervalls
            
            average_1 = np.mean(w[left_bound: right_bound])
            average_2 = np.mean(w[right_bound: right_bound+sample_number_intervall])
            
            allan_variance[i]+=(average_2-average_1)**2
        allan_variance[i] /= (2*K)

    return (times, allan_variance)

# [x] implement the static Detector
def static_detector(a, T_init):
    detector_threshold_tuples = [] # tuple holding [threshold, s_detector_values]

    sample_number_init_intervall = T_init*sample_rate

    roh_init = static_detector_functional(a[0:sample_number_init_intervall, :])
    
    sample_number_intervall = t_wait*sample_rate
    
    for k in range(1,10): # Anomalien, wenn das k>1 ist
        size_acc = len(a)
        threshold = k*roh_init**2
        static_detector_values = np.zeros(shape = (size_acc,) )
        left_bound = (int)(sample_number_init_intervall-sample_number_intervall/2)
        right_bound = (int)(sample_number_init_intervall+sample_number_intervall/2)
        for t, _ in enumerate(a[left_bound: ,:], sample_number_init_intervall):

            variance_magnitude = static_detector_functional(a[left_bound:right_bound,:])
            if variance_magnitude< threshold:
                static_detector_values[t] = 1
            if variance_magnitude >= threshold:
                static_detector_values[t] = 0
            
            left_bound += 1
            right_bound += 1

            if right_bound>=size_acc:
                break
        
        detector_threshold_tuples.append((static_detector_values, threshold))      
    

    return detector_threshold_tuples

def static_detector_functional(a):
    acc_x = a[:,0]
    acc_y = a[:,1]
    acc_z = a[:,2]
    return np.linalg.norm([np.var(acc_x), np.var(acc_y), np.var(acc_z)])

def static_interval_detector(static_detector_values):
    static_intervals = [] 
    static_samples_interval = t_wait*sample_rate
    
    i = 0
    while i<(len(static_detector_values)):
        if static_detector_values[i] != False:
            left_bound = i
            right_bound = i+static_samples_interval # intervall length has to be t_wait
            while(static_detector_values[i]==True):
                i+=1
            if (i-1) >= right_bound:
                right_bound=i-1 # wenn die Ergebnisse falsch werden, dann diesen Punkt raus machen, bzw nochmal in der Software der Italiener nachschauen (zentralisierung um t_wait noch nicht ganz verstanden)
                static_intervals.append((left_bound, right_bound))
        i+=1

    return static_intervals

# TODO implement the optimizer for the parameters for the sensor error model with the levenberg marquard algorithnm
# TODO use the same threshold, default is min cost threshold accelerometer optimum
def optimize_acc_lm(a, static_intervals_list, thresholds):
    opt_param = []
    max_nfev = 100000
    ftol=1e-10

    for static_intervals in static_intervals_list:
        avg_measurements = []
        calibration_params = None
        avg_measurements = avg_measurements_static_interval(a, static_intervals)
        initial_parameter_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] # 12 Params for the Matrix and the bias
        calibration_params = least_squares(acc_residuals, initial_parameter_vector, args=(avg_measurements, []), max_nfev=max_nfev, ftol=ftol, method='lm', verbose=1)
        opt_param.append((calibration_params['x'], calibration_params['cost']))
        
    costs = [cost for _, cost in opt_param]
    min_cost = costs.index(min(costs))
    
    return opt_param[min_cost][0], thresholds[min_cost]

def avg_measurements_static_interval(dataset, static_intervals):
    avg_meas = []
    for static_interval in static_intervals:
        meas_avg = np.mean(dataset[static_interval[0]:static_interval[1]+1, :], axis=0)
        avg_meas.append(meas_avg)
    return np.array(avg_meas)

# def calibration_algorithm(raw_measurements, sample_rate):
#     time = raw_measurements[:, 0]
#     acc_x = raw_measurements[:, 1]
#     acc_y = raw_measurements[:, 2]
#     acc_z = raw_measurements[:, 3]
#     mag_x =  raw_measurements[:, 7]
#     mag_y =  raw_measurements[:, 8]
#     mag_z =  raw_measurements[:, 9]
#     gyro_x = raw_measurements[:,4] 
#     gyro_y = raw_measurements[:, 5]
#     gyro_z = raw_measurements[:, 6]

#     t_init = allan_variance(np.array([time, gyro_x, gyro_y, gyro_z]).T, sample_rate)
#     pass

def sensor_error_model_acc_transformation(parameter_vector, static_measurement):
    Theta = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    b = np.array([parameter_vector[9], parameter_vector[10], parameter_vector[11]])
    return (Theta @ static_measurement.T) -b.T

# TODO implement fitness function (look Equaiton 9 and 10 "Robust and Easy Implementation for IMU Calibration")
def acc_residuals(parameter_vector, *args):
    static_measurements, _ = args
    residuals = np.zeros(len(static_measurements))
    for i, measurement in enumerate(static_measurements):
        residuals[i]= ((magnitude_acc_local)-np.linalg.norm(sensor_error_model_acc_transformation(parameter_vector, measurement)))
    return residuals
    

def gyro_fitness():
    pass

# nicht verwendet, da MAGNETO das Ellipsoid Fitting uebernimmt
def sensor_error_model_mag_transformation(parameter_vector, static_measurement):
    # axis_misalignment_matrix = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    # scaling_matrix = np.eye(3)
    # scaling_matrix[0,0]=parameter_vector[9]
    # scaling_matrix[1,1]=parameter_vector[10]
    # scaling_matrix[2,2]=parameter_vector[11]
    # bias = np.array([parameter_vector[12], parameter_vector[13], parameter_vector[14]])
    # inverse = np.linalg.inv(axis_misalignment_matrix)
    # return (inverse@scaling_matrix@axis_misalignment_matrix@static_measurement.T)-bias.T
    A = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    b = np.array([parameter_vector[9],parameter_vector[10], parameter_vector[11]])

    return A@static_measurement-b



def mag_residuals(parameter_vector, *args):
    static_measurements, _ = args
    residuals = np.zeros(len(static_measurements))
    for i, measurement in enumerate(static_measurements):
        transformation = sensor_error_model_mag_transformation(parameter_vector, measurement)
        transformation_norm = np.linalg.norm(transformation)
        residuals[i] = (1-transformation_norm)**2
    return np.sum(residuals)

def optimize_mag_lm(m, static_intervals):
    max_nfev = 1000000
    ftol=1e-10

    avg_measurements = []

    avg_measurements = avg_measurements_static_interval(m, static_intervals)
    initial_parameter_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1,0, 0, 0] # 12 Params for the Matrix and the bias
    calibration_params = least_squares(mag_residuals, initial_parameter_vector, args=(avg_measurements, []), max_nfev=max_nfev, ftol=ftol, verbose=1, method='lm')
    
    return calibration_params['x']

def optimize_mag_diff_ev(m, static_intervals):
    avg_measurements = []

    avg_measurements = avg_measurements_static_interval(m, static_intervals)

    bounds = [(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-1000, 1000),(-1000, 1000),(-1000, 1000)]
    result = differential_evolution(mag_residuals, args=(avg_measurements,[]), bounds=bounds,maxiter=10000)
    return result['x']



def optimze_gyro_lm(w, static_intervals, a_O, m_O, T_init,time):
    max_nfev = 1000000
    ftol=1e-12

    b_w = np.mean(w[0:T_init*sample_rate, :], axis=0)
    w_b_free = w - b_w

    a_O_avg = avg_measurements_static_interval(a_O, static_intervals)
    m_O_avg = avg_measurements_static_interval(m_O, static_intervals)

    initial_parameter_vector = [1, 0, 0, 0, 1, 0, 0, 0, 1] 
    calibration_params = least_squares(gyro_residuals, initial_parameter_vector, args=(a_O_avg, m_O_avg, w_b_free, static_intervals,time), max_nfev=max_nfev, ftol=ftol, verbose=1, method='lm')

    return np.append(calibration_params['x'], b_w.tolist())


def gyro_residuals(parameter_vector, *args):
    a_O_avg, m_O_avg, w_b_free, static_intervals,time = args

    residuals = []

    for i in range(len(static_intervals)):

        if i > 0:
            # a_t_1 = a_O_avg[i-1]
            # m_t_1 = m_O_avg[i-1]
            # a_t = a_O_avg[i]
            # m_t = m_O_avg[i]

            q_rot = np.quaternion(0,1,0,0) # Rotation about 180 degree about x-axis in local XYZ NED-Frame

            a_t_1 = quaternion.as_vector_part(q_rot*quaternion.from_vector_part(a_O_avg[i-1])*q_rot.conjugate())
            m_t_1 = m_O_avg[i-1]
            a_t = quaternion.as_vector_part(q_rot*quaternion.from_vector_part(a_O_avg[i])*q_rot.conjugate())
            m_t = m_O_avg[i]
    
            motion_interval_bound_left = static_intervals[i-1][1]
            motion_interval_bound_right = static_intervals[i][0]
        
            w_t_1_to_t = w_b_free[motion_interval_bound_left:motion_interval_bound_right, :]
            time_rotation = time[motion_interval_bound_left:motion_interval_bound_right]

            q = quaternion_integration(parameter_vector, w_t_1_to_t,time_rotation)

            a_quat = quaternion.from_vector_part(a_t_1)
            q_conj = q.conjugate()
            quat_v_a = quaternion.as_vector_part(q_conj*a_quat*q)

            # v_a = quaternion.as_vector_part(q*quaternion.from_vector_part(a_t_1)*q.conjugate())

            v_a = quat_v_a
            v_m = quaternion.as_vector_part(q.conjugate()*quaternion.from_vector_part(m_t_1)*q)

            diff_a = np.linalg.norm(v_a-a_t)
            diff_m = np.linalg.norm(v_m-m_t)
            diff_m=0

            residuals.append(diff_a+diff_m)

    return residuals

# vllt ist hier die Dimension nicht richtig!!
def sensor_error_model_gyro_transformation(parameter_vector, w):
    Theta = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    return Theta@w.T

def quaternion_integration(parameter_vector, w,time):
    w_bar=[]
    for gyro_measurement in w:
        w_bar.append(sensor_error_model_gyro_transformation(parameter_vector, gyro_measurement).T)

    w_bar = np.array(w_bar)
    q_t = np.quaternion(1,0,0,0)
    #dt = 1/samplerate

    for i in range(len(w_bar)):
        # w_x=w_bar[i,0]
        # w_y=w_bar[i,1]
        # w_z=w_bar[i,2]

        dt = time[i]-time[i-1]

        q_rot = np.quaternion(0,1,0,0) # Rotation about 180 degree about x-axis in local XYZ NED-Frame

        w_bar_NED = q_rot*quaternion.from_vector_part(w_bar[i,:])*q_rot.conjugate()

        # Watch Out Axes of Rotation are not NED (x, -y, -z) are the axes compared to ned-sensor-frame
        # w_quat = np.quaternion(0,w_x,-w_y,-w_z)
        w_quat = w_bar_NED
        # Equation (8) aus "Automatic Calibration IMU"
        q_t = q_t+(1./2)*w_quat*q_t*dt
        q_t = q_t/np.linalg.norm(quaternion.as_float_array(q_t))

    return q_t

 # axis_misalignment_matrix = np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]])
    # scaling_matrix = np.eye(3)
    # scaling_matrix[0,0]=parameter_vector[9]
    # scaling_matrix[1,1]=parameter_vector[10]
    # scaling_matrix[2,2]=parameter_vector[11]
    # bias = np.array([parameter_vector[12], parameter_vector[13], parameter_vector[14]])
    # inverse = np.linalg.inv(axis_misalignment_matrix)
    #return inverse@scaling_matrix@axis_misalignment_matrix@(static_measurement-bias).T
