# halbe stunde daten sammeln, entspricht bei einer sample-Frequenz von 100 Hz 180000 samplen
# for statistic reasoning - damit das Intervall nicht statistisch zu viele Daten enthaelt - ist die Grenze 225s

import numpy as np

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
    M = [] # Matrix holding [Residual, Params_acc, threshold, s_intervals]

    sample_number_init_intervall = T_init*sample_rate

    roh_init = static_detector_functional(acc_dataset[0:sample_number_init_intervall, :])
    
    sample_number_intervall = t_wait*sample_rate
    left_bound = (int)(sample_number_init_intervall-sample_number_intervall/2)
    right_bound = (int)(sample_number_init_intervall+sample_number_intervall/2)


    for k in range(1,10):
        size_acc = len(acc_dataset)
        threshold = k*roh_init**2
        static_detector_values = np.zeros(shape = (size_acc,) )
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
        M.append((static_detector_values, threshold))      
    

    return M

def static_detector_functional(acc_dataset):
    acc_x = acc_dataset[:,0]
    acc_y = acc_dataset[:,1]
    acc_z = acc_dataset[:,2]
    return np.linalg.norm([np.var(acc_x), np.var(acc_y), np.var(acc_z)])

def static_interval_detector(static_detector_values):
    static_intervals = [] # indixes of the M distinct intervalls of the window size t_wait
    prior_value = False
    left_bound = 0
    right_bound = 0

    for i in range(len(static_detector_values)):
        if static_detector_values[i] != prior_value: # detect High and Lows, so static and non static intervals
            prior_value = not(prior_value)
            left_bound = i
            i+=1
            in_static_interval = True
            number_of_static_samples = 1
            while(in_static_interval and number_of_static_samples<t_wait*sample_rate and i<len(static_detector_values)):
                if  static_detector_values[i] != prior_value:
                    prior_value = not(prior_value)
                    right_bound=i-1
                    static_intervals.append((left_bound, right_bound))
                number_of_static_samples+=1
                i+=1
    
    return static_intervals


# TODO implement the optimizer for the 12 parameters for the sensor error model with the levenberg marquard algorithnm
def optimize_lm(acc_dataset, static_intervals, t_wait):
    

    pass

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













# TODO implement fitness function (look Equaiton 9 and 10 "Robust and Easy Implementation for IMU Calibration")
def acc_fitness():
    pass

def gyro_fitness():
    pass

def mag_fitness():
    pass

