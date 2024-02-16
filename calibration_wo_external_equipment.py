# halbe stunde daten sammeln, entspricht bei einer sample-Frequenz von 100 Hz 180000 samplen
# for statistic reasoning - damit das Intervall nicht statistisch zu viele Daten enthaelt - ist die Grenze 225s

import numpy as np


def allan_variance(gyro_measurements, sample_rate):
    # one axis allan variance computing
    time_begin = 1 # begin time in s
    time_end = 225 # end time in s
    sample_time = 1/sample_rate

    num_elements = int((time_end - time_begin) / sample_time) + 1

    times = [[time for time in np.linspace(start=time_begin,stop=time_end,num=num_elements)]]

    measurement_time = int(len(gyro_measurements)/1000)

    for time in times: # evenly space 0.1 Zeitsteps
        allan_variance = np.zeros(shape=(1,len(times)))
        K = int(measurement_time/time)
        sample_number_intervall = time*sample_rate
        for k in len(K):
            left_bound = k*sample_number_intervall
            right_bound = left_bound+sample_number_intervall # goes through the equal length intervalls
            allan_variance+=(np.sum(gyro_measurements[right_bound+1, right_bound+1+sample_number_intervall])/sample_number_intervall-np.sum(gyro_measurements[left_bound, left_bound+sample_number_intervall])/sample_number_intervall)**2
        allan_variance= allan_variance/(2*K)

    return (times, allan_variance)

def static_detector():
    #return 1 if chi<threshold else 0
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

    t_init = allan_variance([time, gyro_x, gyro_y, gyro_z].T, sample_rate)
    pass














def acc_fitness():
    pass

def gyro_fitness():
    pass

def mag_fitness():
    pass

