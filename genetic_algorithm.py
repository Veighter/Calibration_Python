
"""_summary_
Die Python Datei ist ein differentieller evoluionaerer(genetischer) Algorithmus zur direkten Suche des globalen Maximums
der Kostenfunktion (Euklid-Norm) für die Kalibrierung einer IMU
"""
import data_plotting as dp
import random as rd 
import numpy as np
import math
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec

ORDER = 5
POPULATION_SIZE = 10e0 # typical size for differential evolution is 10*(number of inputs)
SEARCH_SPACE_DEFAULT = [(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2)]
SEARCH_SPACE_MAG = [(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5), (-100, 100),(-100,100), (-100,100)]
CROSSOVER_PROBABILITY = 0.9
DIFFERENTIAL_WEIGHT = 0.8 # inital values guessed by wikipedia



def rectification(dataset):
    rectification = []
    for data in dataset:
        rectification.append(abs(data))

    return rectification

def relu(dataset):
    relu = []

    for data in dataset:
        relu.append(max(0.0,data))

    return relu


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
    mag_x =  dataset[:,7]
    mag_y =  dataset[:, 8]
    mag_z =  dataset[:, 9]
    gyro_x = dataset[:,4]
    gyro_y = dataset[:, 5]
    gyro_z = dataset[:, 6]


    # Low and Highfilter of the Accelerometer
    lowpass_filter_acc = sig.butter(ORDER, 5, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_acc = sig.butter(ORDER, 49, btype="highpass", output="sos", fs = 100.0)
    filtered_acc_x = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, acc_x))) # ACC_X
    filtered_acc_y = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, acc_y))) # ACC_Y
    filtered_acc_z = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, acc_z))) # ACC_Z

    filtered_acc = np.array([(filtered_acc_x), (filtered_acc_y), (filtered_acc_z)])

    quasi_static_acc_coefficient = []

    
    for acc in filtered_acc.T:
        quasi_static_acc_coefficient.append(np.linalg.norm(acc))


    # Magnetometer quasi-static detector
    lowpass_filter_mag = sig.butter(ORDER, 1, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_acc = sig.butter(ORDER, 20, btype="highpass", output="sos", fs=100.0)

    filtered_mag_x = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_acc, mag_x))) # MAG_X
    filtered_mag_y = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_acc, mag_y))) # MAG_Y
    filtered_mag_z = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_acc, mag_z))) # MAG_Z

    filtered_mag = np.array([(filtered_mag_x), (filtered_mag_y), (filtered_mag_z)])

    quasi_static_mag_coefficient = []

    for mag in filtered_mag.T:
        quasi_static_mag_coefficient.append(np.linalg.norm(mag))

    
    # Gyroscope quasi-static detector
    lowpass_filter_gyro = sig.butter(ORDER, 1, btype="lowpass", output="sos", fs=100.0)

    filtered_gyro_x = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_x)) # GYRO_X
    filtered_gyro_y = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_y)) # GYRO_Y
    filtered_gyro_z = sig.sosfilt(lowpass_filter_gyro, rectification(gyro_z)) # GYRO_Z

    filtered_gyro = np.array([(filtered_gyro_x), (filtered_gyro_y), (filtered_gyro_z)])

    quasi_static_gyro_coefficient = []

    for gyro in filtered_gyro.T:
        quasi_static_gyro_coefficient.append(np.linalg.norm(gyro))

    quasi_static_coefficients = np.array([(quasi_static_acc_coefficient), (quasi_static_mag_coefficient), (quasi_static_gyro_coefficient)])


    static_coefficients = []
    for coefficient in quasi_static_coefficients.T:
        static_coefficients.append(1./(1.+coefficient[0]))

    static_coefficients = np.array(static_coefficients)


    # fig, [ax1, ax2] = plot.subplots(2,1)
    # ax1.plot(time, acc_x)
    # ax1.plot(time, acc_y)
    # ax1.plot(time, acc_z)
    # ax1.set_xlim(time[0], time[-1])
    # ax1.set_ylabel('acc [mG]')
    # ax1.set_xlabel('t [s]')
    # ax1.set_title('Raw Accelerometer Measurements')
    # ax2.plot(time, static_coefficients)
    # ax2.set_xlim(time[0], time[-1])
    # ax2.set_ylabel('q-s-Coeff')
    # ax2.set_xlabel('t [s]')
    # ax2.set_title('Quasi-static-Coefficient')
    # plot.show()


    # np.extract(condition, data) fuer spaeter

    return static_coefficients


""" Erklärung/ Terminologie
1. Individuen sind die Grundform eines gentischen Algorithmus 
    - jedes Individuum welches existiert, existiert in einer Generation, einem Zeitschritt
    - Individuen einer Generation bilden die Population
2. Natuerliche Selektion
    - die besten Individueen setzen sich durch und geben ihre Gene weiter
    - Fitnessfunktion, meist Zielfunktion, ist das Auswahlkriterium
3. Nachwuchs / Nächste Generation
    - Die Erzeugung einer nächsten Generation erzeugt durch Nachwuchs aus den Zusammenkommen der vorherigen
    - verschiedene Paarungsmöglichkeiten für Gene (Cross-Over, Zahlendreher, Ausschneiden, weitere in Vorlesung "Computational Intelligence")
4. Mutation in der Generation bringt Vielfalt und Diversität
    - meist zufällig
5. Solange Selektion bis Abbruchkriterium (durch Fintessfunktion) erreicht ist








# TODO variation (yiel offspring)
# TODO evaluation (of offspring)
# TODO survival selection (yields new population)
# TODO stop
# TODO ouput of best individual

"""
# TODO parent selection / evolution
def evolution(parameter_vectors, dimension, quasi_static_measurements, sensor):
    new_population = []
    for parameter_vector in parameter_vectors:
        picked_parents = rd.sample([_ for _ in parameter_vectors if not np.array_equal(_, parameter_vector)], 3)
        picked_parents = np.array(picked_parents)
        # print(picked_parents)
        random_index = rd.randint(0,dimension-1)
        new_individuum = np.zeros(dimension)
        for i in range(dimension):
            if rd.uniform(0,1)<CROSSOVER_PROBABILITY or i == random_index:
                new_individuum[i] = picked_parents[0][i]+DIFFERENTIAL_WEIGHT*(picked_parents[1][i]-picked_parents[2][i])
            else:
                new_individuum[i] = parameter_vector[i] 
        if evaluation([new_individuum], quasi_static_measurements, sensor)<=evaluation([parameter_vector], quasi_static_measurements, sensor):
            new_population.append(new_individuum)
        else:
            new_population.append(parameter_vector)
    return np.array(new_population)




def calibrate_sensor(quasi_static_measurements, sensor):
    population = np.array([])
    search_space = None
    if sensor == "acc" or sensor == "gyro":
        search_space = SEARCH_SPACE_DEFAULT
        population = initialize_population(search_space)

    if sensor == "mag":
        population = initialize_population(SEARCH_SPACE_MAG)


    generation = 0
    costs, index_fittest_vector = evaluation(population, quasi_static_measurements, sensor)
    while  costs >= 1000 and generation <= 1000:
        population =  evolution(population, np.shape(search_space)[0], quasi_static_measurements, sensor)
        costs, index_fittest_vector = evaluation(population, quasi_static_measurements, sensor)
        generation+=1

    return population[index_fittest_vector]

    
# [x] initialize population
def initialize_population(search_space):
    dimension = np.shape(search_space)[0]
    population = np.random.uniform(low=[limits[0] for limits in search_space], high=[limits[1] for limits in search_space], size=(int(POPULATION_SIZE), dimension))
    return population

# [x] evalutation
def evaluation(parameter_vectors, quasi_static_measurements, sensor):
    cost = []
    for parameter_vector in parameter_vectors:
        new_cost = 0
        if sensor == "acc":
            for measurement in quasi_static_measurements:
                new_cost += (1-np.linalg.norm(np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]]) @ measurement.T-np.array([parameter_vector[9], parameter_vector[10], parameter_vector[11]])))**2
        cost.append(new_cost)

        min_cost = min(cost)
        index_fittest_vector = cost.index(min_cost)
    return min_cost, index_fittest_vector


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

    #dp.plot_measurements_out_of_data(raw_measurements)

    quasi_static_coefficients = determine_static_coefficients(raw_measurements)

    indixes = quasi_static_coefficients > 0.98

    quasi_static_measurements = np.array([raw_measurements[i,:] for i in range(len(raw_measurements)) if indixes[i]])

    calibration_parameters = calibrate_sensor(quasi_static_measurements=quasi_static_measurements[:, 1:4], sensor='acc')
    
    calibrated_measurements = get_calibrated_measurement(raw_measurements[:, 1:4], calibration_parameters, sensor="acc")

    time = raw_measurements[:,0]
    acc_x = raw_measurements[:,1]
    acc_y = raw_measurements[:,2]
    acc_z = raw_measurements[:,3]


    calibrated_acc_x = calibrated_measurements[:,0]
    calibrated_acc_y = calibrated_measurements[:,1]
    calibrated_acc_z = calibrated_measurements[:,2]

    fig, [ax1, ax2] = plot.subplots(2,1)
    ax1.plot(time, acc_x)
    ax1.plot(time, acc_y)
    ax1.plot(time, acc_z)
    ax2.plot(time, calibrated_acc_x)
    ax2.plot(time, calibrated_acc_y)
    ax2.plot(time, calibrated_acc_z)
    plot.show()








if __name__ == "__main__":
    main()
