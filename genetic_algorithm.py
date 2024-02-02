
"""_summary_
Die Python Datei ist ein genetischer Algorithmus zur direkten Suche des globalen Maximums
der Kostenfunktion (Euklid-Norm) für die Kalibrierung einer IMU
"""
import data_plotting as dp

import numpy as np
import math
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec

ORDER = 32

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
    # Low and Highfilter of the Accelerometer
    lowpass_filter_acc = sig.butter(ORDER, 15.0, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_acc = sig.butter(ORDER, 45.0, btype="highpass", output="sos", fs = 100.0)

    filtered_acc_x = sig.sosfilt(highpass_filter_acc, dataset[:,1]) # ACC_X

    # Plot Raw Measurement and Original Data
    fig = plot.figure(tight_layout=True)
    gs = gridspec.GridSpec(2,1)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(dataset[:,0], dataset[:,1])
    ax = fig.add_subplot(gs[1,0])
    ax.plot(dataset[:,0], filtered_acc_x)
    plot.show()

    filtered_acc_x = rectification(filtered_acc_x)

    # fig, ax = plot.subplots()
    # ax.plot(dataset[:,0], filtered_acc_x)
    # plot.show()

    filtered_acc_x = sig.sosfilt(lowpass_filter_acc,filtered_acc_x)

    # fig, ax = plot.subplots()
    # ax.plot(dataset[:,0], filtered_acc_x)
    # plot.show()


    filtered_acc_y = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, dataset[:,2]))) # ACC_Y
    filtered_acc_z = sig.sosfilt(lowpass_filter_acc,rectification(sig.sosfilt(highpass_filter_acc, dataset[:,3]))) # ACC_Z

    filtered_acc = np.array([(filtered_acc_x), (filtered_acc_y), (filtered_acc_z)])

    quasi_static_acc_coefficient = []

    

    for acc in filtered_acc.T:
        quasi_static_acc_coefficient.append(np.linalg.norm(acc))

    print(max(quasi_static_acc_coefficient))

    # Magnetometer quasi-static detector
    lowpass_filter_mag = sig.butter(ORDER, 15.0, btype="lowpass", output="sos", fs=100.0)
    highpass_filter_acc = sig.butter(ORDER, 45.0, btype="highpass", output="sos", fs=100.0)

    filtered_mag_x = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_acc, dataset[:, 7]))) # MAG_X
    filtered_mag_y = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_acc, dataset[:, 8]))) # MAG_Y
    filtered_mag_z = sig.sosfilt(lowpass_filter_mag, rectification(sig.sosfilt(highpass_filter_acc, dataset[:, 9]))) # MAG_Z

    filtered_mag = np.array([(filtered_mag_x), (filtered_mag_y), (filtered_mag_z)])

    quasi_static_mag_coefficient = []

    for mag in filtered_mag.T:
        quasi_static_mag_coefficient.append(np.linalg.norm(mag))

    
    # Gyroscope quasi-static detector
    lowpass_filter_gyro = sig.butter(ORDER, 10.0, btype="lowpass", output="sos", fs=100.0)

    filtered_gyro_x = sig.sosfilt(lowpass_filter_gyro, rectification(dataset[:,4])) # GYRO_X
    filtered_gyro_y = sig.sosfilt(lowpass_filter_gyro, rectification(dataset[:,5])) # GYRO_Y
    filtered_gyro_z = sig.sosfilt(lowpass_filter_gyro, rectification(dataset[:,6])) # GYRO_Z

    filtered_gyro = np.array([(filtered_gyro_x), (filtered_gyro_y), (filtered_gyro_z)])

    quasi_static_gyro_coefficient = []

    for gyro in filtered_gyro.T:
        quasi_static_gyro_coefficient.append(np.linalg.norm(gyro))

    quasi_static_coefficients = np.array([(quasi_static_acc_coefficient), (quasi_static_mag_coefficient), (quasi_static_gyro_coefficient)])

    
    # min = np.max(quasi_static_acc_coefficient)
    # print(f"Quasi-Static-Detectors: \n{min}")

    static_coefficient = []
    for coefficient in quasi_static_coefficients.T:

        #print(f"Coefficient sum: {coefficient[0]+coefficient[1]+coefficient[2]}")
        static_coefficient.append(1./(1.+coefficient[0]+coefficient[1]+coefficient[2]))

    print(max(static_coefficient))
    static_coefficient.sort()
    print(static_coefficient)

    # fig, ax = plot.subplots()
    # ax.plot(dataset[:,0]/10e2, static_coefficient)
    # plot.show()

    # np.extract(condition, data) fuer spaeter

    return 


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


# TODO initialize population
# TODO evalutation
# TODO parent selection
# TODO variation (yiel offspring)
# TODO evaluation (of offspring)
# TODO survival selection (yields new population)
# TODO stop
# TODO ouput of best individual

"""


def main ():
    raw_measurements = get_measurements('../../Datalogs/IMU_0.txt') # Format of Raw Measurements is that as in the datalogs

    #dp.plot_measurements_out_of_data(raw_measurements)

    determine_static_coefficients(raw_measurements)


if __name__ == "__main__":
    main()




