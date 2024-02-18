"""_summary_
Die Python Datei ist ein differentieller evoluionaerer(genetischer) Algorithmus zur direkten Suche des globalen Maximums
der Kostenfunktion (Euklid-Norm) für die Kalibrierung einer IMU
"""

""" 
Erklärung/ Terminologie
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
"""
# Sensortyp, Suchraum Sensor, Measurements for Cost function, Cost Function
import numpy as np
import random as rd

magnitude_acc_local = 1000.16106 # mg
magnitude_mag_locla = 49.4006 # uTesla

# [x] initialize population
def init_population(search_space, population_size):
    dimension = np.shape(search_space)[0]
    population = np.random.uniform(low=[limits[0] for limits in search_space], high=[limits[1] for limits in search_space], size=(int(population_size), dimension))
    return population

# [x] evalutation
def evaluation(parameter_vectors, quasi_static_measurements, sensor):
    cost = []
    for parameter_vector in parameter_vectors:
        new_cost = 0
        if sensor == "acc":
            for quasi_static_measurement in quasi_static_measurements:
                new_cost += acc_fitness(parameter_vector, quasi_static_measurement)
        if sensor == "gyro":
            for quasi_static_measurement in quasi_static_measurements:
                new_cost += gyro_fitness()
        if sensor == "mag":
            for quasi_static_measurement in quasi_static_measurements:
                new_cost += mag_fitness() # norm by magnetic field at my position in same unit 

        cost.append(new_cost)

    min_cost = min(cost)
    index_fittest_vector = cost.index(min_cost)
    return min_cost, index_fittest_vector

# [x] parent selection / evolution
def evolution(parameter_vectors, dimension, quasi_static_measurements, sensor, crossover_probability, differential_weight ):
    new_population = []
    for parameter_vector in parameter_vectors:
        picked_parents = rd.sample([_ for _ in parameter_vectors if not np.array_equal(_, parameter_vector)], 3)
        picked_parents = np.array(picked_parents)
        # print(picked_parents)
        random_index = rd.randint(0,dimension-1)
        new_individuum = np.zeros(dimension)
        for i in range(dimension):
            if rd.uniform(0,1)<crossover_probability or i == random_index:
                new_individuum[i] = picked_parents[0][i]+differential_weight*(picked_parents[1][i]-picked_parents[2][i])
            else:
                new_individuum[i] = parameter_vector[i] 
        if evaluation([new_individuum], quasi_static_measurements, sensor)<=evaluation([parameter_vector], quasi_static_measurements, sensor):
            new_population.append(new_individuum)
        else:
            new_population.append(parameter_vector)
    return np.array(new_population)

def acc_fitness(parameter_vector, quasi_static_measurement):
    return ((magnitude_acc_local)-np.linalg.norm(np.array([parameter_vector[0:3], parameter_vector[3:6], parameter_vector[6:9]]) @ quasi_static_measurement.T-np.array([parameter_vector[9], parameter_vector[10], parameter_vector[11]])))**2

def gyro_fitness():
    pass

def mag_fitness():
    pass

def algorithm(quasi_static_measurements, sensor, search_space, threshold, population_size, crossover_probability, differential_weight):
    population = init_population(search_space, population_size)

    generation = 0
    costs, _ = evaluation(population, quasi_static_measurements, sensor)
    while  costs >= threshold and generation <= 1000: # adjust costs due to the unit searching for?? Or one general residual error
        population =  evolution(population, np.shape(search_space)[0], quasi_static_measurements, sensor, crossover_probability, differential_weight)
        costs, index_fittest_vector = evaluation(population, quasi_static_measurements, sensor)
        generation+=1

    return population[index_fittest_vector]
















"""

# TODO variation (yiel offspring)
# TODO evaluation (of offspring)
# TODO survival selection (yields new population)
# TODO stop
# TODO ouput of best individual

"""