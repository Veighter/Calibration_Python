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















"""

# TODO variation (yiel offspring)
# TODO evaluation (of offspring)
# TODO survival selection (yields new population)
# TODO stop
# TODO ouput of best individual

"""