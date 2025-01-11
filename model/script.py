import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

from model.Evaluation.ElbowMethod import find_optimal_k

data = np.load("Clustering\\preloaded\\X.npy")

#neighbors = NearestNeighbors(n_neighbors=5)
#neighbors_fit = neighbors.fit(data)
#distances, indices = neighbors.kneighbors(data)

# Ordina le distanze per trovare l'angolo del grafico
#distances = np.sort(distances[:, 4], axis=0)
#plt.plot(distances)
#plt.xlabel("Punti ordinati")
#plt.ylabel("Distanza")
#plt.title("k-distance plot per scegliere eps")
#plt.show()

k = find_optimal_k(data, 150, 300)