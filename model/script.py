import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

data = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Clustering\\preloaded\\X.npy")

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(data)
distances, indices = neighbors.kneighbors(data)

# Ordina le distanze per trovare l'angolo del grafico
distances = np.sort(distances[:, 4], axis=0)
plt.plot(distances)
plt.xlabel("Punti ordinati")
plt.ylabel("Distanza")
plt.title("k-distance plot per scegliere eps")
plt.show()
