import os
import numpy as np
from sklearn.metrics import silhouette_score
from model.Algoritmi import Kmeans
from model.Evaluation.ElbowMethod import find_optimal_k
from model.Clustering.ImageLoader import load_dataset_from_folder
from model.Evaluation.show import show_clusters_3d
from model.Evaluation.utils import get_clusters

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"
preloaded_path_elbowpoint = "preloaded\\elbowpoint.npy"

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list) and os.path.exists(preloaded_path_elbowpoint):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)
    with open(preloaded_path_elbowpoint, "r") as f:
        k = int(f.read())
else:
    X, image_list = load_dataset_from_folder("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\dataset2\\01.mixed", 1700, "mixed")#841587
    np.save("preloaded\\X.npy", X)
    np.save("preloaded\\image_list.npy", image_list)
    # Punto di gomito
    k = find_optimal_k(X, "kmeans")
    with open("preloaded\\elbowpoint.txt", "w") as f:
        f.write(str(k))

#Eseguiamo il clustering k-means su k cluster
centroids, labels = Kmeans.kmeans(X, k)
n_clusters = len(centroids)

#calcolo i cluster
clusters = get_clusters(n_clusters, labels, image_list)

#valutiamo la silhouette del clustering ottenuto
silhouette = silhouette_score(X, labels)
print(silhouette)

#salviamo i risultati
result = []
i = 0
for lab in labels:
    result.append([str(image_list[i]), int(lab)])
    i = i+1

with open("Kmeans\\resultTraining.txt", "w") as file:
    file.write(str(silhouette))
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("Kmeans\\centroidsKmeans.npy", centroids)
np.save("Kmeans\\labelsKmeans.npy", labels)

for i in range(len(clusters)):
  print(str(i) + ": " + str(clusters[i]))

#Visualizziamo i Clusters
show_clusters_3d(X, labels)