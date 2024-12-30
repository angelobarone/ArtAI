import os
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from model.Algoritmi import Kmeans
from model.Evaluation.ElbowMethod import find_optimal_k
from model.Evaluation.MetricheInterne import calculate_inertia
from scipy.cluster.hierarchy import linkage, dendrogram
from model.Clustering.ImageLoader import load_dataset_from_folder

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"
preloaded_path_elbowpoint = "preloaded\\elbowpoint.npy"

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list) and os.path.exists(preloaded_path_elbowpoint):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)
    k = np.load(preloaded_path_elbowpoint)
else:
    X, image_list = load_dataset_from_folder("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\dataset2\\01.mixed", 1700, "mixed")#841587
    np.save("preloaded\\X.npy", X)
    np.save("preloaded\\image_list.npy", image_list)
    # Punto di gomito
    k = find_optimal_k(X, "kmeans")
    np.save("preloaded\\elbowpoint.npy", k)

#Eseguiamo il clusteing k-means su k cluster
centroids, labels = Kmeans.kmeans(X, k)

#valutiamo la silhouette del clustering ottenuto
silhouette = silhouette_score(X, labels)
print(silhouette)


#salviamo i risultati
result = []
i = 0
for lab in labels:
    result.append([str(image_list[i]), int(lab)])
    i = i+1

#calcolo i cluster
clusters = [[] for _ in range(len(centroids))]
for item in result:
    file_name, cluster_index = item
    clusters[cluster_index].append(file_name)

with open("Kmeans\\resultTraining.txt", "w") as file:
    file.write(str(silhouette))
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("Kmeans\\centroidsKmeans.npy", centroids)
np.save("Kmeans\\labelsKmeans.npy", labels)

for i in range(len(clusters)):
  print(str(i) + ": " + str(clusters[i]))

#Visualizziamo i Clusters
plt.figure(figsize=(8, 6))
for cluster_num in np.unique(labels):
    # Punti del cluster
    cluster_points = X[labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_num}")

plt.title("Visualizzazione dei Cluster Agglomerativi")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()