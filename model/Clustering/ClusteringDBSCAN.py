import os

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from model.Clustering.ImageLoader import load_dataset_from_folder
from model.Evaluation.show import show_clusters_3d
from model.Evaluation.utils import get_clusters, get_centroids

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)

else:
    X, image_list = load_dataset_from_folder("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\dataset2\\01.mixed",13967 , "mixed")
    np.save("preloaded\\X.npy", X)
    np.save("preloaded\\image_list.npy", image_list)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

eps = 0.2       #Distanza massima tra due punti per essere considerati vicini
min_samples = 5 #Minimo numero di punti per formare un cluster

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters)

#generazione dei clusters
clusters = get_clusters(n_clusters, labels, image_list)

#calcolo dei centroidi
centroids = get_centroids(n_clusters, clusters, image_list, X)

#valutiamo la silhouette del clustering ottenuto
silhouette = silhouette_score(X, labels)
print(silhouette)

#Visualizziamo i Clusters in 3d
show_clusters_3d(X, labels)

with open("DBSCAN\\resultTraining.txt", "w") as file:
    file.write(str(silhouette) + "\n")
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("DBSCAN\\centroidsBottomUp.npy", centroids)
np.save("DBSCAN\\labelsBottomUp.npy", labels)

for i in range(len(clusters)):
    print(str(i) + ": " + str(clusters[i]))