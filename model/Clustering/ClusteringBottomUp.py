import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from model.Clustering.ImageLoader import load_dataset_from_folder
from model.Evaluation.ElbowMethod import find_optimal_k

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"
k = 0

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)
    k = find_optimal_k(X) #with open('k.pkl', 'rb') as file: k = pickle.load(file)
else:
    X, image_list = load_dataset_from_folder("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\dataset2\\01.mixed",13967 , "mixed")
    np.save("preloaded\\X.npy", X)
    np.save("preloaded\\image_list.npy", image_list)
    # Punto di gomito
    k = find_optimal_k(X)
    with open('k.pkl', 'wb') as file:
        pickle.dump(k, file)

#clustering bottomup
clustering = AgglomerativeClustering(k, linkage = "complete")
clustering.fit(X)
original_labels = clustering.labels_
#generazione dei cluster
clusters = [[] for _ in range(clustering.n_clusters_)]
for i in range(clustering.n_clusters_):
    for h in range(np.size(original_labels)):
        if original_labels[h] == i:
            clusters[i].append(image_list[h])

#calcolo dei centroidi
centroids = np.zeros([clustering.n_clusters_, X.shape[1]])
for h in range(clustering.n_clusters_):
    somma = np.zeros(X.shape[1])
    for z in range(np.size(clusters[h])):
        image_name = clusters[h][z]
        index = np.where(image_list == image_name)[0]
        somma += X[index].reshape(-1)
    centroids[h] = somma / np.size(clusters[h])

#valutiamo la silhouette del clustering ottenuto
#silhouette = calculate_silhouette_score(X, labels)
silhouette = silhouette_score(X, original_labels)
print(silhouette)

with open("BottomUp\\resultTraining.txt", "w") as file:
    file.write(str(silhouette) + "\n")
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("BottomUp\\centroidsBottomUp.npy", centroids)
np.save("BottomUp\\labelsBottomUp.npy", original_labels)

for i in range(len(clusters)):
    print(str(i) + ": " + str(clusters[i]))

#Visualizziamo i Clusters
plt.figure(figsize=(10, 10))
for cluster_num in np.unique(original_labels):
    # Punti del cluster
    cluster_points = X[original_labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_num}")

plt.title("Visualizzazione dei Cluster Agglomerativi")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()