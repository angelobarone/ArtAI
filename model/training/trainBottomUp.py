import os
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from model.training.creazioneMatrice import load_dataset_from_folder
from scipy.cluster.hierarchy import linkage, dendrogram

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)
else:
    X, image_list = load_dataset_from_folder("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\MET", 1700)#841587
    np.save("preloaded\\X.npy", X)
    np.save("preloaded\\image_list.npy", image_list)
    #Standardizzazione dei dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "trained\\scaler.pkl")

#clustering bottomup selfmade
#centroids, clusters, labels = bottomup_clustering(X, 30, image_list)
#original_labels = labels[:, 1]

#clustering bottomup scikit
clustering = AgglomerativeClustering(59, linkage = "average")
clustering.fit(X)
original_labels = clustering.labels_
#generazione dei cluster
clusters = [[] for _ in range(clustering.n_clusters_)]
for i in range(clustering.n_clusters_):
    for h in range(np.size(original_labels)):
        if original_labels[h] == i:
            clusters[i].append(image_list[h])

#calcolo dei centroidi
centroids = np.zeros([clustering.n_clusters_, 100])
for h in range(clustering.n_clusters_):
    somma = np.zeros(100)
    for z in range(np.size(clusters[h])):
        image_name = clusters[h][z]
        index = np.where(image_list == image_name)[0][0]
        somma += X[index]
    centroids[h] = somma / np.size(clusters[h])

#valutiamo la silhouette del clustering ottenuto
#silhouette = calculate_silhouette_score(X, labels)
silhouette = silhouette_score(X, original_labels)
print(silhouette)

#valutiamo l'inertia
inertia = 0.00 #calculate_inertia(X, centroids, labels)
print(inertia)

with open("BottomUp\\resultTraining.txt", "w") as file:
    file.write(str(silhouette) + "\n")
    file.write(str(inertia) + "\n")
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("BottomUp\\centroidsBottomUp.npy", centroids)
np.save("BottomUp\\labelsBottomUp.npy", original_labels)


for i in range(len(clusters)):
    print(str(i) + ": " + str(clusters[i]))

#Visualizzaimo il Dendogramma
Z = linkage(X, method='average')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogramma del Clustering Agglomerativo")
plt.xlabel("Indice degli oggetti")
plt.ylabel("Distanza")
plt.show()

#Visualizziamo i Clusters
plt.figure(figsize=(8, 6))
for cluster_num in np.unique(original_labels):
    # Punti del cluster
    cluster_points = X[original_labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_num}")

plt.title("Visualizzazione dei Cluster Agglomerativi")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()