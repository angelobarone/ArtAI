import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from model.Clustering.ImageLoader import load_dataset_from_folder
from mpl_toolkits.mplot3d import Axes3D

from model.Evaluation.show import show_dendrogram, show_clusters_3d
from model.Evaluation.utils import get_centroids, get_clusters

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

#visualizziamo il dendrogramma
Z = linkage(X, method='ward')
show_dendrogram(Z)

#calcolo della distanza di taglio
distances = Z[:, 2]
diffs = np.diff(distances)
max_diff_index = np.argmax(diffs)
threshold = distances[max_diff_index]

#clustering agglomerativo con sklearn
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
clustering.fit(X)
labels = clustering.labels_
n_clusters = clustering.n_clusters_

#generazione dei clusters
clusters = get_clusters(n_clusters, labels, image_list)

#calcolo dei centroidi
centroids = get_centroids(n_clusters, clusters, image_list, X)

#valutiamo la silhouette del clustering ottenuto
silhouette = silhouette_score(X, labels)
print(silhouette)

#Visualizziamo i Clusters in 3d
show_clusters_3d(X, labels)

with open("BottomUp\\resultTraining.txt", "w") as file:
    file.write(str(silhouette) + "\n")
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("BottomUp\\centroidsBottomUp.npy", centroids)
np.save("BottomUp\\labelsBottomUp.npy", labels)

for i in range(len(clusters)):
    print(str(i) + ": " + str(clusters[i]))

