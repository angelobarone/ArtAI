import os
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model.Clustering.ImageLoader import load_dataset_from_folder
from model.Evaluation.Silhouette import find_optimal_silhouette
from model.Evaluation.show import show_dendrogram, show_clusters_3d
from model.Evaluation.utils import get_centroids, get_clusters

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"
preloaded_path_dendrogram = "BottomUp\\dendrogram.npy"
preloaded_path_silhouette = "BottomUp\\optimal_k_on_silhouette.txt"

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list) and os.path.exists(preloaded_path_dendrogram) and os.path.exists(preloaded_path_silhouette):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)
    Z = np.load(preloaded_path_dendrogram)
    with open(preloaded_path_silhouette, "r") as f:
        k = int(f.read())
else:
    X, image_list = load_dataset_from_folder("..\\..\\dataset\\01.mixed")
    Z = linkage(X, method='ward')
    k = find_optimal_silhouette(X, 190, 300)
    np.save(preloaded_path_X, X)
    np.save(preloaded_path_image_list, image_list)
    np.save(preloaded_path_dendrogram, Z)
    with open(preloaded_path_silhouette, "w") as f:
        f.write(str(k))

#visualizziamo il dendrogramma
show_dendrogram(Z)

#clustering agglomerativo con sklearn
clustering = AgglomerativeClustering(n_clusters = k, linkage = "ward", metric="euclidean", memory="..\\tmp\\cache")
clustering.fit(X)
labels = clustering.labels_
n_clusters = clustering.n_clusters_
print(n_clusters)

#generazione dei clusters
clusters = get_clusters(n_clusters, labels, image_list)

#calcolo dei centroidi
centroids = get_centroids(n_clusters, clusters, image_list, X)

#valutiamo quanto gli elementi siano nel cluster corretto con la silhouette
silhouette = silhouette_score(X, labels)
print(silhouette)

#valutiamo la compattezza e separabilità dei cluster
dbi = davies_bouldin_score(X, labels)
print(dbi)

results = [silhouette, dbi]

#Visualizziamo i Clusters in 3d
show_clusters_3d(X, labels)


with open("BottomUp\\clusters.txt", "w") as file:
    for i in range(len(clusters)):
        file.write(str(i) + ": " + str(clusters[i]))

np.save("BottomUp\\centroidsBottomUp.npy", centroids)
np.save("BottomUp\\labelsBottomUp.npy", labels)
np.save("BottomUp\\resultsBottomUp.npy", results)

for i in range(len(clusters)):
    print(str(i) + ": " + str(clusters[i]))

