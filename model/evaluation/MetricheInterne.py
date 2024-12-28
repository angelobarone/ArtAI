import numpy as np
from scipy.spatial.distance import cdist

def calculate_silhouette_score(X, labels):
    n_samples = len(X)

    silhouette_scores = []

    for i in range(n_samples):
        #Punti e cluster
        point = X[i]
        cluster_label = int(labels[i][1])

        #Punti dello stesso cluster (intra-cluster)
        same_cluster_points = []
        for h in range(X.shape[0]):
            if labels[h][1] == cluster_label :
                same_cluster_points.append(X[h])
        a = np.mean(cdist([point], same_cluster_points)[0])

        #Distanza media al cluster più vicino (extra-cluster)
        b = np.inf
        unique_clusters = np.unique(labels[:,1])

        for h in range(np.size(unique_clusters)):
            if unique_clusters[h] != cluster_label:
                same_cluster_points = []
                for z in range(X.shape[0]):
                    if labels[z][1] == unique_clusters[h]:
                        same_cluster_points.append(X[z])
            a = np.mean(cdist([point], same_cluster_points)[0])
            if a < b:
                b = a

        #Silhouette score per il punto i
        s = (b - a) / max(a, b) if max(a, b) != 0 else 0
        silhouette_scores.append(s)

    #Silhouette score medio
    return np.mean(silhouette_scores)


def calculate_inertia(X, centroids, labels):
    inertia = 0.0
    for i in range(len(X)):
        # Trova il centroide assegnato
        cluster_index = labels[i]
        centroid = centroids[int(cluster_index)]
        # Calcola la distanza al quadrato
        inertia += np.sum((X[i] - centroid) ** 2)
    return inertia
