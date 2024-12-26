import numpy as np
from scipy.spatial.distance import cdist

def calculate_silhouette_score(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)

    silhouette_scores = []

    for i in range(n_samples):
        # Punti e cluster
        point = X[i]
        cluster_label = labels[i]

        # Punti dello stesso cluster (intra-cluster)
        same_cluster_points = X[labels == cluster_label]
        a = np.mean(cdist([point], same_cluster_points)[0])  # Distanza intra-cluster media

        # Distanza media al cluster più vicino (extra-cluster)
        b = np.inf
        for label in unique_labels:
            if label != cluster_label:
                other_cluster_points = X[labels == label]
                dist_to_other_cluster = np.mean(cdist([point], other_cluster_points)[0])
                b = min(b, dist_to_other_cluster)

        # Silhouette score per il punto i
        s = (b - a) / max(a, b) if max(a, b) != 0 else 0
        silhouette_scores.append(s)

    # Silhouette score medio
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
