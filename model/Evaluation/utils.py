import numpy as np


def get_clusters(n_clusters, labels, image_list):
    clusters = [[] for _ in range(n_clusters)]
    for i in range(n_clusters):
        for h in range(np.size(labels)):
            if labels[h] == i:
                clusters[i].append(image_list[h])

    return clusters

def get_centroids(n_clusters, clusters, image_list, X):
    centroids = np.zeros([n_clusters, X.shape[1]])
    for h in range(n_clusters):
        somma = np.zeros(X.shape[1])
        for z in range(np.size(clusters[h])):
            image_name = clusters[h][z]
            index = np.where(image_list == image_name)[0]
            somma += X[index].reshape(-1)
        centroids[h] = somma / np.size(clusters[h])

    return centroids