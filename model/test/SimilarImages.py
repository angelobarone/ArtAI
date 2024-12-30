import numpy as np
from model.Features.CNN import extract_features_CNNauto

def get_similar_images(image_path, alg):
    if alg == "kmeans":
        centroids = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Clustering\\Kmeans\\centroidsKmeans.npy")
        labels = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Clustering\\Kmeans\\labelsKmeans.npy")
    else:
        centroids = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Clustering\\BottomUp\\centroidsBottomUp.npy")
        labels = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Clustering\\BottomUp\\labelsBottomUp.npy")

    images_names = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Clustering\\preloaded\\image_list.npy")

    new_features = extract_features_CNNauto(image_path)
    c = 0
    centroide = 0
    min_dist = np.sqrt(np.sum((new_features - centroids[c]) ** 2))
    for val in centroids:
        distance = np.sqrt(np.sum((new_features - val) ** 2))
        if distance < min_dist:
            min_dist = distance
            centroide = c
        c+=1

    cluster = []
    h = 0
    for i in range(np.size(labels)):
        if labels[i] == centroide:
            cluster.append(images_names[i])

    if np.size(cluster) < 5:
        if np.size(cluster) < 4:
            if np.size(cluster) < 3:
                if np.size(cluster) < 2:
                    h = 1
                else: h = 2
            else: h = 3
        else: h = 4
    else: h = 5

    similar_images = np.random.choice(cluster, size=h, replace=False)
    return similar_images

