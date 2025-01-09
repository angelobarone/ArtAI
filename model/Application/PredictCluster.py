import numpy as np

def predict_cluster(new_image_features, centroids):

    distances = []
    i = 0
    for centroid in centroids:
        distances.append([int(i), np.sqrt(np.sum((new_image_features - centroid.reshape(1, -1)) ** 2))])
        i = i+1

    min_dist = [0, float('inf')]
    for distance in distances:
        if distance[1] < min_dist[1]:
            min_dist = distance

    return min_dist[0]