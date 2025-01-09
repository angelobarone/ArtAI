import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from model.Features.CNN import extract_features_CNNauto

def get_similar_images(image_path, alg):
    results = None
    if alg == "kmeans":
        centroids = np.load("..\\Clustering\\Kmeans\\centroidsKmeans.npy")
        labels = np.load("..\\Clustering\\Kmeans\\labelsKmeans.npy")
        results = np.load("..\\Clustering\\Kmeans\\resultsKmeans.npy")
        print("dati kmeans caricati")
    elif alg == "bottomup":
        centroids = np.load("..\\Clustering\\BottomUp\\centroids.npy")
        labels = np.load("..\\Clustering\\BottomUp\\labelsBottomUp.npy")
        results = np.load("..\\Clustering\\BottomUp\\resultsBottomUp.npy")
        print("dati bottomup caricati")
    else:
        centroids = np.load("..\\Clustering\\DBSCAN\\centroidsDBSCAN.npy")
        labels = np.load("..\\Clustering\\DBSCAN\\labelsDBSCAN.npy")
        results = np.load("..\\Clustering\\DBSCAN\\resultsDBSCAN.npy")
        print("dati dbscan caricati")

    images_names = np.load("..\\Clustering\\preloaded\\image_list.npy")
    X = np.load("..\\Clustering\\preloaded\\X.npy")

    new_features = extract_features_CNNauto(image_path)

    plt.bar(range(len(new_features)), new_features)
    plt.title('Bar Plot delle Feature')
    plt.xlabel('Indice')
    plt.ylabel('Valore della Feature')
    plt.show()

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
    for i in range(np.size(labels)):
        if labels[i] == centroide:
            cluster.append([images_names[i], np.abs(X[i] - new_features)])

    if len(cluster) < 5:
        if len(cluster) < 4:
            if len(cluster) < 3:
                if len(cluster) < 2:
                    h = 1
                else: h = 2
            else: h = 3
        else: h = 4
    else: h = 5

    #Estrazione delle immagini più simili all'input
    sorted_cluster = sorted(cluster, key=lambda c: np.linalg.norm(c[1]))
    similar_images = [row[0] for row in sorted_cluster[:h]]
    #Nel caso di estrazione casuale
    #similar_images = np.random.choice(cluster, size=h, replace=False)
    return similar_images, results

