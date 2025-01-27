import threading
import time
import csv
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from sklearn.decomposition import PCA

from model.Clustering.UpdateClustering import update_dataset
from model.Features.CNN import extract_features_CNNauto

def get_similar_images(image_path, alg, n):
    results = None
    start = time.time()
    if alg == "kmeans":
        centroids = np.load("..\\Clustering\\Kmeans\\centroidsKmeans.npy")
        labels = np.load("..\\Clustering\\Kmeans\\labelsKmeans.npy")
        clustering_quality = np.load("..\\Clustering\\Kmeans\\resultsKmeans.npy")
        print("dati kmeans caricati")
    elif alg == "bottomup":
        centroids = np.load("..\\Clustering\\BottomUp\\centroidsBottomUp.npy")
        labels = np.load("..\\Clustering\\BottomUp\\labelsBottomUp.npy")
        clustering_quality = np.load("..\\Clustering\\BottomUp\\resultsBottomUp.npy")
        print("dati bottomup caricati")
    elif alg == "dbscan":
        centroids = np.load("..\\Clustering\\DBSCAN\\centroidsDBSCAN.npy")
        labels = np.load("..\\Clustering\\DBSCAN\\labelsDBSCAN.npy")
        clustering_quality = np.load("..\\Clustering\\DBSCAN\\resultsDBSCAN.npy")
        print("dati dbscan caricati")
    else:
        raise ValueError("alg non valida, i valori consentiti sono: kmeans, bottomup, dbscan")

    images_names = np.load("..\\Clustering\\preloaded\\image_list.npy")
    X = np.load("..\\Clustering\\preloaded\\X.npy")

    new_features = extract_features_CNNauto(image_path)

    #plt.bar(range(len(new_features)), new_features)
    #plt.title('Bar Plot delle Feature')
    #plt.xlabel('Indice')
    #plt.ylabel('Valore della Feature')
    #plt.show()

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

    if len(cluster) < n:
        h = len(cluster)
    else:
        h = n

    #Estrazione delle immagini più simili all'input
    sorted_cluster = sorted(cluster, key=lambda c: np.linalg.norm(c[1]))
    similar_images = [row[0] for row in sorted_cluster[:h]]
    end = time.time()
    research_time = end - start

    #Predizione del tipo di opera d'arte
    type_arts = []
    with open("../../dataset/WikiArt.csv", mode = "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for img in similar_images:
                if img in row[0]:
                    type_arts.append(row[1])
    f.close()
    contatore = Counter(type_arts)
    c = contatore.most_common(1)[0]
    tipo, occorrenze = c
    quality = occorrenze * 100/len(similar_images)

    results = [clustering_quality[0], clustering_quality[1], research_time, tipo, quality]

    #Aggiornamento del dataset
    #thread = threading.Thread(target=update_dataset(X, new_features, results[3], image_path, "../../dataset/WikiArt.csv"))
    #thread.start()

    #Nel caso di estrazione casuale
    #similar_images = np.random.choice(cluster, size=h, replace=False)
    return similar_images, results

