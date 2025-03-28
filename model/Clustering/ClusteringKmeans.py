import os
import time

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model.Algoritmi import Kmeans
from model.Evaluation.ElbowMethod import find_optimal_k
from model.Clustering.ImageLoader import load_dataset_from_folder
from model.Evaluation.show import show_clusters_3d, show_clusters_2d, show_centroids_2d
from model.Evaluation.utils import get_clusters

def applicate_Kmeans(preloaded_path, save_folder, dataset_path):

    preloaded_path_X = preloaded_path + "/X.npy"
    preloaded_path_image_list = preloaded_path + "/image_list.npy"
    preloaded_path_elbowpoint = save_folder + "/elbowpoint.txt"
    # carica il Dataset
    if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list):
        X = np.load(preloaded_path_X)
        image_list = np.load(preloaded_path_image_list)
        with open(preloaded_path_elbowpoint, "r") as f:
            k = int(f.read())
    else:
        X, image_list = load_dataset_from_folder("..\\" + dataset_path)
        np.save(preloaded_path_X, X)
        np.save(preloaded_path_image_list, image_list)
        # Punto di gomito
        #k = find_optimal_k(X, 150, 300)
        #with open(preloaded_path_elbowpoint, "w") as f:
        #    f.write(str(k))


    #Eseguiamo il clustering k-means su k cluster
    centroids, labels = Kmeans.kmeans(X, k)
    n_clusters = len(centroids)

    #calcolo i cluster
    clusters = get_clusters(n_clusters, labels, image_list)


    #valutiamo la silhouette del clustering ottenuto
    silhouette = silhouette_score(X, labels)
    print(silhouette)

    #valutiamo la compattezza e separabilità dei cluster
    dbi = davies_bouldin_score(X, labels)
    print(dbi)

    results = [silhouette, dbi]

    #salviamo i risultati
    result = []
    i = 0
    for lab in labels:
        result.append([str(image_list[i]), int(lab)])
        i = i+1

    with open(save_folder + "\\resultTraining.txt", "w") as file:
        file.write(str(silhouette))
        for i in range(len(clusters)):
            file.write(str(i) + ": " + str(clusters[i]))

    np.save(save_folder + "\\centroidsKmeans.npy", centroids)
    np.save(save_folder + "\\labelsKmeans.npy", labels)
    np.save(save_folder + "\\resultsKmeans.npy", results)

    for i in range(len(clusters)):
      print(str(i) + ": " + str(clusters[i]))

    #Visualizziamo i Clusters
    show_clusters_3d(X, labels)
    show_centroids_2d(centroids)

applicate_Kmeans("preloaded", "Kmeans", "..\\dataset\\01.mixed")