import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from model.algorithm import KmeansClustering
from model.evaluation.MetricheInterne import calculate_silhouette_score, calculate_inertia
from model.training.creazioneMatrice import load_dataset_from_folder

preloaded_path_X = "preloaded\\X.npy"
preloaded_path_image_list = "preloaded\\image_list.npy"

# carica il Dataset
if os.path.exists(preloaded_path_X) and os.path.exists(preloaded_path_image_list):
    X = np.load(preloaded_path_X)
    image_list = np.load(preloaded_path_image_list)
else:
    X, image_list = load_dataset_from_folder("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\MET", 5000)#841587
    np.save("preloaded\\X.npy", X)
    np.save("preloaded\\image_list.npy", image_list)
    #Standardizzazione dei dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "trained\\scaler.pkl")

#Eseguiamo il clusteing k-means su k cluster
k = 30
centroids, labels = KmeansClustering.kmeans(X, k)

#valutiamo la silhouette del clustering ottenuto
silhouette = calculate_silhouette_score(X, labels)
print(silhouette)

#valutiamo l'inertia
inertia = calculate_inertia(X, centroids, labels)
print(inertia)

#salviamo i risultati
result = []
i = 0
for lab in labels:
    result.append([int(image_list[i]), int(lab)])
    i = i+1

with open("resultTraining.txt", "w") as file:
    file.write(str(result))

np.save("preloaded\\centroids.npy", centroids)
np.save("preloaded\\labels.npy", labels)
np.save("resultTraining.npy", result)

print(result)
