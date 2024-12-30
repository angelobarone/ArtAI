import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def find_optimal_silhouette(X, max_k=10):
    silhouette = []

    for k in tqdm(range(2, max_k + 1), desc="Silhouette Calculating"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        silhouette.append(silhouette_score(X, kmeans.labels_))

    optimal_silhouette = silhouette.index(max(silhouette))
    optimal_silhouette += 2

    # Grafico Silhouette
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), silhouette, marker='o', linestyle='--')
    plt.vlines(optimal_silhouette, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r',label=f'Ottimale k={optimal_silhouette}')
    plt.title('Valore della Silhouette al variare di K')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Silhouette')
    plt.xticks(range(2, max_k + 1))
    plt.grid(True)
    plt.legend()
    plt.show()

    return optimal_silhouette
