import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def find_optimal_silhouette(X, min_k, max_k):
    silhouette = []

    pca = PCA(n_components=3)
    X = pca.fit_transform(X)

    for k in tqdm(range(min_k, max_k + 1), desc="Silhouette Calculating"):
        clustering = AgglomerativeClustering(n_clusters=k, linkage = "ward", memory="..\\tmp\\cache")
        clustering.fit(X)
        silhouette.append(silhouette_score(X, clustering.labels_))

    optimal_index = np.argmax(silhouette)
    optimal_k = min_k + optimal_index

    # Grafico Silhouette
    plt.figure(figsize=(8, 5))
    plt.plot(range(min_k, max_k + 1), silhouette, marker='o', linestyle='--')
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r',label=f'Ottimale k={optimal_k}')
    plt.title('Valore della Silhouette al variare di K')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Silhouette')
    plt.xticks(range(min_k, max_k + 1))
    plt.grid(True)
    plt.legend()
    plt.show()

    return optimal_k
