import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA


def show_clusters_3d(X, labels):
    pca = PCA(n_components=3)
    X_trasformed = pca.fit_transform(X)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_trasformed[:, 0], X_trasformed[:, 1], X_trasformed[:, 2], c=labels, cmap='viridis', s=100)
    ax.set_title("Clusters in 3D")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()
    plt.show()

def show_clusters_2d(X, labels):
    pca = PCA(n_components=2)
    X_trasformed = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for cluster_num in np.unique(labels):
        # Punti del cluster
        cluster_points = X_trasformed[labels == cluster_num]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_num}")
    plt.title("Visualizzazione dei Cluster Agglomerativi")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def show_dendrogram(linkage):
    plt.figure(figsize=(40, 10))
    dendrogram(linkage)
    plt.title("Dendrogramma del Clustering Gerarchico")
    plt.xlabel("Indice dei campioni")
    plt.ylabel("Distanza")
    plt.show()