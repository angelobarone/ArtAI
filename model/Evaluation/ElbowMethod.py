import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def find_optimal_k(X, max_k):
    sse = []

    for k in tqdm(range(1, max_k + 1), desc="Elbow method calculating"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_) #l'inerzia corrisponde all'sse

    #Trovare il punto di gomito con kneedle
    kneedle = KneeLocator(range(1, max_k + 1), sse, curve="convex", direction="decreasing")
    optimal_k = kneedle.knee

    #Grafico metodo del gomito
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='--')
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r', label=f'Ottimale k={optimal_k}')
    plt.title('Metodo del Gomito con K ottimale evidenziato')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.legend()
    plt.show()


    return optimal_k