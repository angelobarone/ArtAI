import numpy as np

def kmeans(X, k, max_iter=100, tol=1e-4):

    global new_centroids, labels

    #Inizializzazione casuale dei centroidi (scegliamo k punti casuali)
    #Genera k indici casuali unici tra 0 e len(X)-1
    random_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[random_indices]
    #Variabile per tracciare la convergenza (valore precedente dei centroidi)
    prev_centroids = centroids.copy()

    #Ciclo che assegna gli elementi del dataset ai centroidi
    for i in range(max_iter):
        #Assegnazione di ogni punto al centroide più vicino
        n_points = X.shape[0]
        n_centroids = centroids.shape[0]

        #Inizializzazione della lista per le etichette (assegnazioni) a zero per ogni punto
        labels = np.zeros(n_points)

        #Ciclo per calcolare la distanza tra ogni punto e ogni centroide
        for h in range(n_points):
            min_distance = float('inf')  # Impostiamo una distanza infinita come valore di partenza
            closest_centroid = -1  # Indice del centroide più vicino
            for j in range(n_centroids):
                #Distanza euclidea tra il punto (X[i]) e il centroide (centroids[j])
                distance = np.sqrt(np.sum((X[h] - centroids[j]) ** 2))
                #Distanza minore
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = j

            #Assegniamo il punto h al centroide più vicino
            labels[h] = closest_centroid

        #Calcolare i nuovi centroidi
        #Inizializzazione di un array vuoto per i nuovi centroidi
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):  # Per ogni cluster
            #Punti appartenenti al cluster j
            cluster_points = X[labels == j]

            #Calcolo della media di ogni caratteristica (dimensione) dei punti nel cluster per ottenere il nuovo centroide
            new_centroids[j] = np.mean(cluster_points, axis=0)

        #Controlla la convergenza (se i centroidi non cambiano più)
        #Condizione di terminazione
        if np.all(np.abs(new_centroids - prev_centroids) < tol):
           print(f"Convergenza raggiunta dopo {i + 1} iterazioni.")
           break

        #np.abs calcola il valore assoluto della differenza
        #<tol restituisce true se la differenza tra i centroidi è inferiore a tol altrimente false
        #np.all verifica se tutti i valori nella matrice booleana sono true (se tutti i valori sono true vuol dire che i centroidi sono inferiori alla soglia di tolleranza)
        #Aggiorna i centroidi
        prev_centroids = new_centroids

    return new_centroids, labels