import numpy as np

def bottomup_clustering(X, n_clusters, file_names):

    #Calcolo del numero di cluster iniziale (tutti i punti di X)
    n_samples = X.shape[0]

    clusters = [[file_names[i]] for i in range(len(X))]
    labels = np.zeros([n_samples, 2])
    for i in range(n_samples):
        labels[i][0] = file_names[i]
        labels[i][1] = int(i)

    #Calcolo la distanza di ogni punto dagli altri
    #Inizializzazione a 0 della matrice: sarà n_samples x n_samples e conterrà la distanza di ogni punto dagli altri punti
    matrice_distanze = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            #Distanza euclidea
            distance = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            #ricordiamo che la matrice è simmetrica
            matrice_distanze[i][j] = distance
            matrice_distanze[j][i] = distance

    #Agglomerazione fino al raggiungimento di n_clusters
    while len(clusters) > n_clusters:
        #troviamo i 2 cluster più vicini
        min_dist = np.inf
        cluster1, cluster2 = -1,-1
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                if matrice_distanze[i][j] < min_dist:
                    min_dist = matrice_distanze[i][j]
                    cluster1 = i
                    cluster2 = j

        # Aggiornamento dei labels
        h = min(cluster1, cluster2)
        z = max(cluster1, cluster2)
        labels[z][1] = h
        #Aggiornamento del centroide
        X[h] = ((X[h]*len(clusters[h])) + X[z]*len(clusters[z]))/(len(clusters[h]) + len(clusters[z]))

        clusters[h].extend(clusters[z])
        X = np.delete(X, z, 0)

        #Rimozione del cluster maggiore e unione sul cluster minore
        clusters.pop(z)

        #rimozione delle distanze del cluster eliminato dalla matrice
        matrice_distanze = np.delete(matrice_distanze, z, axis=0)
        matrice_distanze = np.delete(matrice_distanze, z, axis=1)

        #Distanze del nuovo cluster
        for i in range(len(clusters)):
            if i != h:
                distance = np.sqrt(np.sum((X[i] - X[h]) ** 2))
                matrice_distanze[h][i] = distance
                matrice_distanze[i][h] = distance

    return X, clusters, labels