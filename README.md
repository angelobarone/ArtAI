<p align="center"><img src="" height="400"></p>

# ArtAi - Clusterizzazione di Immagini e Ricerca di Immagini Simili
ArtAi `e un progetto sviluppato per fornire un sistema in grado di clusterizzare un dataset
di immagini e, successivamente, predire la tipologia di opera fornita sulla base della
ricerca di immagini simili. Il sistema sfrutta tecniche di clustering non supervisionato e
reti neurali convoluzionali per l’estrazione delle feature visive.

# Tecnologie Utilizzate
Il progetto è implementato mediante:
- **CNN**: reti neurali convoluzionali
- **Algoritmi di Clustering**: K-means, Clustering Gerarchico e DBSCAN, per provare diversi approcci al problema.
- **Python**: linguaggio di programmazione utilizzato.

# Inizializzazione e Utilizzo
Per installare e utilizzare il modello, seguire questi passaggi
1. Clonare il repository da GitHub
2. Avviare lo script starter personalizzando i campi `alg` e `dataset_path` per eseguire il clustering sul dataset al path specificato
3. Ora è possibile implementare e utilizzare la funzione `similar_images, results = get_similar_images("path-immagine", "algoritmo", numero-immagini-restituite, "path-del-file-csv")`

# Contributi e Segnalazioni di Bug
E' possibile contribuire al progetto e fornire segnalazioni per migliorare ArtAi. Per contribuire, seguire queste linee guida:
1. Forkare il repository
2. Creare un branch per il proprio lavoro: `git checkout -b feature/nuova-funzionalità`
3. Committare le modifiche: `git commit -m 'Aggiunta nuova funzionalità'`
4. Pushare il branch al repository remoto: `git push origin feature/nuova-funzionalità`
5. Aprire una Pull Request

Per segnalare bug o proporre miglioramenti, aprire una nuova issue nel repository.
