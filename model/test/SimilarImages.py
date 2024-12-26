import joblib
import numpy as np

from model.test.PredictCluster import predict_cluster
from model.preprocessing.combinedFeatures import extract_combined_features


def similarimages_set(result_test):
    result_training = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\resultTraining.npy")

    final_result = []

    for image in result_test:
        cluster = image[1]
        #filtriamo i dati di training estraendo solo quelli aderenti al cluster dell'immagine che stiamo analizzando
        filtered_train = result_training[result_training[:, 1].astype(int) == cluster]
        if filtered_train.shape[0] < 2:
            similar_images = np.random.choice(filtered_train[:, 0], size=1, replace=False)
        elif filtered_train.shape[0] < 3:
            similar_images = np.random.choice(filtered_train[:, 0], size=2, replace=False)
        elif filtered_train.shape[0] < 4:
            similar_images = np.random.choice(filtered_train[:, 0], size=3, replace=False)
        elif filtered_train.shape[0] < 5:
            similar_images = np.random.choice(filtered_train[:,0], size=4, replace=False)
        else:
            similar_images = np.random.choice(filtered_train[:, 0], size=5, replace=False)

        final_result.append([image[0], similar_images])

    return final_result

def similarimages_single(image):
    result_training = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\resultTraining.npy")

    features = extract_combined_features(image)

    # modifica della dimensionalità
    pca_loaded = joblib.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\trained\\pca_model.pkl")
    features = pca_loaded.transform(features.reshape(1, -1))

    # scaling
    scaler_loaded = joblib.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\trained\\scaler.pkl")
    scaler = scaler_loaded.transform(features)

    clusters = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\preloaded\\centroids.npy")
    closest_cluster = predict_cluster(features, clusters)
    filtered_train = result_training[result_training[:, 1].astype(int) == closest_cluster]
    similar_images = []
    for i in range(5):
        similar_images.append([int(np.random.choice(filtered_train[:, 0])), ])

    return similar_images