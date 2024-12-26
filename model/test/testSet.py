import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from model.test.ImageLoaderTest import image_loader_test
from model.test.PredictCluster import predict_cluster
from model.test.SimilarImages import similarimages_set
from model.preprocessing.combinedFeatures import extract_combined_features

images = image_loader_test("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\test_met")
result_test = []
i = 0
for image in images:
    features = extract_combined_features(image[1])

    #modifica della dimensionalità
    pca_loaded = joblib.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\trained\\pca_model.pkl")
    features = pca_loaded.transform(features.reshape(1, -1))

    #scaling
    scaler_loaded = joblib.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\trained\\scaler.pkl")
    scaler = scaler_loaded.transform(features)

    clusters = np.load("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\training\\preloaded\\centroids.npy")
    closest_cluster = predict_cluster(features, clusters)
    result_test.append([image[0], closest_cluster])
    i=i+1
    if i > 30:
        break

print(str(result_test))

final_result = similarimages_set(result_test)

with open("result.txt", "w") as file:
    for item in final_result:
        file.write(str(item))
        file.write("\n")