from model.Application.gui import start_gui
from model.Clustering.ClusteringBottomUp import applicate_BottomUp
from model.Clustering.ClusteringDBSCAN import applicate_DBSCAN
from model.Clustering.ClusteringKmeans import applicate_Kmeans

alg = "kmeans"
dataset_path = "..\\dataset\\01.mixed"


if alg == "kmeans":
    applicate_Kmeans("preloaded", "Kmeans", dataset_path)
if alg == "bottompup":
    applicate_BottomUp("preloaded", "BottomUp", dataset_path)
else:
    applicate_DBSCAN("preloaded", "DBSCAN", dataset_path)
