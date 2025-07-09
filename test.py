import flashcluster
import numpy as np


data = np.random.random((100, 20)).astype(dtype=np.float32)
um = flashcluster.compute_clustering(data, 1.5, "fast")
x = flashcluster.Ultrametric(data, 1.5, "fast")


data = np.random.random((100, 20)).astype(dtype=np.float64)
um = flashcluster.compute_clustering(data, 1.5, "fast")
x = flashcluster.Ultrametric(data, 1.5, "fast")


data = np.random.random((1000, 20))
um = flashcluster.compute_clustering(data, 2, "fast")

print("Done!")
