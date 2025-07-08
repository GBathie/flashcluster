import swift_cluster
import numpy as np

print(dir(swift_cluster))

data = np.random.random((100, 20)).astype(dtype=np.float32)
um = swift_cluster.compute_clustering(data, 1.5, "fast")
x = swift_cluster.Ultrametric(data, 1.5, "fast")

for i in range(100):
    for j in range(i + 1, 100):
        print(i, j, x.dist(i, j))
