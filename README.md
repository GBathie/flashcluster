# Flashcluster, a lightning-fast hierarchical clustering package.

## Getting started

### Installing the package

`flashcluster` can be installed with `pip`:
```bash
pip install flashcluster
```

### Basic usage

`flashcluster` is designed work with `numpy` arrays.
```py
from flashcluster import compute_clustering
import numpy as np

# Create dataset with 100 points of dimension 20
data = np.random.random((100, 20))

# Compute clustering
clustering = compute_clustering(data, c=1.5, mode="fast")

# Print the clustering distance (i.e. dissimilarity) of points 0 and 1.
print(clustering.dist(0, 1))
```