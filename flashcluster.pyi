"""
Typing stubs and documentation for flashcluster
"""

from typing import Literal
import numpy as np

def compute_clustering(
    points: np.ndarray, c: float, mode: Literal["fast", "precise"]
) -> Ultrametric:
    """
    Compute an approximation of the optimal hierarchical clustering of the given set of points.

    :param points: set of points for clustering. Must be a 2D numpy array, with dtype=np.f32
    :param c: approximation factor. Must be greater than 1.0
    :param mode: "fast": may under-estimate the distance in some cases, but performs better.
        "precise": almost never under-estimates the distance, but returns larger distances.
    """

class Ultrametric:
    """
    A class representing a hierarchical clustering.

    :param points: set of points for clustering. Must be a 2D numpy array, with dtype=np.f32
    :param c: approximation factor. Must be greater than 1.0
    :param mode: "fast": may under-estimate the distance in some cases, but performs better on average.
        "precise": almost never under-estimates the distance, but returns larger distances.
    """
    def __init__(self, points: np.ndarray, c: float, mode: Literal["fast", "precise"]) -> None: ...
    def dist(self, i: int, j: int) -> float:
        """
        Returns the distance in the clustering between points[i, :] and points[j, :].
        """
