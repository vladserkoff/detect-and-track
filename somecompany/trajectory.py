"""Vehicle trajectory processing and clustering"""

from typing import Sequence

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import HDBSCAN
from tqdm.auto import tqdm

from somecompany.logger import logging

LOG = logging.getLogger(__name__)


def cluster_trajectories(trajectories: Sequence[np.ndarray], min_samples: int = 1) -> np.ndarray:
    """Clusters trajectories using the Hausdorff distance metric and HDBSCAN.
    The (directed) Hausdorff distance serves as a measure of similarity between trajectories.
    HDBSCAN is used to cluster the trajectories based on the distance matrix into an unknown number of clusters.

    Args:
        trajectories (np.ndarray): Trajectories to cluster, represented by points in [x, y] format.
        min_cluster_size (int, optional): Minimum number of trajectories in a cluster.

    Returns:
        np.ndarray: Cluster labels for each trajectory.
    """
    clusterer = HDBSCAN(min_samples=min_samples, metric="precomputed")
    distance_matrix = compute_distance_matrix(trajectories)
    cluster_labels = clusterer.fit_predict(distance_matrix)
    LOG.info(f"Clustered {len(trajectories)} trajectories into {len(np.unique(cluster_labels))} clusters")
    return cluster_labels


def compute_distance_matrix(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    """Computes the distance matrix between trajectories using the Hausdorff distance metric.

    Args:
        trajectories (np.ndarray): Trajectories to cluster, represented by points in [x, y] format.

    Returns:
        np.ndarray: Distance matrix between the trajectories.
    """
    distance_matrix = np.zeros((len(trajectories), len(trajectories)), dtype=np.float16)  # 16 bit floats are enough
    for i, trajectory in tqdm(
        enumerate(trajectories),
        total=len(trajectories),
        desc="Computing distance matrix",
        disable=not LOG.isEnabledFor(logging.DEBUG),
    ):
        # distance matrix is symmetric, so we only need to compute the upper triangle
        for j, other_trajectory in enumerate(trajectories[i:], start=i):
            dist = directed_hausdorff(trajectory, other_trajectory)[0]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix
