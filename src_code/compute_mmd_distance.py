import numpy as np
import random
from scipy.spatial.distance import pdist, cdist, squareform


def compute_mmd(business_data: np.ndarray, personal_data: np.ndarray) -> float:
    """

    :param business_data: the embeddings of business network
    :param personal_data: the embeddings of social network
    :return: the maximum mean discrepancy
    """
    num_graphs = 2
    allFeatures = {}
    allFeatures[0] = business_data
    allFeatures[1] = personal_data
    num_features = len(business_data[0])

    sample_num = 5
    sample_feature = np.empty((0, num_features))
    # print(sample_feature.shape)

    for gidx in range(num_graphs):
        M = allFeatures[gidx]
        n = len(M)
        # print(M.shape)
        if n >= sample_num:
            ind = random.sample(range(n), sample_num)
            for index in ind:
                sample_feature = np.append(sample_feature, [M[index, :]], axis=0)
    sample_dist = pdist(sample_feature, 'euclidean')
    sigma = np.median(sample_dist)
    if sigma == 0:
        sigma = 0.00001

    self_dist = np.zeros(num_graphs)

    for gidx in range(num_graphs):
        M = allFeatures[gidx]
        n = len(M)
        feature_dist = squareform(pdist(M, 'euclidean'))
        Kself = np.exp(-1 * np.divide(feature_dist, 1.0 * sigma))

        for i in range(n):
            for j in range(n):
                self_dist[gidx] += Kself[i][j]
        self_dist[gidx] = np.divide(self_dist[gidx], 1.0 * n * n)

    X = allFeatures[0]
    nx = len(X)

    Y = allFeatures[1]
    ny = len(Y)
    xy_feature_dist = cdist(X, Y, 'euclidean')
    Kxy = np.exp(-1 * np.divide(xy_feature_dist, 1.0 * sigma))

    xy_dist = 0
    for i1 in range(nx):
        for j1 in range(ny):
            xy_dist += Kxy[i1][j1]
    xy_dist = np.divide(xy_dist, 1.0 * nx * ny)

    mmdist = self_dist[0] + self_dist[1] - 2 * xy_dist

    return mmdist