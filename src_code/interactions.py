import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from employee import employee_id_to_username_from_file

def interactions_clustering(message_matrix: np.ndarray, num_cluster: int, file_path: str) -> Tuple[np.ndarray,dict]:
    """
    Compute the embeddings of social network, business network, and joint network using spectral embedding.
    Find clusters in each network
    :param message_matrix: each entry represents the number of sent and recieved instant messages between two traders
    :param num_cluster: the number of clusters
    :param file_path: path to Employee Master File
    :return:
    embeddings: the embeddings of each employee in eigenspace
    all_cluster_username_dict: Key: cluster id, Value : list of employees' names: last_name + "_" + first_name
    """
    message_matrix1 = message_matrix[1:message_matrix.shape[0], :]
    message_matrix1 = message_matrix1 + message_matrix1.T
    G = nx.Graph()
    for src in range(len(message_matrix1)):
        for dest in range(len(message_matrix1[0])):
            if src != dest and message_matrix1[src][dest] != 0:
                # if (message_matrix1[src][dest] > threshold):
                G.add_edge(src, dest, weight=message_matrix1[src][dest])

    lap = nx.normalized_laplacian_matrix(G, weight='weight')
    # U, S, V = np.linalg.svd(normalized_matrix, full_matrices=True)
    embeddings = np.zeros((len(message_matrix1), num_cluster))
    lamb, U = np.linalg.eigh(lap.todense())
    for i in range(len(message_matrix1)):
        for j in range(num_cluster):
            embeddings[i][j] = U[i][j]

    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)

    employee_id_to_username_dict, _ = employee_id_to_username_from_file(file_path)

    all_cluster_username_dict = {}
    for i in range(len(unique_labels)):
        ids = []
        for j in range(len(labels)):
            if unique_labels[i] == labels[j]:
                ids.append(employee_id_to_username_dict[j + 1])
        all_cluster_username_dict[i] = ids

    return embeddings, all_cluster_username_dict

