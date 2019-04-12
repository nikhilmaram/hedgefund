import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from typing import List
from typing import Tuple
import networkx
import pandas as pd
from sklearn.preprocessing import normalize
import os
from collections import Counter
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering, AffinityPropagation


import network
import misc
import compute_mmd_distance as dist
import plot
import config as cfg
import employee


# =========================================================================
# ============= Compute Frobenius norm between two matrices================
# =========================================================================

def compute_frobenius_norm(business_embedding : np.ndarray, social_embedding: np.ndarray) -> float:
    """Computes frobenius norm between two embeddings.

    Args:
        business_embedding : Embedding of business network.
        social_embedding   : Embedding of social network.

    Returns:
        frobenius_distance : Frobenius distance between two embeddings.
    """
    frobenius_distance = np.linalg.norm(business_embedding - social_embedding)

    return frobenius_distance
# =========================================================================
# ============= Compute spectral embeddings given Matrix====================
# =========================================================================
def compute_spectral_embeddings_given_matrix(message_matrix, k:int) -> np.matrix:
    message_matrix1 = np.zeros((len(message_matrix), len(message_matrix)))
    for i in range(len(message_matrix)):
        for j in range(i + 1, len(message_matrix)):
            if (message_matrix[i][j] >=1):
                message_matrix1[i][j] = 1 ; message_matrix1[j][i] = 1
                # message_matrix1[i][j] = message_matrix[i][j] + message_matrix[j][i]; message_matrix1[j][i] = message_matrix1[i][j]

    # message_matrix = message_matrix1 + np.identity(298)
    message_matrix = message_matrix1

    print(message_matrix[0])
    print(message_matrix[1])
    lap = laplacian(message_matrix,normed=True)
    eigenValues, eigenVectors = np.linalg.eigh(lap)
    print(eigenValues)


# =========================================================================
# ============= Compute spectral embeddings given Graph====================
# =========================================================================
def compute_spectral_embeddings_given_graph(inp_G: nx.Graph, k : int) -> Tuple[np.matrix, List]:
    """Computes embeddings of given Graph.

    Args:
        inp_G : Input Graph.
        k : number of eigen-vectors to be considered.

    Returns:
        embeddings : Embeddings of input graph.
        nodelist   : To know the orderings of the node incase all the nodes in graph is not considered.
    """
    ## calculate normalized laplacian of input Graph. Normalized because comparison between two graphs makes sense
    print(nx.algorithms.components.number_connected_components(inp_G))
    lap = nx.normalized_laplacian_matrix(inp_G, weight='weight')
    # lap = nx.normalized_laplacian_matrix(inp_G)

    eigenValues, eigenVectors = np.linalg.eigh(lap.todense())
    # print(eigenValues)

    embeddings = eigenVectors[:,:k]
    embeddings = normalize(embeddings,norm='l2')
    nodelist = list(inp_G.nodes())
    return embeddings, nodelist


# =========================================================================
# ============= Compute spectral embeddings given Dataframe================
# =========================================================================

def compute_spectral_embeddings_given_im_df(im_df:pd.DataFrame, k : int, in_network:bool = True, include_all_nodes:bool = True) -> Tuple[np.matrix, List]:

    """Computes spectral embeddings given a file.

    Args:
        im_df : Input Dataframe
        k : number of eigen vectors to be considered.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message
        nodelist   : To know the orderings of the node incase all the nodes in graph is not considered.
    Returns:
        embeddings : Embeddings of input file.

    """

    message_matrix,_,_ = network.create_matrix(im_df,in_network=in_network)

    G = network.create_graph(message_matrix,un_directed=True,include_all_nodes = include_all_nodes)

    embeddings, nodelist = compute_spectral_embeddings_given_graph(G,k)
    return embeddings, nodelist


# =========================================================================
# ============= Compute embeddings given file list=========================
# =========================================================================

def compute_embedding_given_file_list(file_list : List, src_dir_path:str, user_name_list : List, k : int,
                                      in_network:bool = True, only_week:bool=False, include_all_nodes:bool = True) -> dict:
    """Computes embeddings from all files in file_list for the users present in user_name_list.

         Args:
             src_dir_path : path to the input files.
             user_name_list : user names for whose graph is calculated.
             k : Number of embeddings.
             in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)
             only_week           : data is calculated weekly instead of each date.
            include_all_nodes : whether to include nodes in the graph, which doesnt share a message

        Returns:
            embeddings_dict : Embeddings Dictionary. key : Date , value : embeddings of user messages on the day.
    """

    embeddings_dict = {}
    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(src_dir_path,file_name)
        im_df = pd.read_csv(file_path)

        ## get the dataframe for the chats corresponds to users in userlist.
        im_df = im_df[im_df["sender_user_name"].isin(user_name_list) | im_df["receiver_user_name"].isin(user_name_list)]

        if only_week:
            embeddings, _ = compute_spectral_embeddings_given_im_df(im_df,k,in_network,include_all_nodes)
            week_num = int(file_name.split('.')[0].split('_')[-1][4:])
            curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
            embeddings_dict[curr_date] = embeddings

        else:
            ## Group the data according to curr_date.
            for curr_date, df_curr_date in im_df.groupby("day"):
                print(curr_date)
                try: ## Incase there are no edges between users in user list.
                    embeddings, _ = compute_spectral_embeddings_given_im_df(df_curr_date,k,in_network,include_all_nodes)
                    embeddings_dict[curr_date] = embeddings
                except:
                    pass

    return embeddings_dict


# =========================================================================
# ============= Compute distance between social and business network=======
# =========================================================================

def compute_distance_between_business_and_social_embedding(business_dir_path: str, social_dir_path: str, user_name_list:List, k : int,
                                                           start_week:int, end_week:int, in_network:bool=True,only_week:bool = False):

    """Computes the distance between social and business embedding.

    Args:
        business_dir_path   : Business messages directory path.
        social_dir_path     : Social messages directory path.
        user_name_list      : Users for which the distance is calculated.
        k                   : Number of embeddings to be considered.
        start_week          : Start week.
        end_week            : End week.
        in_network          : Only users in the network considered. (True : IN, False: All)
        only_week           : Only week data is considered.

    Returns:
        distance_dict       : Distance dictionary. Key - data, value - distance between two embeddings.

    """

    distance_dict  = {}

    list_of_file_list = misc.splitting_all_files(business_dir_path, 1, start_week, end_week)
    file_list  = list_of_file_list[0]

    business_embeddings_dict = compute_embedding_given_file_list(file_list, business_dir_path, user_name_list, k, in_network,only_week)
    social_embeddings_dict = compute_embedding_given_file_list(file_list, social_dir_path, user_name_list, k,
                                                                 in_network, only_week)
    # print(business_embeddings_dict)

    business_embeddings_dict,social_embeddings_dict =  misc.common_keys(business_embeddings_dict,social_embeddings_dict)

    for date,_ in business_embeddings_dict.items():
        print(date)
        distance_dict[date] = compute_frobenius_norm(business_embeddings_dict[date],social_embeddings_dict[date])

    return distance_dict

# =========================================================================
# ============= Compute clusters given embeddings =========================
# =========================================================================

def clusters_given_embeddings(inp_embeddings : np.matrix,nodelist : List, n_clusters :int) -> Tuple[List[List], float]:
    """Computes clusters given embeddings.

    Args:
        inp_embedings : Input Embeddings.
        nodelist   : To know the orderings of the node incase all the nodes in graph is not considered.
        n_clusters             : Number of clusters.

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
        inertia        : sum of squared distances of samples to the nearest cluster centre.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(inp_embeddings)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    clusters =[[] for _ in range(len(unique_labels))]
    inertia = kmeans.inertia_

    for i in range(len(labels)):
        clusters[labels[i]].append(nodelist[i]) ## id + 1 because id indexing starts from 1.

    return clusters, inertia

# =========================================================================
# ============= Compute clusters given Dataframe =========================
# =========================================================================

def clusters_given_a_df(im_df : pd.DataFrame, k : int,n_clusters:int, in_network: bool =True,include_all_nodes:bool = True) -> Tuple[List[List], float]:
    """Compute clusters given a IM dataframe.

    Args:
        im_df : Input Dataframe
        k : number of eigen vectors to be considered.
        n_clusters : Number of clusters.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
        inertia        : sum of squared distances of samples to the nearest cluster centre.
    """
    embeddings, nodelist = compute_spectral_embeddings_given_im_df(im_df,k,in_network,include_all_nodes)
    clusters, inertia = clusters_given_embeddings(embeddings,nodelist, n_clusters)
    return clusters, inertia


# =========================================================================
# ============= Clusters given file list=================================
# =========================================================================
def compute_clusters_given_file_list(file_list : List, src_dir_path:str, k : int, n_clusters:int,
                                      in_network:bool = True, only_week:bool=False, include_all_nodes:bool = True) -> dict:
    """Computes clusters from all files in file_list for the users present in user_name_list.

         Args:
             src_dir_path : path to the input files.
             k : Number of embeddings vectors.
             n_clusters : Number of clusters.
             in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)
             only_week           : data is calculated weekly instead of each date.
             include_all_nodes : whether to include nodes in the graph, which doesnt share a message
        Returns:
            clusters_dict : Clusters Dictionary. key : Date , value : clusters present in that day.
    """

    clusters_dict = {}
    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(src_dir_path,file_name)
        im_df = pd.read_csv(file_path)

        if only_week:
            clusters,_ = clusters_given_a_df(im_df,k,n_clusters,in_network,include_all_nodes)
            week_num = int(file_name.split('.')[0].split('_')[-1][4:])
            curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
            clusters_dict[curr_date] = clusters

        else:
            ## Group the data according to curr_date.
            for curr_date, df_curr_date in im_df.groupby("day"):
                print(curr_date)
                try: ## Incase there are no edges between users in user list.
                    clusters, _ = clusters_given_a_df(df_curr_date, k,n_clusters, in_network,include_all_nodes)
                    clusters_dict[curr_date] = clusters
                except:
                    pass

    return clusters_dict


# =========================================================================
# ============= Common things in given clusters=============================
# =========================================================================

def compute_common_things_in_cluster(cluster_list : List):
    """Computes common thins in the clusters.

    Args:
        cluster_list : Cluster list.

    Returns:
        None.
    """
    for cluster in cluster_list:
        users_in_cluster = [employee.employee_id_to_username_dict[x] for x in cluster]

        user_title_in_cluster = [employee.employee_dict[user].title for user in users_in_cluster]
        user_front_office_sector_in_cluster = [employee.employee_dict[user].front_office_sector for user in users_in_cluster]
        user_dept_head_in_cluster = [employee.employee_dict[user].dept_head for user in users_in_cluster]
        user_top_user_in_hierarchy_in_cluster = [employee.employee_dict[user].top_user_in_hierarchy for user in users_in_cluster]
        user_location_in_cluster = [employee.employee_dict[user].location for user in users_in_cluster]
        user_title_name_in_cluster = [employee.employee_dict[user].title + " : " + user for user in users_in_cluster]

        user_book_in_cluster = []
        for user in users_in_cluster:
            book_list = employee.employee_to_account_dict.get(user,[])
            if(len(book_list) > 0):
                user_book_in_cluster.extend(book_list)

        # print(users_in_cluster)
        # print(len(users_in_cluster))
        print(user_title_name_in_cluster)

        # user_location_title_in_cluster = [employee.employee_dict[user].location + " : " + employee.employee_dict[user].title for user in users_in_cluster]
        # current_consideration = user_location_title_in_cluster
        # counter_in_cluster = sorted(Counter(current_consideration).items())
        # print(counter_in_cluster)

        # current_consideration = user_book_in_cluster
        # counter_in_cluster = sorted(Counter(current_consideration).items())
        # print(counter_in_cluster)


# =========================================================================
# ============= Plot Elbow Curve =========================
# =========================================================================

def plot_elbow_curve(im_df : pd.DataFrame, k:int, max_n_clusters : int, in_network: bool = True, include_all_nodes: bool = True):
    """Plots the elbow curve with different number of clusters for given dataframe.

    Args:
        im_df : Input Dataframe
        k : number of eigen vectors
        max_n_clusters : Maximum number of clusters.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All).
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message
    Returns:
        None.
    """

    inertia_list = []
    for n_cluster in range(1, max_n_clusters+1):
        _, inertia = clusters_given_a_df(im_df, k,n_cluster,in_network,include_all_nodes)
        inertia_list.append(inertia)

    num_clusters_list = list(range(1,max_n_clusters+1))
    plot.general_plot(num_clusters_list,inertia_list,"Number of Clusters", "Sum of Squared Distance", "Elbow curve", "")


# =========================================================================
# ============= Spectral clustering using Scikit===========================
# =========================================================================

def spectral_clustering_using_scikit(im_df : pd.DataFrame,n_clusters :int, in_network = True) -> Tuple[List[List], float]:
    """Compute spectral clustering using data frame.

    Args:
        im_df : input dataframe.
        n_clusters : Number of clusters.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All).

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
        inertia        : sum of squared distances of samples to the nearest cluster centre.
    """

    message_matrix, buddy_to_idx,idx_to_buddy = network.create_matrix_for_edge_present(im_df, in_network)
    clustering = SpectralClustering(n_clusters=n_clusters,  assign_labels = "discretize", random_state=0).fit(message_matrix)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    clusters = [[] for _ in range(len(unique_labels))]

    for i in range(1,len(labels)):
        clusters[labels[i]].append(idx_to_buddy[i]) ## i + 1 because id indexing starts from 1.

    return clusters, 0.0


if __name__ == "__main__":
    # ========================================================================================
    # ===========Spectral embedding=====================
    # =========================================================================================
    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL + "/im_df_week123.csv")
    # message_matrix,_,_ = network.create_matrix(im_df,in_network=True)
    # compute_spectral_embeddings_given_matrix(message_matrix,5)

    # ========================================================================================
    # ===========Plotting the elbow curve=====================
    # =========================================================================================

    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL+"/im_df_week123.csv")
    # plot_elbow_curve(im_df, k=10,max_n_clusters=25,in_network=True,include_all_nodes=False)

    # ========================================================================================
    # =========== Clusters given file list=====================
    # =========================================================================================
    #
    src_dir_path = cfg.SENTIMENT_PERSONAL ;  start_week = 125 ;  end_week = 125
    list_of_file_list = misc.splitting_all_files(src_dir_path, 1, start_week, end_week)
    file_list = list_of_file_list[0]

    cluster_dict = compute_clusters_given_file_list(file_list, src_dir_path, k=10,n_clusters = 10,
                                                    in_network=True, only_week=True, include_all_nodes = False)
    print(cluster_dict)
    for date, cluster_list in cluster_dict.items():
        print("================================={0}==============================".format(date))
        compute_common_things_in_cluster(cluster_list)

    src_dir_path = cfg.SENTIMENT_BUSINESS; start_week = 123; end_week = 123
    list_of_file_list = misc.splitting_all_files(src_dir_path, 1, start_week, end_week)
    file_list = list_of_file_list[0]

    cluster_dict = compute_clusters_given_file_list(file_list, src_dir_path, k=10, n_clusters=15,
                                                    in_network=True, only_week=True, include_all_nodes=False)
    print(cluster_dict)
    for date, cluster_list in cluster_dict.items():
        print("================================={0}==============================".format(date))
        compute_common_things_in_cluster(cluster_list)

    # ========================================================================================
    # ===========Spectral clustering using scikit =====================
    # =========================================================================================

    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL + "/im_df_week123.csv")
    # clusters,_ = spectral_clustering_using_scikit(im_df, 10, True)
    #
    # print(clusters)
    # # compute_common_things_in_cluster(clusters)
    #
    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL + "/im_df_week124.csv")
    # clusters_2, _ = spectral_clustering_using_scikit(im_df, 10, True)
    # print(clusters_2)
    #
    # for i in range(len(clusters)):
    #     print(clusters[i])
    #
    # print("==========================================================================================================================================")
    # for j in range(len(clusters_2)):
    #     print(clusters_2[j])

    # for i in range(10):
    #     for j in range(10):
    #         clusters[i] = sorted(clusters[i])
    #         clusters_2[j] = sorted(clusters_2[j])
    #         print(len(clusters[i]), len(clusters_2[j]))
    #         if( clusters[i] == clusters_2[j]):
    #             print("both clusters are same")
    pass


