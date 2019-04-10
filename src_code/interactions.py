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
# ============= Compute spectral embeddings given Graph====================
# =========================================================================
def compute_spectral_embeddings_given_graph(inp_G: nx.Graph, k : int) -> np.matrix:
    """Computes embeddings of given Graph.

    Args:
        inp_G : Input Graph.
        k : number of eigen vectors to be considered.

    Returns:
        embeddings : Embeddings of input graph.
    """
    ## calculate normalized laplacian of input Graph. Normalized because comparison between two graphs makes sense
    lap = nx.normalized_laplacian_matrix(inp_G, weight='weight')
    # lap = nx.normalized_laplacian_matrix(inp_G)

    eigenValues, eigenVectors = np.linalg.eigh(lap.todense())
    embeddings = eigenVectors[:,:k]
    embeddings = normalize(embeddings,norm='l2')
    # print(type(embeddings))
    return embeddings


# =========================================================================
# ============= Compute spectral embeddings given Dataframe================
# =========================================================================

def compute_spectral_embeddings_given_im_df(im_df:pd.DataFrame, k : int, in_network:bool = True) -> np.matrix:

    """Computes spectral embeddings given a file.

    Args:
        im_df : Input Dataframe
        k : number of eigen vectors to be considered.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)

    Returns:
        embeddings : Embeddings of input file.

    """

    message_matrix,_,_ = network.create_matrix(im_df,in_network=in_network)

    G = network.create_graph(message_matrix,un_directed=True)

    embeddings = compute_spectral_embeddings_given_graph(G,k)
    return embeddings



# =========================================================================
# ============= Compute embeddings given file list=========================
# =========================================================================

def compute_embedding_given_file_list(file_list : List, src_dir_path:str, user_name_list : List, k : int,
                                      in_network:bool = True, only_week:bool=False) -> dict:
    """Computes embeddings from all files in file_list for the users present in user_name_list.

         Args:
             src_dir_path : path to the input files.
             user_name_list : user names for whose graph is calculated.
             k : Number of embeddings.
             in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)
             only_week           : data is calculated weekly instead of each date.

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
            embeddings = compute_spectral_embeddings_given_im_df(im_df,k,in_network)
            week_num = int(file_name.split('.')[0].split('_')[-1][4:])
            curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
            embeddings_dict[curr_date] = embeddings

        else:
            ## Group the data according to curr_date.
            for curr_date, df_curr_date in im_df.groupby("day"):
                print(curr_date)
                try: ## Incase there are no edges between users in user list.
                    embeddings = compute_spectral_embeddings_given_im_df(df_curr_date,k,in_network)
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

def clusters_given_embeddings(inp_embeddings : np.matrix, k :int) -> Tuple[List[List], float]:
    """Computes clusters given embeddings.

    Args:
        inp_embedings : Input Embeddings.
        k             : Number of clusters.

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
        inertia        : sum of squared distances of samples to the nearest cluster centre.
    """

    kmeans = KMeans(n_clusters=k, random_state=0).fit(inp_embeddings)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    clusters =[[] for _ in range(len(unique_labels))]
    inertia = kmeans.inertia_

    for id in range(len(labels)):
        clusters[labels[id]].append(id+1) ## id + 1 because id indexing starts from 1.

    return clusters, inertia

# =========================================================================
# ============= Compute clusters given Dataframe =========================
# =========================================================================

def clusters_given_a_df(im_df : pd.DataFrame, k : int,in_network: bool =True) -> Tuple[List[List], float]:
    """Compute clusters given a IM dataframe.

    Args:
        im_df : Input Dataframe
        k : number of eigen vectors to be considered.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
        inertia        : sum of squared distances of samples to the nearest cluster centre.
    """
    embeddings = compute_spectral_embeddings_given_im_df(im_df,k,in_network)
    clusters, inertia = clusters_given_embeddings(embeddings,k)
    return clusters, inertia


# =========================================================================
# ============= Clusters given file list=================================
# =========================================================================
def compute_clusters_given_file_list(file_list : List, src_dir_path:str, k : int,
                                      in_network:bool = True, only_week:bool=False) -> dict:
    """Computes clusters from all files in file_list for the users present in user_name_list.

         Args:
             src_dir_path : path to the input files.
             k : Number of embeddings/clusters.
             in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)
             only_week           : data is calculated weekly instead of each date.

        Returns:
            clusters_dict : Clusters Dictionary. key : Date , value : clusters present in that day.
    """

    clusters_dict = {}
    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(src_dir_path,file_name)
        im_df = pd.read_csv(file_path)

        if only_week:
            clusters,_ = clusters_given_a_df(im_df,k,in_network)
            week_num = int(file_name.split('.')[0].split('_')[-1][4:])
            curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
            clusters_dict[curr_date] = clusters

        else:
            ## Group the data according to curr_date.
            for curr_date, df_curr_date in im_df.groupby("day"):
                print(curr_date)
                try: ## Incase there are no edges between users in user list.
                    clusters, _ = clusters_given_a_df(df_curr_date, k, in_network)
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



        current_consideration = user_location_in_cluster
        counter_in_cluster = sorted(Counter(current_consideration).items())
        print(counter_in_cluster)

        # current_consideration = user_title_in_cluster
        # counter_in_cluster = sorted(Counter(current_consideration).items())
        # print(counter_in_cluster)





# =========================================================================
# ============= Plot Elbow Curve =========================
# =========================================================================

def plot_elbow_curve(im_df : pd.DataFrame, max_k : int, in_network: bool = True):
    """Plots the elbow curve with different number of clusters for given dataframe.

    Args:
        im_df : Input Dataframe
        max_k : Maximum number of clusters.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All).

    Returns:
        None.
    """

    inertia_list = []
    for k in range(1, max_k+1):
        _, inertia = clusters_given_a_df(im_df, k,in_network)
        inertia_list.append(inertia)

    num_clusters_list = list(range(1,max_k+1))
    plot.general_plot(num_clusters_list,inertia_list,"Number of Clusters", "Sum of Squared Distance", "Elbow curve", "")



if __name__ == "__main__":

    # ========================================================================================
    # ===========Plotting the elbow curve=====================
    # =========================================================================================

    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL+"/im_df_week180.csv")
    # plot_elbow_curve(im_df, 25,True)

    # ========================================================================================
    # =========== Clusters given file list=====================
    # =========================================================================================

    src_dir_path = cfg.SENTIMENT_PERSONAL ;  start_week = 124 ;  end_week = 125
    list_of_file_list = misc.splitting_all_files(src_dir_path, 1, start_week, end_week)
    file_list = list_of_file_list[0]

    cluster_dict = compute_clusters_given_file_list(file_list, src_dir_path, 10, in_network=True, only_week=True)
    print(cluster_dict)
    for date, cluster_list in cluster_dict.items():
        print("================================={0}==============================".format(date))
        compute_common_things_in_cluster(cluster_list)
    pass


