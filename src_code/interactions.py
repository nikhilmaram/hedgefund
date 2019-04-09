import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from typing import List
from typing import Tuple
import networkx
import pandas as pd
from sklearn.preprocessing import normalize
import os

import network
import misc
import compute_mmd_distance as dist

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
    # print(G.edges(data=True))
    embeddings = compute_spectral_embeddings_given_graph(G,k)
    return embeddings

# =========================================================================
# ============= Compute clusters given embeddings =========================
# =========================================================================

def clusters_given_embeddings(inp_embeddings : np.matrix, k :int) -> List[List]:
    """Computes clusters given embeddings.

    Args:
        inp_embedings : Input Embeddings.
        k             : Number of clusters.

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(inp_embeddings)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    clusters =[[] for _ in range(unique_labels)]

    for id in range(len(labels)):
        clusters[labels[id]].append(id+1) ## id + 1 because id indexing starts from 1.

    return clusters

# =========================================================================
# ============= Compute clusters given Dataframe =========================
# =========================================================================

def clusters_given_a_df(im_df : pd.DataFrame, k : int,in_network: bool =True) -> List[List]:
    """Compute clusters given a IM dataframe.

    Args:
        im_df : Input Dataframe
        k : number of eigen vectors to be considered.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)

    Returns:
        clusters       : List of lists, where each list correspond to ids of users belong to a cluster.
    """
    embeddings = compute_spectral_embeddings_given_im_df(im_df,k,in_network)
    clusters = clusters_given_embeddings(embeddings,k)
    return clusters

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
    print(business_embeddings_dict)

    business_embeddings_dict,social_embeddings_dict =  misc.common_keys(business_embeddings_dict,social_embeddings_dict)

    for date,_ in business_embeddings_dict.items():
        print(date)
        distance_dict[date] = compute_frobenius_norm(business_embeddings_dict[date],social_embeddings_dict[date])

    return distance_dict


