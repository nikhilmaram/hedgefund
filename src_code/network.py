""" This file contains functions which are used to create network."""
import numpy as np
import pandas as pd
import networkx as nx
from typing import List
from typing import Tuple
from datetime import  datetime,timedelta,date
import os

import employee
import config as cfg
import misc


# =========================================================================
# ==================== Compute necessary dictionaries =====================
# =========================================================================
address_to_user_dict,user_to_address_dict = employee.map_user_address(cfg.ADDRESS_LINK_FILE)
employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
employee_list = list(employee_dict.keys())
employee_id_to_username_dict,employee_username_to_id_dict = employee.employee_id_to_username_from_file(cfg.EMPLOYEE_MASTER_FILE)

# =========================================================================
# ==================== Creating Graph and K-Core===========================
# =========================================================================

def create_matrix(file_path : str, in_network: bool = True) -> Tuple[np.ndarray,List[set],dict]:
    """ Creates Adjacency Matrix and Adjacency lists by reading the IM Data file.

    Args:
        file_path : IM file path.
        in_network : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)

    Returns:
        Matrix : Adjacency matrix created with each row corresponding to the user. 1-297 indexes are Employees in the network.
        List : Adjacency List with each element corresponding to the user.
        Dict : index to username dictionary (since outside users are possible in the network).


    297 Employees are present in the hedgefund. A row is assigned to them, irrespective of employee being in the message file of the week.
    """

    im_df = pd.read_csv(file_path)

    ## start with in_network users
    id_to_username_dict = employee_id_to_username_dict
    username_to_id_dict = employee_username_to_id_dict
    user_count = cfg.TOTAL_EMPLOYEES

    ### Filter, outside hedgefund users from the data.
    if in_network:
        im_df = im_df[im_df["sender_in_network"]==1]
        im_df = im_df[im_df["receiver_in_network"] == 1]
        user_count = cfg.TOTAL_EMPLOYEES
    else:
        users_in_curr_file = im_df['sender_user_name'].append(im_df['receiver_user_name']).unique().tolist()
        ## Get only outside users. sort them alphabetically.
        total_outside_users = sorted(list(set(users_in_curr_file) - set(employee_list)))
        user_count = cfg.TOTAL_EMPLOYEES + len(total_outside_users)
        ## Fill the dictionary with outside users.
        idx_count = cfg.TOTAL_EMPLOYEES + 1
        for user in total_outside_users:
            id_to_username_dict[idx_count] = user
            username_to_id_dict[user] = idx_count
            idx_count = idx_count + 1

    message_matrix = np.zeros((user_count + 1, user_count + 1)) ## indexing from 1.
    message_adj_list = [set() for _ in range(user_count + 1)]

    for index, row in im_df.iterrows():
        sender_idx = username_to_id_dict[row['sender_user_name']]
        receiver_idx = username_to_id_dict[row['receiver_user_name']]
        message_matrix[sender_idx][receiver_idx] = message_matrix[sender_idx][receiver_idx] + 1
        message_adj_list[sender_idx].add(receiver_idx)

    return message_matrix,message_adj_list,id_to_username_dict

def create_graph(message_matrix, un_directed:bool = True, weight_threshold=1) -> nx.Graph :
    """Creates Graphs from the given message matrix.

    Args:
        message_matrix : Adjacency matrix for messages.
        un_directed : If true, returns undirected graph.
        weight_threshold : Threshold for the weight for edge to be considered.
    Returns:
        Directed Graph.

    We use create_matrix to get the matrix and adjacent list.
    """

    G = nx.DiGraph()

    for src in range(len(message_matrix)):
        for dest in range(len(message_matrix)):
            if(message_matrix[src][dest] >= weight_threshold):
                G.add_edge(src, dest,weight = message_matrix[src][dest])

    if un_directed:
        G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def compute_kcore(G: nx.Graph, kcore_num :int) -> nx.Graph:
    """Computes the kcore of the given graph.

    Args:
        G : Input Graph.
        kcore_num : Kcore Number.

    Returns:
        k-core of the given graph.
    """
    kcore_G = nx.k_core(G,kcore_num)
    return kcore_G

def compute_kcore_values(G:nx.Graph,kcore_num:int) -> Tuple[int,int,int,int,str]:
    """Computes various values associated with k-core(k = kcore_num) of the given graph G.

        Args:
            G : Input Graph
            max_k_core : Maximum K value of the K-core.
        Returns:
            kcore_number :  K-core number i.e value of K.
            kcore_num_of_nodes : contains number of nodes present in corresponding K-core.
            kcore_num_components : contains number of connected components in corresponding K-core.
            kcore_largest_cc_num_nodes : contains number of nodes present in the largest connected component of corresponding K-core.
            kcore_largest_cc_nodes : nodes present in largest connected component of K-core (list written as str).
        """
    kcore_G = compute_kcore(G, kcore_num)
    kcore_num_of_nodes = len(kcore_G.nodes)
    if (kcore_num_of_nodes == 0):
        return kcore_num,0,0,0,"0"
    subgraphs = nx.connected_component_subgraphs(kcore_G)
    kcore_num_components = len(list(subgraphs))
    kcore_largest_cc = max(nx.connected_component_subgraphs(kcore_G), key=len)
    kcore_largest_cc_num_nodes = len(kcore_largest_cc.nodes)
    kcore_largest_cc_nodes = kcore_largest_cc.nodes
    ## create a string from the list.
    kcore_largest_cc_nodes = [str(x) for x in kcore_largest_cc_nodes]
    kcore_largest_cc_nodes = "-".join(kcore_largest_cc_nodes)
    return kcore_num,kcore_num_of_nodes,kcore_num_components,kcore_largest_cc_num_nodes,kcore_largest_cc_nodes

def compute_kcore_values_list(G : nx.Graph, max_k_core:int) -> Tuple[List,List,List,List,List]:
    """Computes various values associated with k-core of the given graph G.

    Args:
        G : Input Graph
        max_k_core : Maximum K value of the K-core.
    Returns:
        kcore_number_list : contains K-core numbers i.e value of K.
        kcore_num_of_nodes_list : contains number of nodes present in corresponding K-core.
                                (i.e 2nd element corresponds to number of nodes in K-core where K value is 2nd element in kcore_number_list_
        kcore_num_components_list : contains number of connected components in corresponding K-core.
        kcore_largest_cc_num_nodes_list : contains number of nodes present in the largest connected component of corresponding K-core.
        kcore_largest_cc_nodes_list : contains nodes present in largest connected component of K-core.
    """

    kcore_number_list = [];  kcore_num_of_nodes_list = []
    kcore_num_components_list = [] ; kcore_largest_cc_num_nodes_list = [] ; kcore_largest_cc_nodes_list = []

    for kcore_num in range(max_k_core):

        kcore_num, kcore_num_of_nodes, kcore_num_components, kcore_largest_cc_num_nodes,kcore_largest_cc_nodes = compute_kcore_values(G,kcore_num)
        if (kcore_num_of_nodes == 0):
            break
        kcore_num_components_list.append(kcore_num_components)
        kcore_num_of_nodes_list.append(kcore_num_of_nodes)
        kcore_number_list.append(kcore_num)
        kcore_largest_cc_num_nodes_list.append(kcore_largest_cc_num_nodes)
        kcore_largest_cc_nodes_list.append(kcore_largest_cc_nodes)

        print("Number of {0}-core Nodes: {1}, Connected Components : {2}, largest CC Size: {3}"
              .format(kcore_num,kcore_num_of_nodes,kcore_num_components,kcore_largest_cc_num_nodes))

    return kcore_number_list, kcore_num_of_nodes_list, kcore_num_components_list, kcore_largest_cc_num_nodes_list,kcore_largest_cc_nodes_list




# =========================================================================
# ==================== Values for plots====================================
# =========================================================================

def compute_element_kcore_for_plots(dir_path:str, start_week :int, end_week :int,
                       element_filename_start : str,maximum_core : int = 25) -> Tuple[List,List]:
    """Computes values for plots.

    Args:
        dir_path : path to where files are present.
        start_week : starting week to be considered.
        end_week : ending week.
        element_filename_start : generates value based on filename starting.
    Returns:
        dates : list of dates
        y_list : kcore attributes for weeks ranging from start_week to end_week.
    """
    kcore_element_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for file_name in filenames:
            if file_name.startswith(element_filename_start):
                file_path = os.path.join(dir_path, file_name)
                week_num = int(file_name.split('.')[0].split('_')[-1][4:])
                kcore_element_dict[week_num] = {}
                ## Get the corresponding kcore_number_week file because it has the k value
                ## mapping string to integers
                kcore_num_file = os.path.join(dir_path, "kcore_number_week{0}.csv".format(week_num))
                kcore_num_list = list(map(int, open(kcore_num_file).readline().strip("\n").split(",")[:-1]))
                kcore_element_list = list(map(int, open(file_path).readline().strip("\n").split(",")[:-1]))
                ### {<week_num> : {<core_number>: <largest_cc_num_nodes>} }
                kcore_element_dict[week_num] = dict(zip(kcore_num_list, kcore_element_list))


    x = list(range(start_week, end_week+1))
    y_list = []
    for core_number in range(maximum_core):
        y = []
        for week in x:
            ## In case the core number is not present in the corresponding week.
            if (week in kcore_element_dict.keys()) and (core_number in kcore_element_dict[week].keys()):
                y.append(kcore_element_dict[week][core_number])
            else:
                y.append(0)
        y_list.append(y)

    dates = []
    start_date = date(2006, 8, 3)
    for i in range(start_week, end_week+1):
        dates.append(misc.calculate_date(start_date, i))

    return dates, y_list


if __name__ == "__main__":
    pass

