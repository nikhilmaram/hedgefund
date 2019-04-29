""" This file contains functions which are used to check balance theory."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List
from typing import Tuple
from datetime import  datetime,timedelta,date
import os
import seaborn
import matplotlib.pyplot as plt

import employee
import config as cfg
import misc
import sentiment
import balance_theory_src_code as balance_theory_src
import balance_theory_sparse_src_network_utils as balance_theory_sparse
import network
import plot
import performance
import relationships

# =========================================================================
# ==================== Compute necessary dictionaries =====================
# =========================================================================
address_to_user_dict,user_to_address_dict = employee.map_user_address(cfg.ADDRESS_LINK_FILE)
employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
employee_list = list(employee_dict.keys())
employee_id_to_username_dict,employee_username_to_id_dict = employee.employee_id_to_username_from_file(cfg.EMPLOYEE_MASTER_FILE)

# =========================================================================
# ==================== Creating Message matrix===========================
# =========================================================================
def create_matrix_for_balance_theory(im_df : pd.DataFrame, in_network: bool = True, user_list : List = None):
    """ Creates Adjacency Matrix and Adjacency lists by reading the IM Data file.

    Args:
        im_df       : IM dataframe.
        in_network  : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)
        user_list   : User list for which edges are considered.
    Returns:
        sentiment_matrix : Adjacency matrix created with each row corresponding to the user. 1-297 indexes are Employees in the network.
        id_to_username_dict : index to username dictionary (since outside users are possible in the network).


    297 Employees are present in the hedgefund. A row is assigned to them, irrespective of employee being in the message file of the week.
    An intermediate sentiment matrix is calculated which contains the sentiment values between users.
    sentiment_matrix[i][j] corresponds to resultant sentiment between i and j.

    """

    ## start with in_network users
    id_to_username_dict = employee_id_to_username_dict
    username_to_id_dict = employee_username_to_id_dict


    if user_list is not None:
        ## get the dataframe for the chats corresponds to users in userlist.
        im_df = im_df[im_df["sender_user_name"].isin(user_list) | im_df["receiver_user_name"].isin(user_list)]

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
        idx_count = cfg.TOTAL_EMPLOYEES
        for user in total_outside_users:
            idx_count = idx_count + 1
            id_to_username_dict[idx_count] = user
            username_to_id_dict[user] = idx_count
        user_count = idx_count



    sentiment_matrix = [[0 for x in range(user_count + 1)] for y in range(user_count + 1)] ## indexing from 1.
    sentiment_list_matrix = [[[] for x in range(user_count + 1)] for y in range(user_count + 1)]

    print(len(im_df))
    for index, row in im_df.iterrows():
        sender_idx = username_to_id_dict[row['sender_user_name']]
        receiver_idx = username_to_id_dict[row['receiver_user_name']]
        message_sentiment = row['sentiment']
        sentiment_list_matrix[sender_idx][receiver_idx].append(message_sentiment)

    ## Normalize the sentiment values
    for i in range(user_count + 1):
        for j in range(user_count + 1):
            sentiment_list = sentiment_list_matrix[i][j]
            if(len(sentiment_list) > 0):
                resultant_sentiment_value = sentiment.resultant_sentiment(sentiment_list)
                sentiment_matrix[i][j] = resultant_sentiment_value

    del sentiment_list_matrix
    return sentiment_matrix,id_to_username_dict


def create_graph_for_balance_theory(sentiment_matrix, include_all_nodes :bool = False) -> nx.DiGraph:
    """Creates a directed graph given a dataframe for validating the balance theory.

     Args:
        sentiment_matrix : Adjacency matrix(captures sentiment) for messages .
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message
    Returns:
        Directed Graph.

    We use create_matrix_for_balance_theory to get the matrix .

    """
    len_matrix = len(sentiment_matrix)
    G = nx.DiGraph()
    if include_all_nodes:
        G.add_nodes_from(range(1, len_matrix))

    for src in range(len(sentiment_matrix)):
        for dest in range(len(sentiment_matrix)):
            if (sentiment_matrix[src][dest] != 0):
                # print(sentiment_matrix[src][dest])
                G.add_edge(src, dest, weight=sentiment_matrix[src][dest])
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def compute_balance_theory_value_given_df(im_df:pd.DataFrame, in_network: bool = True, user_list : List = None,
                                          include_all_nodes :bool = False, balance_type:int = 1) -> Tuple[float,float,float]:
    """Computes balance theory values given dataframe.

    Args:
        im_df       : IM dataframe.
        in_network  : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)
        user_list   : User list for which edges are considered.
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message
        balance_type    : balance type.
    Returns:
        ratio, balanced, unbalanced
    """

    # sentiment_matrix, id_to_username_dict = create_matrix_for_balance_theory(im_df, in_network=in_network, user_list = user_list)
    # G = create_graph_for_balance_theory(sentiment_matrix, include_all_nodes=include_all_nodes)


    message_matrix, _, _ = network.create_matrix(im_df, in_network= in_network, user_list=user_list)
    G = create_graph_for_balance_theory(message_matrix, include_all_nodes=include_all_nodes)

    # ratio, balanced, unbalanced = balance_theory_src.sprase_balance_ratio(G, balance_type)

    # ratio = balance_theory_sparse.terzi_sprase_balance_ratio(G, undirected=False)
    # ratio, balanced, unbalanced = balance_theory_sparse.kunegis_sprase_balance_ratio(G, undirected=False)
    # ratio, balanced, unbalanced = balance_theory_sparse.sprase_balance_ratio(G, balance_type)

    ratio, balanced, unbalanced = balance_theory_sparse.classical_balance_ratio(G, balance_type)

    return ratio, balanced, unbalanced
    # return ratio, 0, 0


def compute_balance_theory_given_file_list(src_dir_path : str, start_week:int, end_week:int,balance_type:int = 1,
                                           in_network: bool = True, user_list : List = None, include_all_nodes :bool = False):
    """Computes balance theory values given start and end of the weeks.

    Args:
        src_dir_path: source directory path.
        start_week  : start week
        end_week    : end week.
        balance_type: balance type.
        in_network  : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)
        user_list   : User list for which edges are considered.
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message

    Returns:
        ratio_dict : Balanced Ratio dictionary. key : date, value :  balance ratio.
        balanced_dict : number of balance triads dictionary. key : date, value :  number of balance triads.
        unbalanced_dict : number of unbalance triads dictionary. key : date, value :  number of unbalance triads.

    """
    file_list = misc.splitting_all_files(src_dir_path,1, start_week, end_week)[0]
    ratio_dict = {} ; balanced_dict = {} ; unbalanced_dict = {}
    for file_name in file_list:
        # print(file_name)
        file_path = os.path.join(src_dir_path,file_name)
        df = pd.read_csv(file_path)
        week_num = int(file_name.split('.')[0].split('_')[-1][4:])
        curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
        ratio, balanced, unbalanced = compute_balance_theory_value_given_df(df,in_network,user_list=user_list,
                                                                            include_all_nodes=include_all_nodes,balance_type=balance_type)
        ratio_dict[curr_date] = ratio
        balanced_dict[curr_date] = balanced
        unbalanced_dict[curr_date] = unbalanced


    return ratio_dict, balanced_dict, unbalanced_dict

# =========================================================================
# ==================== Check Causality for balance theory =================
# =========================================================================
def check_cauasality_balance_theory_performance(src_dir_path : str, start_week:int, end_week:int,balance_type:int = 1,
                                           in_network: bool = True, user_list : List = None, include_all_nodes :bool = False):
    """Checks causality in balance theory and performance.
    Args:
        src_dir_path: source directory path.
        start_week  : start week
        end_week    : end week.
        balance_type: balance type.
        in_network  : Boolean Variable determines whether the graph should be built within the hedgefund network. (True : In , False: All)
        user_list   : User list for which edges are considered.
        include_all_nodes : whether to include nodes in the graph, which doesnt share a message
    """
    ratio_dict, balanced_dict, unbalanced_dict = compute_balance_theory_given_file_list(src_dir_path,start_week=start_week, end_week= end_week,
                                                                                        balance_type=balance_type, in_network=in_network,
                                                                                        user_list=user_list, include_all_nodes=include_all_nodes)

    ratio_dict = misc.change_key_string_key_date(ratio_dict)
    misc.write_dict_in_file(ratio_dict,cfg.BALANCE_PKL_FILES+"/fully_connected_cartwright_harary_123_200.pkl")
    plot_balance_theory(ratio_dict)
    if user_list is None:
        book_list = misc.read_book_file(cfg.BOOK_FILE)
    else:
        book_list = employee.books_given_employee_list(user_list)
    dates_dict, performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE, book_list,
                                                                           start_week=start_week,
                                                                           end_week=end_week, only_week=True)
    performance_date_dict = performance.combine_performance_given_book_list(dates_dict, performance_dict,
                                                                            only_week=True)

    performance_date_dict_common, ratio_dict_common = misc.common_keys(performance_date_dict,
                                                                       ratio_dict)
    print(len(performance_date_dict_common), len(ratio_dict_common))
    performance_list, ratio_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,
                                                                             ratio_dict_common)

    balance_theory_causality_dict = relationships.compute_causality(performance_list, ratio_list, 5)
    misc.print_causality_dict(balance_theory_causality_dict)

# =========================================================================
# ==================== Plot Balance Theory===========================
# =========================================================================
def plot_balance_theory(ratio_dict:dict):
    """Plot Balance theory given ratio dict.

    Args:
        ratio_dict : Ratio Dictionary.

    Returns:
        None.
    """
    ratio_list = []
    for date in sorted(ratio_dict.keys()):
        ratio_list.append(ratio_dict[date])
    date_list = list(sorted(ratio_dict.keys()))
    plot.plot_list_vs_dates(date_list, ratio_list, xlabel="Time", ylabel= "Balance Ration",
                            title="Balance Ratio over Time", legend_info="Balance Theory")

# =========================================================================
# ==================== Analyse Regplot Balance Theory======================
# =========================================================================
def regplot_balance_theory(pkl_file):
    """Reads a pkl file and plots a regplot for balance theory.

    Args:
        pkl_file : Input

    Returns:
        None

    """
    inp_dict = misc.read_file_into_dict(pkl_file)
    sorted_dates_list = list(sorted(inp_dict.keys()))
    balance_list = []
    for sorted_date in sorted_dates_list:
        balance_list.append(inp_dict[sorted_date])

    # plot.plot_list_vs_dates(sorted_dates_list, balance_list, xlabel="Time", ylabel="Balance", title="Balance over Time", legend_info="")
    x_range = np.array(list(range(1, len(sorted_dates_list)+1)))[:-10]
    # x_range = np.array([x.toordinal() for x in sorted_dates_list])
    balance_list = np.array(balance_list)[:-10]
    print(x_range)

    seaborn.regplot(x_range, balance_list,order=2)
    plt.xlabel("Time")
    plt.ylabel("Balance Ratio")
    title = pkl_file.split('/')[-1][:-4]
    plt.title(title)
    plt.show()

    # print(inp_dict)

def plot_balance_performance():
    balance_dict = misc.read_file_into_dict(cfg.BALANCE_PKL_FILES + "/fully_connected_cartwright_harary.pkl")
    subordinates_list = list(employee.employee_to_account_dict.keys())
    book_list = employee.books_given_employee_list(subordinates_list)
    dates_dict, performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE, book_list, 123, 160,
                                                                           True)
    print(dates_dict)
    print(performance_dict)
    performance_date_dict = performance.combine_performance_given_book_list(dates_dict, performance_dict, True)
    print(performance_date_dict)
    performance_common, balance_common = misc.common_keys(performance_date_dict, balance_dict)
    performance_list, balance_list = misc.get_list_from_dicts_sorted_dates(performance_common, balance_common)

    causal_dict = relationships.compute_causality(balance_list, performance_list)
    misc.print_causality_dict(causal_dict)
    dates_list = list(sorted(performance_common.keys()))


    plot.plot_two_graphs_in_single_plot(dates_list, y_cause=balance_list, y_effect=performance_list,
                                        ycause_label="Balance of Network", yeffect_label="Performance of Network", xlabel= "Time",
                                        legend_cause="Balance", legend_effect="Performance", title = "Balance vs Performance of Network",
                                        lag=0)

    plot.plot_two_graphs_in_single_plot(dates_list, y_cause=balance_list, y_effect=performance_list,
                                        ycause_label="Balance of Network", yeffect_label="Performance of Network",xlabel="Time",
                                        legend_cause="Balance with lag 4", legend_effect="Performance",
                                        title="Balance vs Performance of Network",
                                        lag=4)

    # plot.plot_two_graphs_in_single_plot(dates_list, y_cause=balance_list, y_effect=performance_list,
    #                                     ycause_label="Balance of Network", yeffect_label="Performance of Network",
    #                                     lag=4)

if __name__ == "__main__":
    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL + "/im_df_week124.csv")
    # ratio, balanced, unbalanced = compute_balance_theory_value_given_df(im_df, in_network=True, user_list=None,
    #                                                                     include_all_nodes=False, balance_type= 1)
    # print(ratio, balanced, unbalanced)

    # ratio_dict, balanced_dict, unbalanced_dict = compute_balance_theory_given_file_list(cfg.SENTIMENT_BUSINESS, start_week=123, end_week=150,
    #                                                                                     balance_type= 1)
    # plot_balance_theory(ratio_dict)

    # =========================================================================
    # ==================== Causality between balance theory =================
    # =========================================================================

    # check_cauasality_balance_theory_performance(cfg.SENTIMENT_BUSINESS, start_week= 123, end_week= 200, balance_type= 1)

    # =========================================================================
    # ==================== Regplot balance theory =================
    # =========================================================================

    # regplot_balance_theory(cfg.BALANCE_PKL_FILES + "/fully_connected_cartwright_harary.pkl")
    regplot_balance_theory(cfg.BALANCE_PKL_FILES+"/fully_connected_cartwright_harary_123_200.pkl")
    regplot_balance_theory(cfg.BALANCE_PKL_FILES+"/performance_week_123_200.pkl")
    # regplot_balance_theory(cfg.BALANCE_PKL_FILES + "/fully_connected_clusterting.pkl")
    # regplot_balance_theory(cfg.BALANCE_PKL_FILES + "/fully_connected_transitivity.pkl")

    # regplot_balance_theory(cfg.BALANCE_PKL_FILES + "/sparse_cartwright_harary.pkl")
    # regplot_balance_theory(cfg.BALANCE_PKL_FILES + "/sparse_clustering.pkl")
    # regplot_balance_theory(cfg.BALANCE_PKL_FILES + "/sparse_transitivity.pkl")

    # plot_balance_performance()
    pass