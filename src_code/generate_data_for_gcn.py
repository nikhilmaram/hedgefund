"Contains all the functions related to generate the data for GCN. "

import pandas as pd
from datetime import datetime
from typing import List
from typing import Tuple
import numpy as np
import queue
import multiprocessing
import os
from scipy import sparse
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from random import sample
import time
import networkx as nx

import employee
import sentiment
import config as cfg
import misc
import network


def create_df(src_dir_path: str, start_week: int=123, end_week:int= 200):
    """Create a dataframe which contains messages from start week to end week.

    Args:
        src_dir_path : source directory path.
        start_week   : start week.
        end_week     : end week.

    Returns:
        im_df        : Output dataframe.
    """

    file_list = misc.splitting_all_files(src_dir_path, num_process=1, start_week=start_week, end_week=end_week)[0]
    im_df = pd.DataFrame()
    for file_name in file_list:
        start_time = time.time()
        week_num = file_name.split('.')[0][10:]
        file_path = os.path.join(src_dir_path, file_name)
        curr_df = pd.read_csv(file_path)
        im_df = pd.concat([im_df, curr_df])
        print("File Name: {0}, Time: {1}".format(file_name, time.time() - start_time))
    return im_df


def write_numpy_ndarray_into_sparse_pkl_file(inp_vector, output_file_path):
    """Converts numpy ndarray into sparse vector and writes into pkl file.

    Args:
        inp_vector          : Input vector.
        output_file_path    : Output file to be written into.
    """
    sparse_inp_vector = sparse.csr_matrix(inp_vector)
    with open(output_file_path,"wb") as handle:
        pickle.dump(sparse_inp_vector,handle)

def write_numpy_ndarray_into_pkl_file(inp_vector, output_file_path):
    """Converts numpy ndarray into sparse vector and writes into pkl file.

    Args:
        inp_vector          : Input vector.
        output_file_path    : Output file to be written into.
    """

    with open(output_file_path,"wb") as handle:
        pickle.dump(inp_vector,handle)

# ==============================================================================================
# ==============================Generate GCN Label data ========================================
# ==============================================================================================
def generate_gcn_label_data():
    """Generate labels for training GCN.

    Returns:
        user_label_list = user label list based on title.
    """
    user_count = cfg.TOTAL_EMPLOYEES
    employee_list = employee.employee_list
    employee_dict = employee.employee_dict
    employee_username_to_id_dict = employee.employee_username_to_id_dict

    user_label_list = np.zeros((user_count, 4))
    title_to_pos_dict = {"trader":0, "research_analyst":1, "portfolio_manager":2, "other":3}

    traders = 0; research_analyst = 0; portfolio_manager = 0; other = 0
    for user in employee_list:
        user_obj = employee_dict[user]
        user_id  = employee_username_to_id_dict[user]
        title = user_obj.title.lower()
        if "trad" in title:
            # print(title)
            traders = traders + 1
            user_label_list[user_id-1][title_to_pos_dict["trader"]] = 1
        elif "research analyst" in title:
            # print(title)
            research_analyst = research_analyst + 1
            user_label_list[user_id - 1][title_to_pos_dict["research_analyst"]] = 1
        elif "portfolio manager" in title:
            # print(title)
            portfolio_manager = portfolio_manager + 1
            user_label_list[user_id - 1][title_to_pos_dict["portfolio_manager"]] = 1
        else:
            # print(title)
            other = other + 1
            user_label_list[user_id - 1][title_to_pos_dict["other"]] = 1

    print("Traders: {0}, Research Analyst: {1}, Portfolio Manager: {2}, Other: {3}".format(traders, research_analyst, portfolio_manager, other))
    write_numpy_ndarray_into_pkl_file(user_label_list, cfg.GCN_DATA_DIR + "/ind.hedgefund.ally")
    return user_label_list


# ==============================================================================================
# ==============================Generate GCN Feature data ======================================
# ==============================================================================================

def generate_gcn_feature_data(im_df:pd.DataFrame, in_network:bool = True) -> np.ndarray:
    """Generate the features required for training the GCN.

    Args:
        im_df : Input dataframe.
        in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: outside)

    Returns:
        feature_vector : feature vector
    """
    user_count =  cfg.TOTAL_EMPLOYEES
    user_sent_sentiment_list = np.zeros(user_count)
    user_recv_sentiment_list = np.zeros(user_count)
    user_working_days        = np.zeros(user_count)
    # user_supervisor_list     = []

    employee_list = employee.employee_list
    employee_username_to_id_dict = employee.employee_username_to_id_dict

    for user in employee_list:
        sent_sentiment, recv_sentiment = sentiment.sentiment_given_user(im_df,user_name=user, complete_network=False, in_network= in_network)
        user_id = int(employee_username_to_id_dict[user])
        user_sent_sentiment_list[user_id-1] = sent_sentiment
        user_recv_sentiment_list[user_id-1] = recv_sentiment

    feature_vector = np.column_stack((user_sent_sentiment_list, user_recv_sentiment_list))
    write_numpy_ndarray_into_sparse_pkl_file(feature_vector, cfg.GCN_DATA_DIR + "/ind.hedgefund.allx")
    return feature_vector


# =========================================================================
# ====================Generate GCN graph data===========================
# =========================================================================

def generate_gcn_graph_data(im_df, in_network:bool = True):
    """Generates Graph data for GCN.

    Args:
        im_df : input dataframe.
        in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: outside)

    """
    _, message_adj_list, _ = network.create_matrix(im_df, in_network=in_network)
    adj_dict = defaultdict()
    for idx, adj_list in enumerate(message_adj_list):
        if idx > 0:
            adj_list = list(adj_list)
            ## Reduce by 1 so that indexing start from 0.
            adj_list = [x-1 for x in adj_list]
            adj_dict[idx-1] = adj_list
    # print(message_adj_list)
    # print(adj_dict)
    misc.write_dict_in_file(adj_dict, cfg.GCN_DATA_DIR + "/ind.hedgefund.graph")



def generate_gcn_graph_data_sparse_matrix(im_df, output_file_path:str, in_network:bool = True):
    """Generates the graph from dataframe and saves it as a sparse matrix.

    Args:
        im_df : input dataframe.
        in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: outside)
    """

    message_matrix, message_adj_list, _ = network.create_matrix(im_df, in_network=in_network)
    G = network.create_graph(message_matrix, un_directed=True)
    sparse_matrix_G = nx.to_scipy_sparse_matrix(G, weight='weight')
    print(type(sparse_matrix_G))

    with open(output_file_path,"wb") as handle:
        pickle.dump(sparse_matrix_G,handle)



def test_indices_for_training(test_size:float = 0.25):
    """To generate test indices for training. Writes the test indices into a file."""

    user_count = cfg.TOTAL_EMPLOYEES
    employee_id_to_username_dict = employee.employee_id_to_username_dict
    employee_dict = employee.employee_dict

    title_to_pos_dict = {"trader": 0, "research_analyst": 1, "portfolio_manager": 2, "other": 3}
    traders = 0; research_analyst = 0; portfolio_manager = 0; other = 0

    labels_list = []
    indices_list = list(range(user_count))

    for i in range(1, user_count+1):
        employee_user_name = employee_id_to_username_dict[i]
        title = employee_dict[employee_user_name].title.lower()

        if "trad" in title:
            # print(title)
            traders = traders + 1
            labels_list.append(title_to_pos_dict["trader"])
        elif "research analyst" in title:
            # print(title)
            research_analyst = research_analyst + 1
            labels_list.append(title_to_pos_dict["research_analyst"])
        elif "portfolio manager" in title:
            # print(title)
            portfolio_manager = portfolio_manager + 1
            labels_list.append(title_to_pos_dict["portfolio_manager"])
        else:
            # print(title)
            other = other + 1
            labels_list.append(title_to_pos_dict["other"])

    train_indices, test_indices, train_labels, test_labels = train_test_split(indices_list, labels_list, test_size=test_size, stratify=labels_list)
    with open(cfg.GCN_DATA_DIR + "/ind.hedgefund.test.index", "w") as myfile:
        for ele in test_indices:
            myfile.write("%s\n" % ele)
    myfile.close()

    return train_indices, test_indices

def generate_test_feature_label_data(feature_vector, user_label_list):
    """ Creates the sparse matrix files required for GCN training.

    Args:
        feature_vector : feature vector.
        user_label_list : labels.
    """

    write_numpy_ndarray_into_sparse_pkl_file(feature_vector, cfg.GCN_DATA_DIR + "/ind.hedgefund.allx")
    write_numpy_ndarray_into_pkl_file(user_label_list, cfg.GCN_DATA_DIR + "/ind.hedgefund.ally")

# def generate_test_feature_label_data(feature_vector, user_label_list, num_test_samples:int = 47):
#     num_training_samples = cfg.TOTAL_EMPLOYEES - num_test_samples
#
#     train_feature_vector = feature_vector[:num_training_samples,:]
#     train_user_label_list = user_label_list[:num_training_samples,:]
#
#     test_feature_vector = feature_vector[num_training_samples:,:]
#     test_user_label_list = user_label_list[num_training_samples:,:]
#
#     write_numpy_ndarray_into_sparse_pkl_file(train_feature_vector, cfg.GCN_DATA_DIR + "/ind.hedgefund.x")
#     write_numpy_ndarray_into_sparse_pkl_file(train_feature_vector, cfg.GCN_DATA_DIR + "/ind.hedgefund.allx")
#     write_numpy_ndarray_into_sparse_pkl_file(test_feature_vector, cfg.GCN_DATA_DIR + "/ind.hedgefund.tx")
#
#     write_numpy_ndarray_into_pkl_file(train_user_label_list, cfg.GCN_DATA_DIR + "/ind.hedgefund.y")
#     write_numpy_ndarray_into_pkl_file(train_user_label_list, cfg.GCN_DATA_DIR + "/ind.hedgefund.ally")
#     write_numpy_ndarray_into_pkl_file(test_user_label_list, cfg.GCN_DATA_DIR + "/ind.hedgefund.ty")
#
#     test_indices = list(range(num_training_samples, cfg.TOTAL_EMPLOYEES))
#
#     with open(cfg.GCN_DATA_DIR + "/ind.hedgefund.test.index", "w") as myfile:
#         for ele in test_indices:
#             myfile.write("%s\n" % ele)
#     myfile.close()

def check_the_label_split(inp_list):
    """Checks the label split for input indices list.

    Args:
        inp_list : Input list which contains indices.

    Returns:
        None
    """
    employee_id_to_username_dict = employee.employee_id_to_username_dict
    employee_dict = employee.employee_dict
    traders = 0; research_analyst = 0; portfolio_manager = 0; other = 0
    for i in inp_list:
        employee_name = employee_id_to_username_dict[i+1] ## i+1 because indices start from 0, where as id of the employee from 1.
        employee_obj = employee_dict[employee_name]
        title = employee_obj.title.lower()
        if "trad" in title:
            # print(title)
            traders = traders + 1

        elif "research analyst" in title:
            # print(title)
            research_analyst = research_analyst + 1

        elif "portfolio manager" in title:
            # print(title)
            portfolio_manager = portfolio_manager + 1

        else:
            # print(title)
            other = other + 1


    print("Traders: {0}, Research Analyst: {1}, Portfolio Manager: {2}, Other: {3}".format(traders, research_analyst,
                                                                                           portfolio_manager, other))

def check_random_assignment_accuracy(user_label_list):
    random_sampling = np.random.choice([0,1,2,3], cfg.TOTAL_EMPLOYEES,p=[0.15, 0.3, 0.21, 0.34])
    random_sampling_one_hot = np.zeros((cfg.TOTAL_EMPLOYEES,4))
    random_sampling_one_hot[np.arange(cfg.TOTAL_EMPLOYEES),random_sampling] = 1
    count = 0.0
    for i in range(cfg.TOTAL_EMPLOYEES):
        if (np.array_equal(user_label_list[i], random_sampling_one_hot[i])):
            # print(user_label_list[i], random_sampling_one_hot[i])
            count = count + 1

    print(count/cfg.TOTAL_EMPLOYEES)





if __name__ == "__main__":
    # ==============================================================================================
    # ===========Generates  Data frame=================================
    # ==============================================================================================
    # im_df = pd.read_csv(cfg.SENTIMENT_BUSINESS+"/im_df_week128.csv")
    im_df = create_df(cfg.SENTIMENT_BUSINESS, start_week= 1, end_week= 263)

    # ==============================================================================================
    # ===========Generates feature, label, graph and test indice data=================================
    # ==============================================================================================
    feature_vector = generate_gcn_feature_data(im_df, True)
    #
    user_label_list = generate_gcn_label_data()

    generate_gcn_graph_data(im_df, True)
    generate_gcn_graph_data_sparse_matrix(im_df, cfg.GCN_DATA_DIR+"/ind.hedgefund_sparse.graph")

    train_indices, test_indices = test_indices_for_training(test_size=0.25)
    # check_the_label_split(train_indices)
    # check_the_label_split(test_indices)

    # ==============================================================================================
    # ==============Random sampling============================
    # ==============================================================================================
    # check_random_assignment_accuracy(user_label_list)
    pass