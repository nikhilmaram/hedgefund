"""Functions that needs to run all the files"""
from typing import List
from typing import Tuple
import os
import pandas as pd
import config as cfg
import multiprocessing
import time
from datetime import datetime

import employee
import misc
import network
import sentiment

pd.set_option('display.max_colwidth', -1)

# =========================================================================
# ==================== Compute necessary dictionaries =====================
# =========================================================================
address_to_user_dict,user_to_address_dict = employee.map_user_address(cfg.ADDRESS_LINK_FILE)
employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
employee_list = list(employee_dict.keys())
# =========================================================================

# =========================================================================
# ==================== Modiying im_dfs to contain necessary information====
# =========================================================================

def modify_im_dfs(src_file_path : str, dst_file_path : str):
    """Modify im_dfs to contain necessary information.

    im_df's contain sender_buddy, sender_address etc... which are irrelevant information.
    create new column with names(lastname_firstname) for sender_receiver.
    create new column stating whether the user is in the network or outside newtork.
    If user is in Employee Master.xlsx then he is in the network else he is outside network.

    Args:
        src_file_path : source path of IM file.
        dst_file_path : Destination path where IM Dataframe has to output.
    """

    im_df = pd.read_csv(src_file_path)

    im_df["sender_user_name"] = im_df["sender_buddy"].apply(lambda x: employee.lambda_func_user_address_mapping(x))
    im_df["receiver_user_name"] = im_df["receiver_buddy"].apply(lambda x: employee.lambda_func_user_address_mapping(x))

    im_df["sender_in_network"] = im_df["sender_user_name"].apply(lambda x: employee.lambda_func_user_in_network(x))
    im_df["receiver_in_network"] = im_df["receiver_user_name"].apply(lambda x: employee.lambda_func_user_in_network(x))

    im_df["day"] = im_df["time_stamp"].apply(lambda x: datetime.strptime(x, '%m-%d-%yT%H:%M:%S').strftime("%m-%d-%Y"))
    im_df = im_df[["sender_user_name","receiver_user_name","content","time_stamp","day","sender_in_network","receiver_in_network","classify","sentiment"]]

    im_df.to_csv(dst_file_path,index=False)

def modify_im_dfs_filelist(file_list : List[str], src_dir_path:str,dst_dir_path : str):
    """ Runs modify_im_dfs function for each file in the filelist.

    Args:
        file_list           : List of files.
        src_dir_path        : path to the files present in file_list
        dst_dir_path        : path to where output files has to be written.

    Returns:
        None
    """
    for file_name in file_list:
        if file_name.startswith("im_df"):
            src_file_path = os.path.join(src_dir_path, file_name)
            dst_file_path = os.path.join(dst_dir_path, file_name)
            modify_im_dfs(src_file_path,dst_file_path)

def modify_im_dfs_filelist_multiprocess(src_dir_path:str,dst_dir_path : str,num_process : int):
    """Runs the modify_im_dfs_filelist on multiple process.

    Args:
        src_dir_path        : path to the files present in file_list
        dst_dir_path        : path to where output files has to be written.
        num_process         : Number of process.
    """
    list_of_file_list = misc.splitting_all_files(src_dir_path,num_process)

    for file_list in list_of_file_list:
        print(file_list)
        p = multiprocessing.Process(target=modify_im_dfs_filelist, args=(file_list,src_dir_path,dst_dir_path))
        p.start()

# =========================================================================
# ================ Generating weekly kcore files===========================
# =========================================================================

def compute_kcore_values_filelist(file_list:List[str], src_dir_path:str, dst_dir_path : str):
    """ Runs compute_kcore_values function for each file in the filelist.

    Args:
        file_list           : List of files.
        src_dir_path        : path to the files present in file_list
        dst_dir_path        : path to where output files has to be written.

    Returns:
        None
    """
    for file_name in file_list:
        if file_name.startswith("im_df"):
            print(file_name)
            week_num = file_name.split('.')[0][10:]
            ## input file_path
            file_path = os.path.join(src_dir_path, file_name)
            message_matrix, _ ,_ = network.create_matrix(file_path,in_network=True)
            G = network.create_graph(message_matrix)
            kcore_number_list, kcore_num_of_nodes_list, kcore_num_components_list, kcore_largest_cc_num_nodes_list = network.compute_kcore_values_list(G,25)

            output_path = dst_dir_path + "/kcore_number_week{0}.csv".format(week_num)
            misc.writing_list_into_file(kcore_number_list,output_path)

            output_path = dst_dir_path + "/kcore_num_of_nodes_week{0}.csv".format(week_num)
            misc.writing_list_into_file(kcore_num_of_nodes_list, output_path)

            output_path = dst_dir_path + "/kcore_num_components_week{0}.csv".format(week_num)
            misc.writing_list_into_file(kcore_num_components_list, output_path)

            output_path = dst_dir_path + "/kcore_largest_cc_num_nodes_week{0}.csv".format(week_num)
            misc.writing_list_into_file(kcore_largest_cc_num_nodes_list, output_path)

def compute_kcore_values_filelist_multiprocess(src_dir_path:str,dst_dir_path : str,num_process : int):
    """Runs the compute_kcore_values_filelist on multiple process.

    Args:
        src_dir_path        : path to the files present in file_list
        dst_dir_path        : path to where output files has to be written.
        num_process         : Number of process.
    """
    list_of_file_list = misc.splitting_all_files(src_dir_path,num_process)

    for file_list in list_of_file_list:
        p = multiprocessing.Process(target=compute_kcore_values_filelist, args=(file_list,src_dir_path,dst_dir_path))
        p.start()


