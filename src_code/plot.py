"""This file contains functions which are used for plotting."""

from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter,DayLocator,MONDAY
from datetime import  datetime,timedelta,date
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import queue
import random
import numpy as np
import math

import network
import employee
import config as cfg
import performance
import sentiment
import misc
import relationships
import liwc_categorization as liwc



def plot_list_of_lists_vs_dates(x :List,y_list : List[List],xlabel :str = "",ylabel: str="",title:str="",legend_info:List=[]):
    """Plots y (list of lists) against x (list).

    Args:
        x : x- variable which are dates.
        y_list : y_varaible (list of lists)
        xlabel : label for x-axis .
        ylabel : label for y-axis.
        title  : title for the plot.
        lengend_info : legend for the plot.
    """

    fig, ax = plt.subplots()
    for i in range(0, len(y_list)):
        # plt.plot(x, y_list[i],'-o', label='%d-core' % (i+1))
        ax.plot_date(x, y_list[i], '-o', label=legend_info[i])

    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    monthsFmt = DateFormatter("%b '%y")
    # every monday
    mondays = WeekdayLocator(MONDAY)

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()

    mpl.rcParams["legend.loc"] = 'upper left'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    # plt.draw()


def plot_list_vs_dates(x :List,y: List,xlabel :str,ylabel: str,title:str,legend_info:str):
    """Plots y (list) against x (list).

    Args:
        x : x- variable which are dates.
        y_list : y_varaible (list : performance, etc....)
        xlabel : label for x-axis .
        ylabel : label for y-axis.
        title  : title for the plot.
        lengend_info : legend for the plot.
    """
    fig, ax = plt.subplots()
    ax.plot_date(x,y,'-o',label=legend_info)

    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")
    # every monday
    mondays = WeekdayLocator(MONDAY)
    days = DayLocator(bymonthday=range(1, 30), interval=1)

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    # ax.xaxis.set_minor_locator(days)
    ax.autoscale_view()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def general_plot(x:List, y:List,xlabel:str="",ylabel:str="",title:str="",legend=""):
    """General plot.

    Args:
        x : x- variable.
        y : y_variable.
        xlabel : label for x-axis .
        ylabel : label for y-axis.
        title  : title for the plot.
        lengend_info : legend for the plot.
    """

    plt.plot(x,y,label = legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

# =========================================================================
# ====================Plotting hierarchy===================================
# =========================================================================

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_employee_hierarchy(top_employee :employee.Employee, employee_dict : dict, file_path :str,title_in_label :bool= True):
    """Plots hierachical structure  starting from the given top employee.

    Args:
        top_employee : Given Top Employee.
        employee_dict : employee dictionary.
        file_path : path to the file where the pic needs to be stored.
        title_in_label : If title needs to be included in the node label.

    Returns:
        None
    """
    G = nx.DiGraph()
    # G.add_node(top_employee.user_name)

    hierarchy_queue = queue.Queue()
    hierarchy_queue.put(top_employee.user_name)
    while(not hierarchy_queue.empty()):
        curr_emp_name = hierarchy_queue.get()
        for subordinate in employee_dict[curr_emp_name].immediate_subordinates:
            if title_in_label:
                curr_emp_label = employee_dict[curr_emp_name].title + ":" + curr_emp_name
                subordinate_label = employee_dict[subordinate].title + ":" + subordinate
                G.add_edge(curr_emp_label,subordinate_label)
            else:
                G.add_edge(curr_emp_name,subordinate)

            hierarchy_queue.put(subordinate)

    A = nx.nx_agraph.to_agraph(G)
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight="2" -Nmargin=0 -Gfontsize=8')
    A.draw(file_path)



# =========================================================================
# ====================Plotting message graph===============================
# =========================================================================

def plot_message_graph(message_matrix, id_to_username_dict:dict,employee_dict:dict,file_path:str,employee_list:list,weight_threshold:int=1):
    """Plots the message graph given message matrix.

    Args:
        message_matrix : Message matrix generated from IM file.
        id_to_username_dict : id to user mapping dictionary.
        employee_dict : To get the roles of employees.
        file_path : file to which graph is drawn.
        employee_list :  Employees to be considered in the graph
        weight_threshold : Threshold for the weight for edge to be considered.

    Returns:
          None

    Plots graph which contains messages exchanged between employees in the employee list.
    """
    print(employee_list)
    G = nx.DiGraph()
    for src in range(len(message_matrix)):
        for dst in range(len(message_matrix)):
            if (message_matrix[src][dst] >= weight_threshold):
                src_user_name = id_to_username_dict[src]
                dst_user_name = id_to_username_dict[dst]
                if( (src_user_name in employee_list) and (dst_user_name in employee_list)):
                    src_user_title = employee_dict[src_user_name].title
                    dst_user_title = employee_dict[dst_user_name].title
                    src_label = src_user_title + ":" + src_user_name
                    dst_label =  dst_user_title + ":" + dst_user_name
                    G.add_edge(src_label,dst_label)

    A = nx.nx_agraph.to_agraph(G)
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight="2" -Nmargin=0 -Gfontsize=8')
    A.draw(file_path)


# ========================================================================================
# ==================Plotting sentiment within same hierarchy==============================
# ========================================================================================

def plot_sentiment_within_hierarchy(src_dir_path,top_user,start_week:int = 75, end_week:int = 120,only_week:bool=False):
    """Plots sentiments within same hierarchy level.

    Args:
        src_dir_path : Directory which contains the message files.
        start_week   : start week.
        end_week     : end week.
        only_week    : data is calculated weekly instead of each date.

    Returns:
        None.
    """

    level_subordinates_list = []
    level_sentiment_dict_list = []

    level_subordinates_list.append(employee.employee_dict["ROOT"].immediate_subordinates)
    prev_level_subordinate_list = level_subordinates_list[-1]

    while(len(prev_level_subordinate_list) > 0 ):
        curr_level_subordinate_list = []
        for curr_member in prev_level_subordinate_list:
            for subordinate in employee.employee_dict[curr_member].immediate_subordinates:
                curr_level_subordinate_list.append(subordinate)
        level_subordinates_list.append(curr_level_subordinate_list)
        prev_level_subordinate_list = curr_level_subordinate_list


    # num_levels_hierarchy = len(level_subordinates_list)-1 ## since the last level is empty.
    ## Depending on number of levels to be considered.
    num_levels_hierarchy = 4
    print(num_levels_hierarchy)

    for i in range(num_levels_hierarchy-1):
        sent_sentiment_dict, receive_sentiment_dict, within_sentiment_dict = sentiment. \
            compute_sentiments_from_filelist_multiproc(src_dir_path=src_dir_path,
                                                       user_name_list=level_subordinates_list[i],
                                                       num_process=4,in_network=False, complete_network=True,
                                                       start_week=start_week, end_week=end_week, only_week=only_week)

        # level_sentiment_dict_list.append(within_sentiment_dict)
        level_sentiment_dict_list.append(sent_sentiment_dict)

    dates_list = list(sorted(level_sentiment_dict_list[0].keys()))
    sorted_level_sentiment_list = [[] for x in range(num_levels_hierarchy-1)]


    for level_num in range(num_levels_hierarchy-1):
        for date in dates_list:
            sorted_level_sentiment_list[level_num].append(level_sentiment_dict_list[level_num][date])


    legend_info = ["level - {0}".format(x+1) for x in range(num_levels_hierarchy-1)]
    dates_list = [datetime.strptime(x, '%Y-%m-%d') for x in dates_list]

    ## just considering 3 levels as the rest of the levels dont exchange messages so often.

    plot_list_of_lists_vs_dates(dates_list,sorted_level_sentiment_list[:num_levels_hierarchy],
                                     xlabel= "Dates",ylabel = "sentiment of messages",title="Sentiment of messages at same hierarchy",
                                     legend_info=legend_info[:num_levels_hierarchy])



# ========================================================================================
# ==========================Plots Performance vs LIWC========================================
# ========================================================================================
def plot_relationship_between_performance_dict_category_dict(performance_date_dict:dict, category_date_dict: dict):
    """Plots relationship between performance and category dict.

    Args:
        performance_date_dict : Performance date dictionary.
        category_date_dict    : Category date dictionary.
    """

    category_date_dict = misc.change_key_string_key_date(category_date_dict)
    performance_date_dict_common, category_date_dict_common = misc.common_keys(performance_date_dict,category_date_dict)
    performance_list, category_dict_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,category_date_dict_common)
    liwc_value_list = [x["tentativeness"] for x in category_dict_list]

    liwc_value_list = [x for _, x in sorted(zip(performance_list, liwc_value_list), key=lambda pair: pair[0])]

    performance_list = sorted(performance_list)
    # general_plot(performance_list, liwc_value_list)

    print(liwc_value_list)
    print(performance_list)
    ## Flooring the performance values and getting the mean of liwc values corresponding to the floored performance value.

    performance_list = [math.floor(x) for x in performance_list]
    updated_performance_list = sorted(list(set(performance_list)))
    updated_liwc_value_list = []

    for x in updated_performance_list:
        accumulated_liwc = 0.0 ; count = 0.0
        for i in range(len(performance_list)):
            if (performance_list[i] == x):
                accumulated_liwc = accumulated_liwc + liwc_value_list[i]
                count = count + 1
        updated_liwc_value_list.append(accumulated_liwc/count)

    ## Remove zeros
    performance_list = updated_performance_list
    liwc_value_list  = updated_liwc_value_list
    updated_performance_list = []
    updated_liwc_value_list  = []

    for i in range(len(performance_list)):
        if(liwc_value_list[i] != 0):
            updated_liwc_value_list.append(liwc_value_list[i])
            updated_performance_list.append(performance_list[i])

    print(updated_liwc_value_list)
    print(updated_performance_list)

    updated_performance_list = [x for x in updated_performance_list]
    general_plot(updated_performance_list, updated_liwc_value_list,xlabel="Performance",ylabel="Percentage of words", title="Tentativeness")



def plot_relationship_performance_liwc(src_dir_path: str,inp_book_list:List, complete_network: bool = False,
                                          in_network:bool = True,start_week : int = 123,end_week : int=150,only_week:bool= False):
    """Plots relationship between performance and LIWC.

    Args:
        src_dir_path        : Directory path to IM files.
        inp_book_list       : List of Input Book.
        complete_network    : complete network to be considered (True: complete network, False: network depending on in_network).
        in_network          : messages considered within/outside network.
        start_week          : start week.
        end_week            : end week.
        only_week           : data is calculated weekly instead of each date.

    Returns:
        None

    """
    inp_employee_list = employee.employees_given_book_list(inp_book_list)

    total_sent_category_dict, total_recv_category_dict, total_within_category_dict = \
        liwc.compute_liwc_categories_from_filelist_multiproc(src_dir_path,inp_employee_list,1,
                                                             complete_network=complete_network,in_network=in_network,
                                                             start_week=start_week,end_week=end_week,only_week=only_week)

    dates_dict, performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE,inp_book_list,
                                                                           start_week=start_week,
                                                                           end_week=end_week, only_week=only_week)
    performance_date_dict = performance.combine_performance_given_book_list(dates_dict, performance_dict,
                                                                            only_week=only_week)

    plot_relationship_between_performance_dict_category_dict(performance_date_dict,total_recv_category_dict)


if __name__ == "__main__":

    employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
    employee_dict = employee.create_employee_hierarchy(employee_dict)
    # =========================================================================
    # ====================Plotting K-core Files================================
    # =========================================================================

    # dates,y_list = network.compute_element_kcore_for_plots(cfg.KCORE_PERSONAL_TOTAL,123,200,"kcore_largest_cc_num_nodes",6)
    # legend_info = [str(i)+"-core" for i in range(len(y_list))]
    # plot_list_of_lists_vs_dates(dates,y_list,"Time","Number of Nodes in Largest Connected Component",
    #                    "Number of Nodes in Largest Connected Component Vs Time : Personal",legend_info)
    # #
    # dates, y_list = network.compute_element_kcore_for_plots(cfg.KCORE_BUSINESS_TOTAL,123, 200, "kcore_largest_cc_num_nodes",6)
    # legend_info = [str(i)+"-core" for i in range(len(y_list))]
    # plot_list_of_lists_vs_dates(dates, y_list, "Time", "Number of Nodes in Largest Connected Component",
    #                                  "Number of Nodes in Largest Connected Component Vs Time : Business", legend_info)
    #
    # dates, y_list = network.compute_element_kcore_for_plots(cfg.KCORE_JOINT_TOTAL, 123, 200, "kcore_largest_cc_num_nodes",
    #                                                         6)
    # legend_info = [str(i) + "-core" for i in range(len(y_list))]
    # plot_list_of_lists_vs_dates(dates, y_list, "Time", "Number of Nodes in Largest Connected Component",
    #                             "Number of Nodes in Largest Connected Component Vs Time : Joint", legend_info)

    # =========================================================================
    # ====================Plotting hierarchy===================================
    # =========================================================================

    # plot_employee_hierarchy(employee_dict["ROOT"], employee_dict, cfg.PLOTS_DIR + "/root.jpg")
    ## If sapanski is the top employee.
    # plot_employee_hierarchy(employee_dict["sapanski_lawrence"],employee_dict,cfg.PLOTS_DIR+"/sapanski_lawrence.jpg")

    # =========================================================================
    # ====================Plotting performance of the book/booklist.===========
    # =========================================================================

    # book_name = "MENG"
    # dates_list, performance_list = performance.performance_given_book(cfg.PERFORMANCE_FILE, book_name, start_week=123,
    #                                                                   end_week=150, only_week=True)
    # plot_list_vs_dates(dates_list, performance_list, xlabel="Dates", ylabel="Performance", title="Performance of Book",
    #                    legend_info=book_name)

    # book_list = misc.read_book_file(cfg.BOOK_FILE)
    # dates_dict, performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE,book_list,0,300)

    # =========================================================================
    # ====================Plotting group performance===========================
    # =========================================================================

    # subordinates_list = employee.subordinates_given_employee(employee_dict,"cacouris_michael")
    # book_list = employee.books_given_employee_list(subordinates_list)
    # dates_dict,performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE,book_list,75,150,True)
    # performance_date_dict = performance.combine_performance_given_book_list(dates_dict,performance_dict)
    # plot_list_vs_dates(list(performance_date_dict.keys()),list(performance_date_dict.values()),"Dates","Performance","Performance vs Dates","performance list")

    # =========================================================================
    # ====================Plotting kcore performance.==========================
    # =========================================================================

    # start_week = 123;   end_week = 200;  k_value = 6;  max_lag = 20
    # dates_list = [misc.calculate_datetime(week_num=x) for x in range(start_week,end_week +1)]
    # total_performance_list = []
    # for k_value in range(1,7):
    #     performance_week_dict,_,_ = relationships.compute_relationships_performance_kcore(cfg.KCORE_PERSONAL,start_week=start_week,
    #                                                                                   end_week=end_week,k_value=k_value,max_lag=max_lag)
    #     performance_list = []
    #     for date_week in dates_list:
    #         performance_week = performance_week_dict.get(date_week,0)
    #         performance_list.append(performance_week)
    #
    #     total_performance_list.append(performance_list)
    #
    # legend_info = [str(i) + "-core" for i in range(1,7)]
    # plot_list_of_lists_vs_dates(dates_list,total_performance_list,"Time","Kcore-performance","Kcore-performance Vs Time",legend_info)

    # =========================================================================
    # ====================Plotting sentiment =================================
    # =========================================================================

    # subordinates_list = employee.subordinates_given_employee(employee_dict, "wolfberg_adam")
    # account_to_employee_dict, employee_to_account_dict = employee.map_employee_account(cfg.TRADER_BOOK_ACCOUNT_FILE)
    # subordinates_list = account_to_employee_dict["MENG"]
    #
    # sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict = \
    #     sentiment.compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_PERSONAL, subordinates_list,4,False,True,123,150,True)
    #
    # dates_list = sorted(sent_sentiment_dict.keys())
    # sent_sentiment_list = [] ; recv_sentiment_list = [] ; within_sentiment_list = []
    # print(dates_list)
    # for curr_date in dates_list:
    #     sent_sentiment_list.append(sent_sentiment_dict[curr_date])
    #     recv_sentiment_list.append(recv_sentiment_dict[curr_date])
    #     within_sentiment_list.append(within_sentiment_dict[curr_date])
    #
    # dates_list = [datetime.strptime(x, '%Y-%m-%d') for x in dates_list]
    #
    # y_list = [sent_sentiment_list, recv_sentiment_list, within_sentiment_list]
    # legend_info = ["sent-sentiment", "receive-sentiment", "within-sentiment"]
    #
    # plot_list_of_lists_vs_dates(dates_list, y_list, "Dates", "Sentiment", "Sentiment vs Dates", legend_info)
    #
    # plot_list_vs_dates(dates_list,y_list[0],"Dates","Sent-Sentiment", "Sent-Sentiment vs Dates","MENG")
    # plot_list_vs_dates(dates_list, y_list[1], "Dates", "Receive-Sentiment", "Receive-Sentiment vs Dates", "MENG")

    # ======================================================================================
    # ====================Plotting sentiment outside to inside users ========================
    # ==================================================================================
    # account_to_employee_dict, employee_to_account_dict = employee.map_employee_account(cfg.TRADER_BOOK_ACCOUNT_FILE)
    # subordinates_list = account_to_employee_dict["MENG"]
    # # subordinates_list = employee.subordinates_given_employee(employee_dict, "carter_richard")
    # sent_sentiment_dict_outside, recv_sentiment_dict_outside, within_sentiment_dict_outside = \
    #         sentiment.compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_PERSONAL, subordinates_list,
    #                                                              num_process=4,complete_network=False,in_network=False,
    #                                                              start_week=75,end_week=150,only_week=False)
    #
    # sent_sentiment_dict_inside, recv_sentiment_dict_inside, within_sentiment_dict_inside = \
    #     sentiment.compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_PERSONAL, subordinates_list,
    #                                                          num_process=4, complete_network=False, in_network=True,
    #                                                          start_week=75, end_week=150, only_week=False)
    #
    # #
    # dates_list = sorted(sent_sentiment_dict_outside.keys())
    # sent_sentiment_list_outside = [] ; recv_sentiment_list_outside = [] ; within_sentiment_list_outside = []
    # sent_sentiment_list_inside = [] ; recv_sentiment_list_inside = []; within_sentiment_list_inside = []
    # print(dates_list)
    # for curr_date in dates_list:
    #     sent_sentiment_list_outside.append(sent_sentiment_dict_outside[curr_date])
    #     recv_sentiment_list_outside.append(recv_sentiment_dict_outside[curr_date])
    #     within_sentiment_list_outside.append(within_sentiment_dict_outside[curr_date])
    #
    #     sent_sentiment_list_inside.append(sent_sentiment_dict_inside[curr_date])
    #     recv_sentiment_list_inside.append(recv_sentiment_dict_inside[curr_date])
    #     within_sentiment_list_inside.append(within_sentiment_dict_inside[curr_date])
    # #
    # dates_list = [datetime.strptime(x, '%Y-%m-%d') for x in dates_list]
    # #
    # # y_list = [sent_sentiment_list_outside, sent_sentiment_list_inside]
    # # legend_info = ["sent-sentiment-outside", "sent-sentiment-inside"]
    #
    # y_list = [recv_sentiment_list_outside, recv_sentiment_list_inside]
    # legend_info = ["receive-sentiment-outside", "receive-sentiment-inside"]
    # #
    # plot_list_of_lists_vs_dates(dates_list, y_list, "Dates", "Sentiment", "Sentiment vs Dates", legend_info)
    # relationships.compute_correlation(recv_sentiment_list_inside,recv_sentiment_list_outside)

    # =========================================================================
    # ====================Plotting Message graph ==============================
    # =========================================================================

    # subordinates_list = employee.subordinates_given_employee(employee_dict, "sapanski_lawrence")
    # message_matrix, message_adj_list, id_to_username_dict = network.create_matrix(cfg.SENTIMENT_PERSONAL+"im_df_week150.csv")
    # plot_message_graph(message_matrix,id_to_username_dict,employee_dict,cfg.PLOTS_DIR+"/message_graph_week150.png",subordinates_list,1)

    # ===========================================================================================================
    # ==============================Computing relationship between sentiment same  hierarchy==============
    # ===========================================================================================================

    # plot_sentiment_within_hierarchy(cfg.SENTIMENT_PERSONAL, "ROOT", start_week=125, end_week=160,
    #                                                 only_week=True)

    # ===========================================================================================================
    # ==============================Plotting Performance and LIWC ==============
    # ===========================================================================================================

    # plot_relationship_performance_liwc(cfg.SENTIMENT_PERSONAL,["ADAM"],complete_network=False,in_network=True, start_week=123,
    #                                    end_week=200, only_week=False)

    pass