"""This file contains functions which are used for plotting."""

from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter,DayLocator,MONDAY
from datetime import  datetime,timedelta,date
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import queue
import random

import network
import employee
import config as cfg
import performance
import sentiment
import misc

def plot_list_of_lists_vs_dates(x :List,y_list : List[List],xlabel :str,ylabel: str,title:str,legend_info:List):
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

    months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")
    # every monday
    mondays = WeekdayLocator(MONDAY)

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


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

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()

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








if __name__ == "__main__":

    employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
    employee_dict = employee.create_employee_hierarchy(employee_dict)
    # =========================================================================
    # ====================Plotting K-core Files================================
    # =========================================================================

    # dates,y_list = network.compute_element_kcore_for_plots(cfg.KCORE_PERSONAL,75,150,"kcore_largest_cc_num_nodes",10)
    # legend_info = [str(i)+"-core" for i in range(len(y_list))]
    # plot_list_of_lists_vs_dates(dates,y_list,"Time","Number of Nodes in Largest Connected Component",
    #                    "Number of Nodes in Largest Connected Component Vs Time : Personal",legend_info)
    #
    # dates, y_list = network.compute_element_kcore_for_plots(cfg.KCORE_BUSINESS,75, 150, "kcore_largest_cc_num_nodes",10)
    #
    # legend_info = [str(i)+"-core" for i in range(len(y_list))]
    # plot_list_of_lists_vs_dates(dates, y_list, "Time", "Number of Nodes in Largest Connected Component",
    #                                  "Number of Nodes in Largest Connected Component Vs Time : Business", legend_info)

    # =========================================================================
    # ====================Plotting hierarchy===================================
    # =========================================================================

    # plot_employee_hierarchy(employee_dict["ROOT"], employee_dict, cfg.PLOTS_DIR + "/root.jpg")
    ## If sapanski is the top employee.
    # plot_employee_hierarchy(employee_dict["sapanski_lawrence"],employee_dict,cfg.PLOTS_DIR+"/sapanski_lawrence.jpg")

    # =========================================================================
    # ====================Plotting performance=================================
    # =========================================================================

    dates_list,performance_list = performance.performance_given_book(cfg.PERFORMANCE_FILE,"MENG",start_week=123,end_week=350,only_week=True)
    plot_list_vs_dates(dates_list,performance_list,xlabel="Dates",ylabel="Performance",title="Performance of Book",legend_info="MENG")

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
    # ====================Plotting sentiment =================================
    # =========================================================================

    # subordinates_list = employee.subordinates_given_employee(employee_dict, "wolfberg_adam")
    # account_to_employee_dict, employee_to_account_dict = employee.map_employee_account(cfg.TRADER_BOOK_ACCOUNT_FILE)
    # subordinates_list = account_to_employee_dict["MENG"]
    #
    # sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict = \
    #     sentiment.compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_PERSONAL, subordinates_list,4,True,75,150,True)
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

    # =========================================================================
    # ====================Plotting Message graph ==============================
    # =========================================================================

    # subordinates_list = employee.subordinates_given_employee(employee_dict, "sapanski_lawrence")
    # message_matrix, message_adj_list, id_to_username_dict = network.create_matrix(cfg.SENTIMENT_PERSONAL+"im_df_week150.csv")
    # plot_message_graph(message_matrix,id_to_username_dict,employee_dict,cfg.PLOTS_DIR+"/message_graph_week150.png",subordinates_list,1)