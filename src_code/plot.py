"""This file contains functions which are used for plotting."""

from typing import List
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter,DayLocator,MONDAY
from datetime import  datetime,timedelta,date
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import queue

import network
import employee
import config as cfg
import performance
import sentiment

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
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
    A.draw(file_path)
    # A.show()




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

    # plot.plot_employee_hierarchy(employee_dict["ROOT"], employee_dict, cfg.PLOTS_DIR + "/root.jpg")
    ## If sapanski is the top employee.
    # plot_employee_hierarchy(employee_dict["sapanski_lawrence"],employee_dict,cfg.PLOTS_DIR+"/sapanski_lawrence.jpg")

    # =========================================================================
    # ====================Plotting performance=================================
    # =========================================================================

    # dates_list,performance_list = performance.performance_given_book(cfg.PERFORMANCE_FILE,"ADAM",start_week=180,end_week=200)
    # plot.plot_list_vs_dates(dates_list,performance_list,xlabel="Dates",ylabel="Performance",title="Performance of Book",legend_info="ADAM")

    # book_list = misc.read_book_file(cfg.BOOK_FILE)
    # dates_dict, performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE,book_list,0,300)

    # =========================================================================
    # ====================Plotting group performance===========================
    # =========================================================================

    subordinates_list = employee.subordinates_given_employee(employee_dict,"cacouris_michael")
    book_list = employee.books_given_employee_list(subordinates_list)
    dates_dict,performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE,book_list,75,150,True)
    performance_date_dict = performance.combine_performance_given_book_list(dates_dict,performance_dict)
    plot_list_vs_dates(list(performance_date_dict.keys()),list(performance_date_dict.values()),"Dates","Performance","Performance vs Dates","performance list")


    # =========================================================================
    # ====================Plotting sentiment =================================
    # =========================================================================

    subordinates_list = employee.subordinates_given_employee(employee_dict, "cacouris_michael")

    sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict = \
        sentiment.compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_PERSONAL, subordinates_list,2,True,75,150,True)

    dates_list = sorted(sent_sentiment_dict.keys())
    sent_sentiment_list = [] ; recv_sentiment_list = [] ; within_sentiment_list = []
    print(dates_list)
    for date in sorted(dates_list):
        sent_sentiment_list.append(sent_sentiment_dict[date])
        recv_sentiment_list.append(recv_sentiment_dict[date])
        within_sentiment_list.append(within_sentiment_dict[date])

    dates_list = [datetime.strptime(x, '%Y-%m-%d') for x in dates_list]

    y_list = [sent_sentiment_list, recv_sentiment_list, within_sentiment_list]
    legend_info = ["sent-sentiment", "receive-sentiment", "within-sentiment"]

    plot_list_of_lists_vs_dates(dates_list, y_list, "Dates", "Sentiment", "Sentiment vs Dates", legend_info)


    pass
