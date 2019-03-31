"""This file contains functions which tests relationships between data. Ex. Correlation/causation."""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from pandas.core import datetools
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from typing import Tuple
import os

import performance
import employee
import sentiment
import misc
import config as cfg


def check_series_stationary(x:List):
    """Checks if input time series is stationary.

    Args:
        x : Input list whose time series need to be calculated.

    Returns:
        None.

    Prints the result of ADF test.
    Null Hypothesis : The series has a unit root (value of a =1) (Series is non-stationary).
    Alternative Hypothesis : The series has no unit root. (Series is stationary).
    if test_statistic < critical value : reject null hypothesis, series is stationary.
    else : series is non stationary.
    """

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(x, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def compute_causality(cause : List,effect : List,max_lag : int)-> dict:
    """Computes if there is a causal relationship between cause and effect.

    Args:
        cause : List .
        effect : List .
        max_lag : maximum lag to be considered between two list.

    Returns:
        causality_dict : key - lag, value - test_result_dict

    test_result_dict : key - ssr_ftest/ssr_chi2test/lrtest/params_ftest, value - Tuple(F-value,p-value, df_denom, df_num)
    """

    causality_dict = {}
    try:
        inp_2d = np.column_stack((effect,cause))
        granger_dict = grangercausalitytests(inp_2d,max_lag,verbose=False)
        for lag in granger_dict.keys():
            causality_dict[lag] = granger_dict[lag][0]
    except:
        for lag in range(max_lag):
            causality_dict[lag] = {}

    return causality_dict
    # print(granger_dict[1][0]["ssr_ftest"][1])

# ========================================================================================
# ===========Performance vs sentiment causality functions=================================
# ========================================================================================



def compute_causal_relationships_performance_sentiment(performance_date_dict,sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict,max_lag = 3) -> dict:
    """Computes granger causality between dictionary values of performance and sentiment.

    Args:
        perfomance_date_dict : Performance dictionary.
        sent_sentiment_dict : Sent sentiment dictionary.
        recv_sentiment_dict : Receive sentiment dictionary.
        within_sentiment_dict : within sentiment dictionary.
        max_lag : Maximum lag that can be applied.

    Returns:
        book_causal_dict = key : sent/recv/within_dict. value : corresponding causal dictionaries.

    """
    sent_sentiment_dict = misc.change_key_string_key_date(sent_sentiment_dict)
    recv_sentiment_dict = misc.change_key_string_key_date(recv_sentiment_dict)
    within_sentiment_dict = misc.change_key_string_key_date(within_sentiment_dict)
    book_causal_dict = {}

    print("===============================SENT==========================================")
    print(len(performance_date_dict), len(sent_sentiment_dict))

    performance_date_dict_common, sent_sentiment_dict_common = misc.common_keys(performance_date_dict,
                                                                                sent_sentiment_dict)
    print(len(performance_date_dict_common), len(sent_sentiment_dict_common))
    performance_list, sentiment_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,
                                                                             sent_sentiment_dict_common)

    sent_dict = compute_causality(performance_list, sentiment_list, max_lag)
    # sent_dict = compute_causality(sentiment_list, performance_list, max_lag)
    book_causal_dict["sent"] = sent_dict

    print("===============================RECEIVED==========================================")
    print(len(performance_date_dict), len(recv_sentiment_dict))
    performance_date_dict_common, recv_sentiment_dict_common = misc.common_keys(performance_date_dict,
                                                                                recv_sentiment_dict)
    print(len(performance_date_dict_common), len(recv_sentiment_dict_common))
    performance_list, sentiment_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,
                                                                             recv_sentiment_dict_common)
    recv_dict = compute_causality(performance_list,sentiment_list, max_lag)
    # recv_dict = compute_causality(sentiment_list, performance_list, max_lag)
    book_causal_dict["recv"] = recv_dict

    print("===============================WITHIN==========================================")
    print(len(performance_date_dict), len(within_sentiment_dict))
    performance_date_dict_common, within_sentiment_dict_common = misc.common_keys(performance_date_dict,
                                                                                  within_sentiment_dict)
    print(len(performance_date_dict_common), len(within_sentiment_dict_common))
    performance_list, sentiment_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,
                                                                             within_sentiment_dict_common)
    within_dict  = compute_causality(performance_list, sentiment_list, max_lag)
    # within_dict = compute_causality(sentiment_list, performance_list, max_lag)
    book_causal_dict["within_dict"] = within_dict

    return book_causal_dict


def compute_relationships_book_list_performance_sentiment(book_list: List, start_week, end_week, maximum_lag, only_week = False,
                                                          in_network:bool = True, complete_network:bool = False) -> dict:
    """Computes relationships between performance of books in book list to sentiment of users corresponding to the book.

    Args:
        book_list : List of books for which relationships are calculated.
        start_week : start week to be considered.
        end_week : end week.
        maximum_lag : maximum lag to check for causality.
        only_week :  For weekly data to be considered instead of daily.
        in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: Outside)
        complete_network : complete network to be considered.

    Returns:
        book_list_causal_dict : key - book, value - book_causal_dict
    """
    account_to_employee_dict, employee_to_account_dict = employee.map_employee_account(cfg.TRADER_BOOK_ACCOUNT_FILE)

    book_list_causal_dict = {}



    for book in book_list:
        ## Get the employee correponding to an account
        employee_account_list = account_to_employee_dict[book]

        ## Compute the performance of account for given weeks.
        dates_dict, performance_dict = performance.performance_given_book_list(cfg.PERFORMANCE_FILE, [book],
                                                                               start_week=start_week,
                                                                               end_week=end_week, only_week=only_week)
        performance_date_dict = performance.combine_performance_given_book_list(dates_dict, performance_dict,
                                                                                only_week=only_week)

        ## Compute the sentiment corresponding to employee list for the given account.

        print(performance_date_dict)
        sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict = sentiment. \
            compute_sentiments_from_filelist_multiproc(src_dir_path= cfg.SENTIMENT_BUSINESS, user_name_list= employee_account_list,
                                                       num_process= 4,in_network= in_network,complete_network=complete_network,
                                                       start_week = start_week,end_week= end_week, only_week=only_week)

        book_causal_dict = compute_causal_relationships_performance_sentiment(performance_date_dict, sent_sentiment_dict, recv_sentiment_dict,
                                     within_sentiment_dict, maximum_lag)
        ## Compute Causality between these dictionaries.
        book_list_causal_dict[book] = book_causal_dict


    return book_list_causal_dict

# ========================================================================================
# ===============Performance vs kcore causality functions=================================
# ========================================================================================
def compute_causal_relationships_performance_kcore(kcore_performance_dict, kcore_num_nodes_dict,max_lag:int = 5) -> dict:
    """Computes granger causality between dictionary values of performance and kcore.

    Args:
        kcore_performance_dict : Performance dictionary.
        kcore_num_nodes_dict : number of nodes in kcore.
        max_lag : Maximum lag that can be applied.

    Returns:
        causal_performance_kcore_dict : key - lag, value : Tuple(F-value,p-value, df_denom, df_num)
    """
    print(len(kcore_performance_dict), len(kcore_num_nodes_dict))
    kcore_performance_dict_common, kcore_num_nodes_dict_common = misc.common_keys(kcore_performance_dict,
                                                                                 kcore_num_nodes_dict)
    print(len(kcore_performance_dict_common), len(kcore_num_nodes_dict_common))

    performance_list, kcore_num_list = misc.get_list_from_dicts_sorted_dates(kcore_performance_dict_common,
                                                                             kcore_num_nodes_dict_common)

    causal_performance_kcore_dict = compute_causality(kcore_num_list, performance_list, max_lag)

    return causal_performance_kcore_dict


def compute_relationships_performance_kcore(src_dir_path: str, start_week : int,end_week : int, k_value : int,max_lag : int = 5) -> Tuple[dict,dict,dict]:
    """Computes the relationships between performance and k-core.

    Args:
         src_dir_path : path to source directory.
         start_week : starting week.
         end_week   : end week.
         k_value  : k-value for which causality is measured.
         max_lag : Maximum lag that can be applied.

    Returns:
        kcore_performance_dict : contains kcore performance for given weeks.
                                    key : date(monday of the week number) ; value : kcore performance of that week.
        kcore_num_nodes_dict   : Number of nodes in the largest cc of in that week.
                                    key : date(monday of the week number) ; value : number of nodes in largest cc of that week.
        causal_performance_kcore_dict : causal data for that week.
    """
    ## key : date(monday of the week number) value : Number of kcore largest cc nodes in the week.
    kcore_num_nodes_dict = {}
    kcore_performance_dict ={}
    employee_id_username_dict, _ = employee.employee_id_to_username_from_file(cfg.EMPLOYEE_MASTER_FILE)
    performance_week_dict = misc.read_file_into_dict(cfg.PKL_FILES+"/performance_weekly.pkl")

    for week_num in range(start_week,end_week + 1):
        ## need employees present in each core and books related to them.
        kcore_largest_cc_nodes_file = os.path.join(src_dir_path,"kcore_largest_cc_nodes_week{0}.csv".format(week_num))
        kcore_largest_cc_num_nodes_file = os.path.join(src_dir_path, "kcore_largest_cc_num_nodes_week{0}.csv".format(week_num))
        kcore_number_file = os.path.join(src_dir_path,"kcore_number_week{0}.csv".format(week_num))

        kcore_largest_cc_nodes_list = misc.read_file_into_list(kcore_largest_cc_nodes_file)

        kcore_largest_cc_num_nodes_list = misc.read_file_into_list(kcore_largest_cc_num_nodes_file)
        kcore_largest_cc_num_nodes_list = [int(x) for x in kcore_largest_cc_num_nodes_list]

        kcore_number_list = misc.read_file_into_list(kcore_number_file)
        kcore_number_list = [int(x) for x in kcore_number_list]
        # print(kcore_number_list)

        if k_value in kcore_number_list:
            idx = kcore_number_list.index(k_value)

            ## Number of nodes.
            kcore_largest_cc_num_node = kcore_largest_cc_num_nodes_list[idx]
            kcore_num_nodes_dict[misc.calculate_datetime(week_num=week_num)] = kcore_largest_cc_num_node
            # ========================================================================================
            ## Performance of the employees present in kcore.
            # ========================================================================================
            ## get the epmployee ids from kcore and get their corresponding user names.
            employee_username_list_kcore = []
            employee_id_list_kcore = kcore_largest_cc_nodes_list[idx].split('-')
            for employee_id in employee_id_list_kcore:
                employee_id = int(employee_id)
                if employee_id in employee_id_username_dict.keys():
                    employee_username_list_kcore.append(employee_id_username_dict[employee_id])


            # print(employee_username_list_kcore)

            ## get the book list corresponding to employees present in k-core and use precomputed weekly performance.
            book_list_kcore = employee.books_given_employee_list(employee_username_list_kcore)

            curr_week_performance_list = []
            for book in book_list_kcore:
                performance_week_dict_book = performance_week_dict.get(book,{})
                curr_week_book_performance = performance_week_dict_book.get(week_num,0)
                curr_week_performance_list.append(curr_week_book_performance)

            if len(curr_week_performance_list) > 0:
                curr_week_performance = sum(curr_week_performance_list)/len(curr_week_performance_list)
            else:
                curr_week_performance = 0

            kcore_performance_dict[misc.calculate_datetime(week_num=week_num)] = curr_week_performance

    # print(kcore_performance_dict)
    # print(kcore_num_nodes_dict)
    causal_performance_kcore_dict = compute_causal_relationships_performance_kcore(kcore_performance_dict,kcore_num_nodes_dict,max_lag)
    for lag,value in causal_performance_kcore_dict.items():
        print(lag, value)

    return kcore_performance_dict,kcore_num_nodes_dict,causal_performance_kcore_dict


if __name__ == "__main__":

    # # ========================================================================================
    # # ===========Computing causality given book list(performance & kcore)=====================
    # # =========================================================================================

    # start_week = 123; end_week = 200 ; k_value = 6 ; max_lag = 20
    # print("===============================BUSINESS==========================================")
    #
    # compute_relationships_performance_kcore(cfg.KCORE_BUSINESS,start_week=start_week,end_week=end_week,k_value=k_value,max_lag=max_lag)
    # print("===============================PERSONAL==========================================")
    #
    # compute_relationships_performance_kcore(cfg.KCORE_PERSONAL, start_week=start_week, end_week=end_week, k_value=k_value, max_lag=max_lag)
    # print("===============================JOINT==========================================")
    # compute_relationships_performance_kcore(cfg.KCORE_JOINT,start_week=start_week, end_week= end_week, k_value= k_value, max_lag= max_lag)

    # ===========================================================================================================
    # ===========Computing causality given book list(performance & sentiment) in_network = True=================
    # ===========================================================================================================

    # book_list = ["MENG"]
    # book_list = misc.read_book_file(cfg.BOOK_FILE)
    # book_list_causal_dict = compute_relationships_book_list_performance_sentiment(book_list, start_week=123,
    #                                                                               end_week=263,
    #                                                                               maximum_lag=10, only_week=False,
    #                                                                               complete_network=False,
    #                                                                               in_network=True)
    #
    # misc.write_dict_in_file(book_list_causal_dict,
    #                         cfg.PKL_FILES + "/books_causal_effect_cause_sentiment_effect_performance_daily.pkl")
    # output_dict = misc.read_file_into_dict(
    #     cfg.PKL_FILES + "/books_causal_effect_cause_sentiment_effect_performance_daily.pkl")
    # print(output_dict)

    # for book,dict1 in book_list_causal_dict.items():
    #     for msg_type,dict2 in dict1.items():
    #         print("==============================={0}==========================================".format(msg_type))
    #         for lag, dict3 in dict2.items():
    #             print(lag,dict3)

    # ===========================================================================================================
    # ===========Computing causality given book list(performance & sentiment) in_network = False=================
    # ===========================================================================================================

    # book_list = ["MENG"]
    book_list = misc.read_book_file(cfg.BOOK_FILE)
    book_list_causal_dict = compute_relationships_book_list_performance_sentiment(book_list, start_week=123,
                                                                                  end_week=263,
                                                                                  maximum_lag=10, only_week=False,
                                                                                  complete_network=False,
                                                                                  in_network=False)

    misc.write_dict_in_file(book_list_causal_dict,
                            cfg.PKL_FILES + "/books_causal_effect_cause_performance_effect_sentiment_daily_out_network.pkl")
    output_dict = misc.read_file_into_dict(
        cfg.PKL_FILES + "/books_causal_effect_cause_performance_effect_sentiment_daily_out_network.pkl")
    print(output_dict)

    # for book, dict1 in book_list_causal_dict.items():
    #     for msg_type,dict2 in dict1.items():
    #             print("==============================={0}==========================================".format(msg_type))
    #             for lag, dict3 in dict2.items():
    #                 print(lag,dict3)


    pass