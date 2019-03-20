"""This file contains functions which tests relationships between data. Ex. Correlation/causation."""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from pandas.core import datetools
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from typing import Tuple

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
    """Computes if there is a causal relationship between x and y.

    Args:
        cause : List .
        effect : List .
        max_lag : maximum lag to be considered between two list.

    Returns:
        causality_dict : key - lag , value = Tuple(F-value,p-value, df_denom, df_num)
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

def compute_causal_relationships_performance_sentiment(performance_date_dict,sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict,max_lag = 3) -> dict:
    """Computes granger causality between dictionary values of performance and sentiment.

    Args:
        perfomance_date_dict : Performance dictionary.
        sent_sentiment_dict : Sent sentiment dictionary.
        recv_sentiment_dict : Receive sentiment dictionary.
        within_sentiment_dict : within sentiment dictionary.
        max_lag : Maximum lag that can be applied.

    Returns:
        book_causal_dict = key : sent/receive/within. value : corresponding causal dictionaries.

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

    sent_dict = compute_causality(sentiment_list, performance_list, max_lag)
    book_causal_dict["sent"] = sent_dict

    print("===============================RECEIVED==========================================")
    print(len(performance_date_dict), len(recv_sentiment_dict))
    performance_date_dict_common, recv_sentiment_dict_common = misc.common_keys(performance_date_dict,
                                                                                recv_sentiment_dict)
    print(len(performance_date_dict_common), len(recv_sentiment_dict_common))
    performance_list, sentiment_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,
                                                                             recv_sentiment_dict_common)
    recv_dict = compute_causality(sentiment_list, performance_list, max_lag)
    book_causal_dict["recv"] = recv_dict

    print("===============================WITHIN==========================================")
    print(len(performance_date_dict), len(within_sentiment_dict))
    performance_date_dict_common, within_sentiment_dict_common = misc.common_keys(performance_date_dict,
                                                                                  within_sentiment_dict)
    print(len(performance_date_dict_common), len(within_sentiment_dict_common))
    performance_list, sentiment_list = misc.get_list_from_dicts_sorted_dates(performance_date_dict_common,
                                                                             within_sentiment_dict_common)
    within_dict  = compute_causality(sentiment_list, performance_list, max_lag)
    book_causal_dict["within_dict"] = within_dict

    return book_causal_dict


def compute_relationships_book_list_performance_sentiment(book_list: List, start_week, end_week, maximum_lag, only_week = False) -> dict:
    """Computes relationships for all books present with sentiments.

    Args:
        book_list : List of books for which relationships are calculated.
        start_week : start week to be considered.
        end_week : end week.
        maximum_lag : maximum lag to check for causality.
        only_week :  For weekly data to be considered instead of daily.

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

        sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict = sentiment. \
            compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_BUSINESS, employee_account_list, 4, True,
                                                       start_week, end_week, only_week=only_week)

        book_causal_dict = compute_causal_relationships_performance_sentiment(performance_date_dict, sent_sentiment_dict, recv_sentiment_dict,
                                     within_sentiment_dict, maximum_lag)
        ## Compute Causality between these dictionaries.
        book_list_causal_dict[book] = book_causal_dict


    return book_list_causal_dict


if __name__ == "__main__":



    # book_list = ["SALZ"]
    book_list = misc.read_book_file(cfg.BOOK_FILE)
    book_list_causal_dict = compute_relationships_book_list_performance_sentiment(book_list,start_week=123, end_week=265,
                                                            maximum_lag= 5, only_week= False)

    # print(book_list_causal_dict)
    misc.write_dict_in_file(book_list_causal_dict,cfg.PKL_FILES+"/book_list_causality_daily.pkl")
    output_dict = misc.read_file_into_dict(cfg.PKL_FILES+"/book_list_causality_daily.pkl")
    # print(output_dict)

