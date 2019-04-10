"""This file contains functions that are used to analyze output of computation functions."""
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import misc
import config as cfg



def analyse_causality(pkl_file_name :str,inp_msg_type : str, ftest_type : str ="ssr_chi2test", extra_label:str = ""):
    """ Analyses causality given a pkl by box plots for each lag.

    Analyses if causality is present in all the books by plotting box-plot for p-values of all lags.
    Histogram for minimum p-value of a book.

    Args:
        pkl_file_name : pickle file which contains the dictionary.
        inp_msg_type : message type for which p-value need to be considered. (sent/receive/within)
        ftest_type : ssr_ftest/ssr_chi2test/lrtest/params_ftest
        extra_label : extra label to be added for title.

    Returns:
        None.

    Dictionary structure.
    inp_dict : key - book, value - book_causal_dict.
    book_causal_dict : key : sent/recv/within_dict. value : corresponding causal dictionaries(causality_dict).
    causality_dict : key - lag, value - test_result_dict
    test_result_dict : key - ssr_ftest/ssr_chi2test/lrtest/params_ftest, value - Tuple(F-value,p-value, df_denom, df_num)

    """

    inp_dict = misc.read_file_into_dict(pkl_file_name)
    lag_pvalues_dict = {} ## key : lag, value : list, which contains p-values for that lag in all books.
    min_pvalue_each_book = []

    for book, book_causal_dict in inp_dict.items():
        causality_dict = book_causal_dict[inp_msg_type]
        for lag, test_result_dict in causality_dict.items():
            if (len(test_result_dict) > 1):  ## we store an empty dict if the granger causality results in an error.
                causal_tuple = test_result_dict[ftest_type]
                # print(causal_tuple)
                lag_list = lag_pvalues_dict.get(lag,[])
                lag_list.append(causal_tuple[1])
                lag_pvalues_dict[lag] = lag_list

    plt.boxplot([x for x in lag_pvalues_dict.values()], 0, 'rs', 1)
    plt.xlabel("lag values")
    plt.ylabel("p-values")
    plt.title("Box plot of p-values vs lag values, test : {0}-{1}".format(ftest_type, extra_label))
    plt.show()

    ## for min p-value in all the lags.

    for book, book_causal_dict in inp_dict.items():
        causality_dict = book_causal_dict[inp_msg_type]
        min_p_value = 10
        for lag, test_result_dict in causality_dict.items():
            if (len(test_result_dict) > 1):  ## we store an empty dict if the granger causality results in an error.
                curr_p_value = test_result_dict[ftest_type][1]
                if(curr_p_value < min_p_value):
                    min_p_value = curr_p_value

        if(min_p_value != 10):
            min_pvalue_each_book.append(min_p_value)
    bin_list = list(np.arange(0,1,0.1))
    print(len(min_pvalue_each_book))
    # sb.distplot(min_pvalue_each_book,norm_hist=True,bins=bin_list)
    plt.hist(min_pvalue_each_book, bins=bin_list)
    plt.xlabel("Minimum p-value for book")
    plt.ylabel("Number of books")
    plt.title("Minimum p-value distribution for books - {0}".format(extra_label))
    plt.show()

    print(lag_pvalues_dict)



if __name__ == "__main__":

    # ========================================================================================
    # =================== Analysis of Performance affect on sentiment ========================
    # ========================================================================================

    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_performance_effect_sentiment_daily.pkl",
    #                   "sent", "ssr_chi2test", "Performance affect on Sent-Sentiment ")
    #
    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_performance_effect_sentiment_daily.pkl",
    #                   "recv", "ssr_chi2test", "Performance affect on Receive-Sentiment")
    # #
    # analyse_causality(cfg.PKL_FILES+"/books_causal_effect_cause_performance_effect_sentiment_daily.pkl",
    #                   "within_dict","ssr_chi2test", "Performance affect on Within-Sentiment")

    # ========================================================================================
    # =================== Analysis of Sentiment affect on Performance ========================
    # ========================================================================================

    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_sentiment_effect_performance_daily.pkl",
    #                   "sent", "ssr_ftest", "Sent-Sentiment affect on Performance")

    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_sentiment_effect_performance_daily.pkl",
    #                   "recv", "ssr_ftest", "Receive-Sentiment affect on Performance")
    #
    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_sentiment_effect_performance_daily.pkl",
    #                   "within_dict", "ssr_ftest", "Within-Sentiment affect on Performance ")

    # ========================================================================================
    # =================== Analysis of Performance affect on sentiment(out-network) ===========
    # ========================================================================================

    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_performance_effect_sentiment_daily_out_network.pkl",
    #                   "sent", "ssr_chi2test", "Performance affect on Sent-Sentiment - Out-Network")
    #
    # analyse_causality(cfg.PKL_FILES + "/books_causal_effect_cause_performance_effect_sentiment_daily_out_network.pkl",
    #                   "recv", "ssr_chi2test", "Performance affect on Receive-Sentiment - Out-Network")
    #
    # analyse_causality(cfg.PKL_FILES+"/books_causal_effect_cause_performance_effect_sentiment_daily_out_network.pkl",
    #                   "within_dict","ssr_chi2test", "Performance affect on Within-Sentiment - Out-Network")
    pass