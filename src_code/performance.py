"""This file contains functions related to performance."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import pandas as pd
from datetime import datetime
from typing import List
from typing import Tuple
import numpy as np


import config as cfg
import misc

def performance_given_book(file_path : str, book:str,start_week : int = 0, end_week : int = 300,only_week:bool=False) -> Tuple[List,List]:
    """Calculates the performance of given book over given weeks.

    Args:
        file_path : path to the performance file.
        book : name of the book.
        start_week : starting week number.
        end_week   : ending week number.
        only_week  : data is calculated weekly instead of each date.

    Returns:
        dates: dates on which performance is present.
        performance : performance on given dates.

    Performance is calculated by dividing PnL with mean of PnL for the given period.
    We need to have a fraction because it is easy to add up performances for different books.
    """
    df = pd.read_csv(file_path)
    df = df[df["book"]== book]

    df = df.dropna()
    ## To remove the duplicates present in a single day.
    df = df.sort_values("date")
    df = df.drop_duplicates("date",keep='last')

    df["date"] = df["date"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    df["week"] = df["date"].apply(lambda x: misc.calculate_week(x.date()))
    df = df.sort_values("date")

    ## consider between give weeks.
    df = df[(df["week"] >= start_week) & (df["week"] <= end_week)]

    ## If only one value is considered for each week.
    if only_week:
        df = df.drop_duplicates("week", keep='last')

    # df = df[["date", "delta", "PnL_MTD_adjusted", "AccountingFile_PnL_MTD", "year_month"]]
    dates = df["date"].tolist()
    performance = df["PnL_MTD_adjusted"].tolist()
    ## calculate the mean of the book and divide each element by its mean.

    if (len(performance) > 0):
        performance = [float(x) for x in performance]
        performance_mean = sum(performance)/len(performance)
        performance = [x/performance_mean for x in performance]

    return dates, performance

def performance_given_book_list(file_path:str, book_list:List,start_week : int = 0, end_week : int = 300,only_week:bool=False) -> Tuple[dict,dict]:
    """Calculates the performance given book list.

    Args:
        file_path : path to the performance file.
        book_list : list of books.
        start_week : starting week number.
        end_week   : ending week number.
        only_week  : data is calculated weekly instead of each date.

    Returns:
        dates: dates on which performance is present.
        performance : performance on given dates.

    """
    dates_dict = {}
    performance_dict = {}

    for book in book_list:
        dates,performance = performance_given_book(file_path,book,start_week,end_week,only_week=only_week)
        # print("Book: {0}, Length of Dates: {1}".format(book,len(dates)))
        dates_dict[book] = dates
        performance_dict[book] = performance

    return dates_dict,performance_dict

def combine_performance_given_book_list(dates_dict : dict, performance_dict :dict, only_week:bool=False) -> dict:
    """ Combines performance of multiple books into a single value for a week.

    Args:
        dates_dict : For each book, the dates in which performance value is present. {key - book_name, value - dates_list}.
        performance_dict : For each book, list of performance values. {key - book_name, value - performance_list}

    Returns:
        total_date_performance_dict : A dictionary
                                      key - dates for which combined performance is calculated, value : combined performance.

    The main issue is due to performance values are not present for each book for all the days.
    cumulative performance is calculated by taking average of performance of books present in that day.

    total_dates_list : dates for which combined performance is calculated.
    total_performance_list : combined performance list values.
    """
    total_dates_list = []
    total_performance_list = []
    ## key - date, value - list of all performance values in that day for all the books.
    intervalised_performance = {}

    for book,book_dates_list in dates_dict.items():
        book_performance_list = performance_dict[book]
        for book_date,book_performance in zip(book_dates_list,book_performance_list):
            if only_week:
                ## This condition is required because different books may have different dates in the week if weekly data is considered.
                ## i.e one book may have wednesday and other might have Thrursday. Need to normalize that to Thursday for every week.
                book_date_week = misc.calculate_week(book_date.to_datetime().date())
                book_date = misc.calculate_date(week_num=book_date_week)

            intervalised_performance[book_date] =intervalised_performance.get(book_date,[])
            intervalised_performance[book_date].append(book_performance)

    ## Take the mean of the performance values w.r.t each date.
    for books_date, books_performance_list in intervalised_performance.items():
        books_mean_performance = sum(books_performance_list)/len(books_performance_list)
        total_dates_list.append(books_date)
        total_performance_list.append(books_mean_performance)

    total_performance_list = [x for _, x in sorted(zip(total_dates_list, total_performance_list), key=lambda pair: pair[0])]
    total_dates_list = sorted(total_dates_list)

    total_date_performance_dict =  dict(zip(total_dates_list,total_performance_list))
    return total_date_performance_dict



