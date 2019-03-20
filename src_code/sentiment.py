"""Contains all the functions related to sentiment."""
import pandas as pd
from datetime import datetime
from typing import List
from typing import Tuple
import numpy as np
import queue
import multiprocessing
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import config as cfg
import misc

sid = SentimentIntensityAnalyzer()

def sentiment_assign(text):
    """Returns the sentiment given the text.

    Args:
        text : input text

    Returns:
        sentiment : sentiment of the text.
    """
    try:
        sentiment_value = sid.polarity_scores(text)['compound']
    except:
        sentiment_value = 0
    if sentiment_value > 0.05:
        sentiment = 1
    elif sentiment_value < -0.05:
        sentiment = -1
    else:
        sentiment = 0
    return sentiment

def sentiment_classify(dir_path):
    """Creates a sentimnet column to the dataframe based on content.

    Args:
        dir_path : directory where IM files are present.

    Returns:
        None
    """

    file_name_list = misc.splitting_all_files(dir_path,1)[0]

    for file_name in file_name_list:
        if file_name.startswith("im_df"):
            df = pd.read_csv(dir_path + "/"+ file_name,dtype=str)
            df['sentiment'] = df['content'].apply(lambda x: sentiment_assign(x))
            df.to_csv("./sentiment_personal/"+file_name)



def resultant_sentiment(sentiment_list : List) -> float:
    """Resultant sentiment given sentiment list. i.e resultant sentiment value for all the sentiments in the input list.

    Args:
        sentiment_list : list of sentiment values.

    Returns:
        sentiment_value : resultant sentiment value.

    total_sentiment = sum of all sentiment values in the list.
    Sentiment is calculated from total_sentiment/(total_positive_messages + total_negative_messages).
    """

    positive_sentiment_count = sentiment_list.count(1)
    negative_sentiment_count = sentiment_list.count(-1)
    zero_sentiment_count     = sentiment_list.count(0)

    # print(positive_sentiment_count/negative_sentiment_count)
    sentiment = sum(sentiment_list) / (positive_sentiment_count + negative_sentiment_count + 1.0)

    return sentiment

def sentiment_given_user(im_df : pd.DataFrame ,user_name : str, in_network:bool = True) -> Tuple[float,float]:
    """Calculates the sentiment for given user.

    Args:
        im_df : Input Dataframe of IMs.
        user_name : User whose sentiment needs to be calculated.
        in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)

    Returns:
        sent_sentiment : sentiment of send messages from the user.
        recv_sentiment : sentiment of received messages to the user.
    """

    if in_network:
        im_df = im_df[im_df["sender_in_network"] == 1]
        im_df = im_df[im_df["receiver_in_network"] == 1]

    ## dataframe with sender as input user.
    im_df_sent = im_df[im_df["sender_user_name"] == user_name]
    ## dataframe with receiver as input user.
    im_df_recv = im_df[im_df["receiver_user_name"] == user_name]

    sent_sentiment_list = im_df_sent["sentiment"].tolist()
    recv_sentiment_list = im_df_recv["sentiment"].tolist()

    sent_sentiment = resultant_sentiment(sent_sentiment_list)
    recv_sentiment = resultant_sentiment(recv_sentiment_list)

    return sent_sentiment,recv_sentiment

# =========================================================================
# ============= Generate sentiments given user list========================
# =========================================================================

def sentiment_given_user_list(im_df : pd.DataFrame, user_name_list, in_network : bool = True) -> Tuple[float,float,float]:
    """Calculates the combined sentiment of all users in user_name_list to others and within.

    Args:
        im_df : Input Dataframe of IMs.
        user_name_list : User Name list.
        in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)

    Returns:
        sent_sentiment : sentiment of send messages from the user list to others.
        recv_sentiment : sentiment of received messages from others to the user list.
        within_sentiment : sentiment of messages within the users.
    """

    if in_network:
        im_df = im_df[im_df["sender_in_network"] == 1]
        im_df = im_df[im_df["receiver_in_network"] == 1]

    im_df_sent = im_df[im_df["sender_user_name"].isin(user_name_list)]
    ## This might also contains receivers which are in user_name_list, so removing the rows in which receivers are not in user_name_list.
    im_df_sent = im_df_sent[~im_df_sent["receiver_user_name"].isin(user_name_list)]

    im_df_recv = im_df[im_df["receiver_user_name"].isin(user_name_list)]
    im_df_recv = im_df_recv[~im_df_recv["sender_user_name"].isin(user_name_list)]

    im_df_within = im_df[im_df["sender_user_name"].isin(user_name_list)]
    im_df_within = im_df_within[im_df_within["receiver_user_name"].isin(user_name_list)]

    sent_sentiment_list = im_df_sent["sentiment"].tolist()
    recv_sentiment_list = im_df_recv["sentiment"].tolist()
    within_sentiment_list = im_df_within["sentiment"].tolist()

    sent_sentiment = resultant_sentiment(sent_sentiment_list)
    recv_sentiment = resultant_sentiment(recv_sentiment_list)
    within_sentiment = resultant_sentiment(within_sentiment_list)

    return sent_sentiment, recv_sentiment , within_sentiment


# =========================================================================
# ============= Generate sentiments from all files=========================
# =========================================================================

def compute_sentiments_from_filelist(file_list : List, src_dir_path:str, user_name_list:List,
                                     return_dict:dict, in_network:bool = True,only_week:bool=False):
    """Computes sentiments from all files in file_list for the users present in user_name_list.

     Args:
         src_dir_path : path to the input files.
         user_name_list : user names for whose sentiment needs to be calculated.
         in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: All)
         return_dict : To pass the return value for multiprocess invocation.
         only_week           : data is calculated weekly instead of each date.

     sent_sentiment_dict = sentiment dictionary of sent messages. key - date ; value - sent_sentiment_value.
     recv_sentiment_dict = sentiment dictionary of received messages. key - date ; value - recv_sentiment_value.
     within_sentiment_dict = sentiment dictionary of within messages. key - date ; value - within_sentiment_value.
     """
    process_count = multiprocessing.current_process().name

    print("In compute_sentiments_from_filelist function", only_week)
    sent_sentiment_dict = {}
    recv_sentiment_dict = {}
    within_sentiment_dict = {}

    for file_name in file_list:
        # print(file_name)
        file_path = os.path.join(src_dir_path,file_name)
        df = pd.read_csv(file_path)

        if only_week:
            ## sentiment is calculated over week data.
            sent_sentiment, recv_sentiment, within_sentiment = sentiment_given_user_list(df, user_name_list,in_network)
            ## Calculate the monday of the week.
            week_num = int(file_name.split('.')[0].split('_')[-1][4:])
            curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
            sent_sentiment_dict[curr_date] = sent_sentiment
            recv_sentiment_dict[curr_date] = recv_sentiment
            within_sentiment_dict[curr_date] = within_sentiment
        else:
            ## Group the data according to curr_date.
            for curr_date, df_curr_date in df.groupby("day"):
                sent_sentiment, recv_sentiment, within_sentiment = sentiment_given_user_list(df_curr_date,user_name_list,in_network)
                sent_sentiment_dict[curr_date] = sent_sentiment
                recv_sentiment_dict[curr_date] = recv_sentiment
                within_sentiment_dict[curr_date] = within_sentiment

    return_dict[process_count] = [sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict]
    # return sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict


def compute_sentiments_from_filelist_multiproc(src_dir_path : str, user_name_list: List,num_process :int ,
                                               in_network:bool = True, start_week : int = 0,
                                               end_week : int = 264,only_week:bool=False):
    """Runs the compute_sentiments_from_filelist on multiple process.

    Args:
        src_dir_path        : path to the files present in file_list
        user_name_list      : users for whom sentiment is calculated.
        num_process         : Number of process.
        in_network          : messages considered within/outside network.
        start_week          : starting week of files to be considered.
        end_week            : Ending week of the to be considered.
        only_week           : data is calculated weekly instead of each date.

    Returns:
        sent_sentiment_dict = sentiment dictionary of sent messages. key - date ; value - sent_sentiment_value.
        recv_sentiment_dict = sentiment dictionary of received messages. key - date ; value - recv_sentiment_value.
        within_sentiment_dict = sentiment dictionary of within messages. key - date ; value - within_sentiment_value.
    """

    sent_sentiment_dict = {}
    recv_sentiment_dict = {}
    within_sentiment_dict = {}
    list_of_file_list = misc.splitting_all_files(src_dir_path, num_process,start_week,end_week)

    process_list = []

    ## Need a dictionary as we need to collect output from each process.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    count = 1
    for file_list in list_of_file_list:
        p = multiprocessing.Process(target=compute_sentiments_from_filelist,
                                    args=(file_list, src_dir_path, user_name_list,return_dict,in_network,only_week),
                                    name=count)
        process_list.append(p)
        p.start()
        count = count + 1

    for process in process_list:
        process.join()

    for process, sentiment_dicts_list in return_dict.items():
        sent_sentiment_dict.update(sentiment_dicts_list[0])
        recv_sentiment_dict.update(sentiment_dicts_list[1])
        within_sentiment_dict.update(sentiment_dicts_list[2])


    return sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict
