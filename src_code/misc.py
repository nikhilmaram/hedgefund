import nltk as nk
import pandas as pd
import csv
from datetime import  datetime,timedelta,date
from typing import List
from typing import Tuple
import os
import pickle

def create_book_file(file):
    """Creates a book file from reading Trader Book Account xlsx"""
    df = pd.read_excel(file)
    stock_list = []

    for i in range(len(df)):
        for stock in df['Account'][i].split(','):
            if not stock.strip() == "":
                stock_list.append(stock.strip())

    stock_list = sorted(set(stock_list))
    stockfile = open("../data/user_data/book.csv", 'w')
    for stock in stock_list:
        stockfile.write(stock+"\n")
    stockfile.close()

def read_book_file(file_path:str) -> List:
    """Reads the book file and writes into list.

    Args:
        file_path : path to stock file.
    Returns:
        stock_list : list of books present.
    """
    stock_list=[]
    f = open(file_path,"r")
    f.readline()
    for line in f.readlines():
        stock_list.append(line.strip("\n").strip())

    return stock_list

def write_dict_in_file(inp_dict : dict,file_path:str):
    """Writes the dictionaries into files.

    Args:
        inp_dict : Input dictionary.
        file_path : file to where the dictionary needs to be written.

    Returns:
        None.
    """
    with open(file_path,"wb") as handle:
        pickle.dump(inp_dict,handle)

def read_file_into_dict(file_path:str) -> dict:
    """Reads file into dictionary.

    Args:
        file_path : Path to file to read data from..

    Returns:
        output_dict : output dictionary.
    """
    output_dict = {}
    with open(file_path,"rb") as handle:
        output_dict = pickle.load(handle)

    return output_dict


def calculate_week(curr_date):
    """
    Args:
        curr_data : input data in dateformat. ## date(yyyy,mm,dd)

    Returns:
        week_num : Week number corresponding to week number corresponfing to input date

    """
    # curr_date = date(year, month, day)
    start_date = date(2006, 8, 3)

    start_monday = (start_date - timedelta(days=start_date.weekday()))
    curr_monday = (curr_date - timedelta(days=curr_date.weekday()))
    week_num = int((curr_monday - start_monday).days / 7)
    # print("Week Number : {0}".format(week_num))
    return week_num

def calculate_date(start_date = date(2006, 7, 31),week_num = 1):
    """
    Args:
        start_date : start date
        week_num : ween number
    Returns:
        end_date : date from start date for with a gap of week_num weeks.
    """
    end_date = start_date + timedelta(days=week_num*7)
    return end_date



def writing_list_into_file(inp_list:List,file_path:str):
    """ To write an element into a new line in a file.
    Args:
        inp_list : Input list to be written into file.
        file_path : path to file where the input list is written.

    Returns:
        None.

    """
    with open(file_path, "w") as myfile:
        for ele in inp_list:
            myfile.write("%d" % ele)
            myfile.write(",")
        myfile.write("\n")
    myfile.close()

def df_week_df_day(inp_df : pd.DataFrame) -> dict:
    """Splits weekly dataframe to daily dataframe.

    Args:
        inp_df : Input weekly dataframe.

    Returns:
        df_daily_dict : Daily Dataframe dict. key - date, value - Dataframe.
    """
    df_daily_dict = {}
    inp_df["day"] = inp_df["time_stamp"].apply(lambda x: datetime.strptime(x, '%m-%d-%yT%H:%M:%S').strftime("%m-%d-%Y"))

    for day, df_grouped in inp_df.groupby("day"):
        df_daily_dict[day] = df_grouped

    return df_daily_dict

def splitting_all_files(dir_path : str, num_process : int, start_week : int = 0, end_week :int = 264) -> List[List[str]]:
    """Splits all files into multiple lists for multi-processing.

    Args:
        dir_path :  Path to directory where all the files are present.
        num_process : Number of lists to split the files into.
        start_week : starting week of files to be considered.
        end_week : Ending week of the to be considered.

    Returns:
        List of Lists of file names.
    """

    file_names_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            if filename.startswith("im_df"):
            ## Directory path is not appended because we will be using filename alone in our later computations.
                week_num = int(filename.split('.')[0].split('_')[-1][4:])
                if((start_week <= week_num ) and (week_num <= end_week)):
                    file_names_list.append(filename)

    file_names_list = sorted(file_names_list, key=lambda x: int(x.split('.')[0].split('_')[-1][4:]))
    file_lists = list(file_names_list[i::num_process] for i in range(num_process))

    return file_lists

def common_keys(dict_1 : dict , dict_2 : dict) -> Tuple[dict,dict]:
    """ Returns two dictionary which has same keys.
    Args:
        dict_1: A dictionary ; key - date, value - performance/sentiment.
        dictt_2 : A dictionary ; key - date, value - performance/sentiment.

    Returns:
        out_dict_1 : subset of dict_1 which has common keys with dict_2.
        out_dict_1 : subset of dict_2 which has common keys with dict_1.
    """

    keys_1 = dict_1.keys()
    keys_2 = dict_2.keys()

    out_dict_1 = {} ; out_dict_2 = {}

    common_keys = list(set(keys_1).intersection(set(keys_2)))
    for key in common_keys :
        out_dict_1[key] = dict_1[key]
        out_dict_2[key] = dict_2[key]

    return out_dict_1, out_dict_2

def change_key_string_key_date(inp_dict):
    """Converts the keys of dictionary from string to dates.

    Args:
        inp_dict : Input dictionary whose keys are strings.

    Returns:
        output_dict : output dictionary whose keys are dates.
    """

    output_dict = {}

    for key,value in inp_dict.items():
        output_dict[datetime.strptime(key,"%Y-%m-%d")] = value

    return output_dict

def get_list_from_dicts_sorted_dates(dict_1 : dict,dict_2 : dict)-> Tuple[List,List]:
    """Returns two lists whose values are sorted in dates.


    Args:
        dict_1,dict_2: Dictionaries whose keys are dates.

    Returns:
        Lists whose elements are values in dict with sorted key values.
    """
    output_list_1 = []
    output_list_2 = []

    dates_list = sorted(dict_1.keys())

    for curr_date in dates_list:
        output_list_1.append(dict_1[curr_date])
        output_list_2.append(dict_2[curr_date])

    return output_list_1,output_list_2
