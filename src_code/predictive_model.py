import pandas as pd
from datetime import datetime
from typing import List
from typing import Tuple
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import performance
import config as cfg
import misc

def predictive_model_for_performance(book_name):

    performance_mean_dict_book = misc.read_file_into_dict(cfg.PKL_FILES + "/performance_mean.pkl")

    dates_list, performance_list = performance.performance_given_book(cfg.PERFORMANCE_FILE, book_name, start_week=123, end_week= 263, only_week= False)

    performance_predict_list = []
    for i in range(len(dates_list)-1):
        if (performance_list[i+1] > performance_list[i]) :
            performance_predict_list.append(1)
        else:
            performance_predict_list.append(0)

    book_performance_mean = performance_mean_dict_book[book_name]
    features = [x-book_performance_mean for x in performance_list][:-1]
    labels = performance_predict_list
    features = features.reshape(1,-1)

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=0)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(features_train, labels_train)
    score = logisticRegr.score(features_test, labels_test)
    print(score)




if __name__ == "__main__":
    predictive_model_for_performance("ADAM")