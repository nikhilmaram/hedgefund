from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import mpl_toolkits
import pandas as pd
import unittest
import os
from parameterized import parameterized
from datetime import  datetime,timedelta,date

import performance
import misc



class PerformanceModuleTest(unittest.TestCase):
    # =========================================================================
    # ==================== Checks the performance given book===================
    # =========================================================================

    def test_performance_given_book(self):
        df = pd.DataFrame()
        df["book"] = ["ADAM","FTEI","ADAM","ADAM","ADAM","FTEI","FTEI"]
        df["date"] = ["12/12/2008","12/12/2008","12/13/2008","12/13/2008","12/19/2008","12/15/2008","12/17/2008"]
        df["PnL_MTD_adjusted"] =[200,100,350,220,400,-150,400]
        file_path = "./temp.csv"
        df.to_csv(file_path)

        dates_obs,performance_obs = performance.performance_given_book(file_path,"ADAM")
        performance_mean = 820/3.0
        performance_exp = [200/performance_mean,220/performance_mean,400/performance_mean]
        self.assertEqual(performance_obs,performance_exp)

        dates_exp = [datetime(2008, 12, 12,0,0), datetime(2008, 12, 13,0,0), datetime(2008, 12, 19,0,0)]
        self.assertEqual(dates_obs,dates_exp)

        ## if only_week is enabled.
        dates_obs, performance_obs = performance.performance_given_book(file_path, "ADAM",only_week=True)
        os.remove(file_path)
        performance_mean = 620/2.0
        performance_exp = [220/performance_mean,400/performance_mean]
        self.assertEqual(performance_obs,performance_exp)

        dates_exp = [datetime(2008, 12, 13,0,0), datetime(2008, 12, 19,0,0)]
        self.assertEqual(dates_obs,dates_exp)

    # =========================================================================
    # ==================Checks the performance given book list=================
    # =========================================================================
    def test_performance_given_book_list(self):
        df = pd.DataFrame()
        df["book"] = ["ADAM", "FTEI", "ADAM", "ADAM", "ADAM", "FTEI", "FTEI"]
        df["date"] = ["12/12/2008", "12/12/2008", "12/13/2008", "12/13/2008", "12/19/2008", "12/15/2008", "12/17/2008"]
        df["PnL_MTD_adjusted"] = [200, 100, 350, 220, 400, -150, 400]
        file_path = "./temp.csv"
        df.to_csv(file_path)

        performance_dict_exp = {} ; dates_dict_exp = {}
        dates_dict_obs,performance_dict_obs = performance.performance_given_book_list(file_path,["ADAM","FTEI"])
        os.remove(file_path)
        ## ADAM
        performance_mean = 820 / 3.0
        performance_exp = [200 / performance_mean, 220 / performance_mean, 400 / performance_mean]
        performance_dict_exp["ADAM"] = performance_exp
        dates_exp = [datetime(2008, 12, 12, 0, 0), datetime(2008, 12, 13, 0, 0), datetime(2008, 12, 19, 0, 0)]
        dates_dict_exp["ADAM"] = dates_exp

        ## FTEI
        performance_mean = 350/3.0
        performance_exp = [100 / performance_mean, -150 / performance_mean, 400 / performance_mean]
        performance_dict_exp["FTEI"] = performance_exp
        dates_exp = [datetime(2008, 12, 12, 0, 0), datetime(2008, 12, 15, 0, 0), datetime(2008, 12, 17, 0, 0)]
        dates_dict_exp["FTEI"] = dates_exp

        self.assertDictEqual(dates_dict_obs,dates_dict_exp)
        self.assertDictEqual(performance_dict_obs,performance_dict_exp)

    # =========================================================================
    # ==============Checks combine performance given book list=================
    # =========================================================================
    def test_combine_performance_given_book_list(self):
        df = pd.DataFrame()
        df["book"] = ["ADAM", "FTEI", "ADAM", "ADAM", "ADAM", "FTEI", "FTEI"]
        df["date"] = ["12/12/2008", "12/12/2008", "12/13/2008", "12/13/2008", "12/19/2008", "12/15/2008", "12/17/2008"]
        df["PnL_MTD_adjusted"] = [300, 300, 300, 600, 300, -300, 300]
        file_path = "./temp.csv"
        df.to_csv(file_path)

        dates_dict_obs, performance_dict_obs = performance.performance_given_book_list(file_path, ["ADAM", "FTEI"])
        total_date_performance_dict_obs = performance.combine_performance_given_book_list(dates_dict_obs,performance_dict_obs)

        total_dates_exp = [datetime(2008,12,12,0,0), datetime(2008,12,13,0,0), datetime(2008,12,15,0,0),datetime(2008,12,17,0,0),datetime(2008,12,19,0,0)]
        total_performance_exp = [1.875 ,1.5,-3.0 ,3.0,0.75]

        self.assertEqual(total_dates_exp, list(total_date_performance_dict_obs.keys()))
        self.assertEqual(total_performance_exp,list(total_date_performance_dict_obs.values()))

        ##  only_week = True
        dates_dict_obs, performance_dict_obs = performance.performance_given_book_list(file_path, ["ADAM", "FTEI"],only_week=True)

        total_date_performance_dict_obs = performance.combine_performance_given_book_list(dates_dict_obs,
                                                                                                 performance_dict_obs,only_week=True)

        # print(performance_dict_obs)
        total_dates_exp = [date(2008,12,8), date(2008,12,15)]
        total_performance_exp = [7/6,5/6]
        self.assertEqual(total_dates_exp,list(total_date_performance_dict_obs.keys()))
        for i in range(len(list(total_date_performance_dict_obs.values()))):
            ## almost equals because division precision is different.
            self.assertAlmostEquals(total_performance_exp[i],list(total_date_performance_dict_obs.values())[i],places=2)






