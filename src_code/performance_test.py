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
    # ==================== Checks the performance given MTD values=============
    # =========================================================================

    def test_performance_mtd_values(self):
        df = pd.DataFrame()
        df["book"] = ["ADAM", "MENG", "ADAM", "ADAM", "ADAM", "MENG", "MENG"]
        df["date"] = ["12/12/2008", "12/12/2008", "12/13/2008", "12/13/2008", "1/1/2009", "1/1/2009", "1/15/2009"]
        df["PnL_MTD_adjusted"] = [200, 100, 350, 220, 400, -150, 400]
        df["year_month"] = ["12/2008","12/2008","12/2008","12/2008","1/2009","1/2009","1/2009"]

        dates_list_obs, performance_list_obs = performance.calulate_performance_from_mtd_values(df[df["book"] == "ADAM"],book="ADAM")

        print(dates_list_obs)
        print(performance_list_obs)

    # =========================================================================
    # ==================== Checks the performance given book===================
    # =========================================================================

    def test_performance_given_book(self):
        df = pd.DataFrame()
        df = df.append({"book": "ADAM", "date": "12/12/2008", "PnL_MTD_adjusted": 200, "year_month": "12/2008"}, ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/12/2008", "PnL_MTD_adjusted": 100, "year_month": "12/2008"}, ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/13/2008", "PnL_MTD_adjusted": 350, "year_month": "12/2008"}, ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/13/2008", "PnL_MTD_adjusted": 220, "year_month": "12/2008"}, ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/19/2008", "PnL_MTD_adjusted": 400, "year_month": "12/2008"}, ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/15/2008", "PnL_MTD_adjusted": -150, "year_month": "12/2008"}, ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/17/2008", "PnL_MTD_adjusted": 400, "year_month": "12/2008"}, ignore_index=True)
        # df = df.append({"book": "", "date": "", "PnL_MTD_adjusted": "", "year_month": ""}, ignore_index=True)

        file_path = "./temp.csv"
        df.to_csv(file_path)
        dates_obs,performance_obs = performance.performance_given_book(file_path,"ADAM")

        ## Expected Data frame fed to calulate_performance_from_mtd_values function.
        df_exp = pd.DataFrame()
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008,12,12), "PnL_MTD_adjusted": 200, "year_month": "12/2008"},
                       ignore_index=True)
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008,12,13), "PnL_MTD_adjusted": 220, "year_month": "12/2008"},
                       ignore_index=True)
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008,12,19), "PnL_MTD_adjusted": 400, "year_month": "12/2008"},
                       ignore_index=True)


        dates_exp, performance_exp = performance.calulate_performance_from_mtd_values(df_exp,book="ADAM")

        self.assertEqual(dates_obs,dates_exp)
        for i in range(len(performance_exp)):
            self.assertAlmostEquals(performance_exp[i],performance_obs[i],delta=0.0001)

        # if only_week is enabled.
        dates_obs, performance_obs = performance.performance_given_book(file_path, "ADAM",only_week=True)

        df_exp = pd.DataFrame()
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008, 12, 13), "PnL_MTD_adjusted": 220, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008, 12, 19), "PnL_MTD_adjusted": 400, "year_month": "12/2008"},
            ignore_index=True)

        os.remove(file_path)
        dates_exp, performance_exp = performance.calulate_performance_from_mtd_values(df_exp,book="ADAM")

        self.assertEqual(dates_obs,dates_exp)
        for i in range(len(performance_exp)):
            self.assertAlmostEquals(performance_exp[i],performance_obs[i],delta=0.0001)

    # =========================================================================
    # ==================Checks the performance given book list=================
    # =========================================================================
    def test_performance_given_book_list(self):
        df = pd.DataFrame()
        df = df.append({"book": "ADAM", "date": "12/12/2008", "PnL_MTD_adjusted": 200, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/12/2008", "PnL_MTD_adjusted": 100, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/13/2008", "PnL_MTD_adjusted": 350, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/13/2008", "PnL_MTD_adjusted": 220, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/19/2008", "PnL_MTD_adjusted": 400, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/15/2008", "PnL_MTD_adjusted": -150, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/17/2008", "PnL_MTD_adjusted": 400, "year_month": "12/2008"},
                       ignore_index=True)
        file_path = "./temp.csv"
        df.to_csv(file_path)

        performance_dict_exp = {} ; dates_dict_exp = {}
        dates_dict_obs,performance_dict_obs = performance.performance_given_book_list(file_path,["ADAM","MENG"])
        os.remove(file_path)
        ## ADAM
        df_exp = pd.DataFrame()
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008, 12, 12), "PnL_MTD_adjusted": 200, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008, 12, 13), "PnL_MTD_adjusted": 220, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append({"book": "ADAM", "date": datetime(2008, 12, 19), "PnL_MTD_adjusted": 400, "year_month": "12/2008"},
            ignore_index=True)

        dates_exp, performance_exp = performance.calulate_performance_from_mtd_values(df_exp,book="ADAM")
        performance_dict_exp["ADAM"] = performance_exp
        dates_dict_exp["ADAM"] = dates_exp

        ## MENG
        df_exp = pd.DataFrame()
        df_exp = df_exp.append({"book": "MENG", "date": datetime(2008, 12, 12), "PnL_MTD_adjusted": 100, "year_month": "12/2008"},
                       ignore_index=True)
        df_exp = df_exp.append({"book": "MENG", "date": datetime(2008, 12, 15), "PnL_MTD_adjusted": -150, "year_month": "12/2008"},
                       ignore_index=True)
        df_exp = df_exp.append({"book": "MENG", "date": datetime(2008, 12, 17), "PnL_MTD_adjusted": 400, "year_month": "12/2008"},
                       ignore_index=True)

        dates_exp, performance_exp = performance.calulate_performance_from_mtd_values(df_exp,book="MENG")
        performance_dict_exp["MENG"] = performance_exp
        dates_dict_exp["MENG"] = dates_exp

        self.assertDictEqual(dates_dict_obs,dates_dict_exp)
        for key in performance_dict_exp.keys():
            for i in range(len(performance_dict_exp[key])):
                self.assertAlmostEquals(performance_dict_exp[key][i],performance_dict_obs[key][i],delta=0.0001)
    #
    # =========================================================================
    # ==============Checks combine performance given book list=================
    # =========================================================================
    def test_combine_performance_given_book_list(self):
        df = pd.DataFrame()
        df = df.append({"book": "ADAM", "date": "12/12/2008", "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/12/2008", "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/13/2008", "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/13/2008", "PnL_MTD_adjusted": 600, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "ADAM", "date": "12/19/2008", "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/15/2008", "PnL_MTD_adjusted": -300, "year_month": "12/2008"},
                       ignore_index=True)
        df = df.append({"book": "MENG", "date": "12/17/2008", "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
                       ignore_index=True)

        file_path = "./temp.csv"
        df.to_csv(file_path)

        dates_dict_obs, performance_dict_obs = performance.performance_given_book_list(file_path, ["ADAM", "MENG"])
        total_date_performance_dict_obs = performance.combine_performance_given_book_list(dates_dict_obs,performance_dict_obs)

        df_exp = pd.DataFrame()
        df_exp = df_exp.append(
            {"book": "ADAM", "date": datetime(2008, 12, 12), "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append(
            {"book": "ADAM", "date": datetime(2008, 12, 13), "PnL_MTD_adjusted": 600, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append(
            {"book": "ADAM", "date": datetime(2008, 12, 19), "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
            ignore_index=True)

        dates_exp_1, performance_exp_1 = performance.calulate_performance_from_mtd_values(df_exp,book="ADAM")

        df_exp = pd.DataFrame()
        df_exp = df_exp.append(
            {"book": "MENG", "date": datetime(2008, 12, 12), "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append(
            {"book": "MENG", "date": datetime(2008, 12, 15), "PnL_MTD_adjusted": -300, "year_month": "12/2008"},
            ignore_index=True)
        df_exp = df_exp.append(
            {"book": "MENG", "date": datetime(2008, 12, 17), "PnL_MTD_adjusted": 300, "year_month": "12/2008"},
            ignore_index=True)

        dates_exp_2, performance_exp_2 = performance.calulate_performance_from_mtd_values(df_exp,book="MENG")

        total_dates_exp = [datetime(2008,12,12,0,0), datetime(2008,12,13,0,0), datetime(2008,12,15,0,0),datetime(2008,12,17,0,0),datetime(2008,12,19,0,0)]
        total_performance_exp = [(performance_exp_1[0] + performance_exp_2[0])/2.0,performance_exp_1[1],performance_exp_2[1],performance_exp_2[2],
                                 performance_exp_1[2]]

        self.assertEqual(total_dates_exp, list(total_date_performance_dict_obs.keys()))
        total_performance_obs = list(total_date_performance_dict_obs.values())
        for i in range(len(total_performance_exp)):
            self.assertAlmostEquals(total_performance_exp[i],total_performance_obs[i],delta=0.0001)