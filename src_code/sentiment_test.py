from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import mpl_toolkits
import pandas as pd
import unittest
import os
from datetime import datetime
from parameterized import parameterized

import sentiment
import config as cfg
pd.set_option('display.max_colwidth', -1)

class SentimentModuleTest(unittest.TestCase):
    # =========================================================================
    # ==================== Checks the resultant sentiment =====================
    # =========================================================================
    def test_resultant_sentiment(self):
        sentiment_list = [1,1,-1,-1,0,0,1,1,-1,-1,1]
        sentiment_exp = 0.1
        sentiment_obs = sentiment.resultant_sentiment(sentiment_list)
        self.assertEqual(sentiment_exp,sentiment_obs)

    # =========================================================================
    # ==================== Checks the sentiment given user ====================
    # =========================================================================

    def test_sentiment_given_user(self):
        im_df  = pd.read_csv(cfg.IM_TEST_FILE)

        sent_sentiment_obs, recv_sentiment_obs = sentiment.sentiment_given_user(im_df,"wade_peter")
        sent_sentiment_exp = 0.5
        recv_sentiment_exp = 0

        self.assertEqual(sent_sentiment_exp,sent_sentiment_obs)
        self.assertEqual(recv_sentiment_exp,recv_sentiment_obs)

        sent_sentiment_obs, recv_sentiment_obs = sentiment.sentiment_given_user(im_df, "wade_peter",in_network=False)
        sent_sentiment_exp = 0.25
        recv_sentiment_exp = 0.5

        self.assertEqual(sent_sentiment_exp, sent_sentiment_obs)
        self.assertEqual(recv_sentiment_exp, recv_sentiment_obs)

    # =========================================================================
    # =================Checks the sentiment given user list====================
    # =========================================================================

    def test_sentiment_given_user_list(self):
        im_df = pd.read_csv(cfg.IM_TEST_FILE)


        sent_sentiment_obs, recv_sentiment_obs, within_sentiment_obs = sentiment.\
            sentiment_given_user_list(im_df,["wade_peter","sherrick_michael"],in_network=True)

        sent_sentiment_exp = sentiment.resultant_sentiment([-1,1])
        recv_sentiment_exp = sentiment.resultant_sentiment([1])
        within_sentiment_exp = sentiment.resultant_sentiment([1])

        self.assertEqual(sent_sentiment_exp, sent_sentiment_obs)
        self.assertEqual(recv_sentiment_exp, recv_sentiment_obs)
        self.assertEqual(within_sentiment_exp, within_sentiment_obs)

        sent_sentiment_obs, recv_sentiment_obs, within_sentiment_obs = sentiment.\
            sentiment_given_user_list(im_df, ["wade_peter", "sherrick_michael"],in_network=False)

        sent_sentiment_exp = sentiment.resultant_sentiment([-1, 1, 1])
        recv_sentiment_exp = sentiment.resultant_sentiment([1])
        within_sentiment_exp = sentiment.resultant_sentiment([1])

        self.assertEqual(sent_sentiment_exp, sent_sentiment_obs)
        self.assertEqual(recv_sentiment_exp, recv_sentiment_obs)
        self.assertEqual(within_sentiment_exp, within_sentiment_obs)

        sent_sentiment_obs, recv_sentiment_obs, within_sentiment_obs = sentiment.\
            sentiment_given_user_list(im_df,["wade_peter","sherrick_michael"],in_network=False,complete_network=True)

        sent_sentiment_exp = sentiment.resultant_sentiment([-1,1,1,-1,1])
        recv_sentiment_exp = sentiment.resultant_sentiment([1,1])
        within_sentiment_exp = sentiment.resultant_sentiment([1])

        self.assertEqual(sent_sentiment_exp, sent_sentiment_obs)
        self.assertEqual(recv_sentiment_exp, recv_sentiment_obs)
        self.assertEqual(within_sentiment_exp, within_sentiment_obs)

    # =========================================================================
    # ================Checks if the sentiment dictionary is created correctly==
    # =========================================================================

    def test_compute_sentiments_from_filelist(self):
        return_dict = {}
        sent_sentiment_dict_obs, recv_sentiment_dict_obs, within_sentiment_dict_obs = sentiment.\
            compute_sentiments_from_filelist_multiproc(cfg.TEST_DIR, ["wade_peter","sherrick_michael"],1,in_network=True,end_week=400)

        # [[sent_sentiment_dict_obs, recv_sentiment_dict_obs, within_sentiment_dict_obs]] = return_dict.values()

        sent_sentiment_dict_exp = {} ; recv_sentiment_dict_exp = {} ; within_sentiment_dict_exp = {}
        date = "08-03-2006"
        # date = datetime.strptime(date, '%m-%d-%Y')
        sent_sentiment_dict_exp[date] = sentiment.resultant_sentiment([-1,1])
        recv_sentiment_dict_exp[date] = sentiment.resultant_sentiment([1])
        within_sentiment_dict_exp[date] = sentiment.resultant_sentiment([1])

        print(sent_sentiment_dict_obs)
        self.assertDictEqual(sent_sentiment_dict_exp,sent_sentiment_dict_obs)
        self.assertDictEqual(recv_sentiment_dict_exp,recv_sentiment_dict_obs)
        self.assertDictEqual(within_sentiment_dict_exp,within_sentiment_dict_obs)

        sent_sentiment_dict_obs, recv_sentiment_dict_obs, within_sentiment_dict_obs = sentiment. \
            compute_sentiments_from_filelist_multiproc(cfg.TEST_DIR, ["wade_peter", "sherrick_michael"], 1, in_network=False,
                                                       complete_network=True,end_week=400)

        sent_sentiment_dict_exp = {}; recv_sentiment_dict_exp = {}; within_sentiment_dict_exp = {}
        sent_sentiment_dict_exp[date] = sentiment.resultant_sentiment([-1,1,1,-1,1])
        recv_sentiment_dict_exp[date] = sentiment.resultant_sentiment([1,1])
        within_sentiment_dict_exp[date] = sentiment.resultant_sentiment([1])

        self.assertDictEqual(sent_sentiment_dict_exp, sent_sentiment_dict_obs)
        self.assertDictEqual(recv_sentiment_dict_exp, recv_sentiment_dict_obs)
        self.assertDictEqual(within_sentiment_dict_exp, within_sentiment_dict_obs)

        sent_sentiment_dict_obs, recv_sentiment_dict_obs, within_sentiment_dict_obs = sentiment. \
            compute_sentiments_from_filelist_multiproc(cfg.TEST_DIR, ["wade_peter", "sherrick_michael"], 1, in_network=True,
                                                       end_week=400,only_week=True)
