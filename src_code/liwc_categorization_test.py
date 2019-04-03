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


import config as cfg
import liwc_categorization as liwc
pd.set_option('display.max_colwidth', -1)

class LiwcCategorizationModuleTest(unittest.TestCase):
    # =========================================================================
    # ==================== Checks if category dictionary is created============
    # =========================================================================
    def test_compute_required_categories_from_text(self):
        gettysburg = '''Four score and seven years ago our fathers brought forth on
              this continent a new nation, conceived in liberty, and dedicated to the
              proposition that all men are created equal. Now we are engaged in a great
              civil war, testing whether that nation, or any nation so conceived and so
              dedicated, can long endure. We are met on a great battlefield of that war.
              We have come to dedicate a portion of that field, as a final resting place
              for those who here gave their lives that that nation might live. It is
              altogether fitting and proper that we should do this.'''


        category_dict_obs = liwc.compute_required_categories_from_text(gettysburg)
        category_dict_exp = {'cognitive_process': 0.0784313725490196,"insight":0,"causation":0.00980392156862745,"certainity":0.0196078431372549,
                            "discrepancy":0.00980392156862745,"tentativeness":0.029411764705882353,"differentiation":0.0196078431372549,
                            "affect_process":0.0784313725490196,"positive_emotion":0.049019607843137254,
                            "negative_emotion":0.029411764705882353,"anxiety":0,"anger":0.029411764705882353, "sadness":0}

        self.assertDictEqual(category_dict_exp,category_dict_obs)

    # =========================================================================
    # ==================== Checks if LIWC categories has been computed. =====
    # =========================================================================
    def test_compute_liwc_user_messages(self):
        df = pd.read_csv(cfg.IM_TEST_FILE)
        sent_message_liwc_dict_obs,recv_message_liwc_dict_obs = liwc.compute_liwc_user_messages(df,"sherrick_michael",complete_network=True)

        sent_message_liwc_dict_exp = liwc.compute_required_categories_from_text("1 sec 2 min")
        recv_message_liwc_dict_exp = liwc.compute_required_categories_from_text("call me when free I would like to buy hig here")


        self.assertDictEqual(sent_message_liwc_dict_exp,sent_message_liwc_dict_obs)
        self.assertDictEqual(recv_message_liwc_dict_exp,recv_message_liwc_dict_obs)

    # ===========================================================================================
    # ==================== Checks if LIWC categories has been computed.(Multiprocess) =========
    # ===========================================================================================

    def test_compute_liwc_cstegories_from_filelist_multiproc(self):
        total_sent_category_dict_obs, total_recv_category_dict_obs, total_within_category_dict_obs = \
            liwc.compute_liwc_categories_from_filelist_multiproc(cfg.TEST_DIR,["sherrick_michael"],4,complete_network=True,end_week=400)
        date = "08-03-2006"
        total_sent_category_dict_exp ={}; total_recv_category_dict_exp = {}; total_within_category_dict_exp = {}

        total_sent_category_dict_exp[date] = liwc.compute_required_categories_from_text("1 sec 2 min")
        total_recv_category_dict_exp[date] = liwc.compute_required_categories_from_text("call me when free I would like to buy hig here")
        total_within_category_dict_exp[date] = liwc.compute_required_categories_from_text("")

        self.assertDictEqual(total_sent_category_dict_exp, total_sent_category_dict_obs)
        self.assertDictEqual(total_recv_category_dict_exp, total_recv_category_dict_obs)
        self.assertDictEqual(total_within_category_dict_exp, total_within_category_dict_obs)
