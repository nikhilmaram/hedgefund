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
import check_balance_theory
import employee

pd.set_option('display.max_colwidth', -1)

class CheckBalanceTheoryModuleTest(unittest.TestCase):
    # =========================================================================
    # ==================== Checks if category dictionary is created============
    # =========================================================================
    def test_create_matrix_for_balance_theory(self):
        user_list = ["sherrick_michael", "wade_peter", "cross_brent"]
        sherrick_michael_id  = employee.employee_username_to_id_dict["sherrick_michael"]
        wade_peter_id = employee.employee_username_to_id_dict["wade_peter"]
        cross_brent_id = employee.employee_username_to_id_dict["cross_brent"]
        user_count = 297
        im_df = pd.read_csv(cfg.IM_TEST_FILE)

        # sentiment_matrix_exp = [[[] for x in range(user_count + 1)] for y in range(user_count + 1)]
        # sentiment_matrix_exp[sherrick_michael_id][cross_brent_id].append(-1)
        # sentiment_matrix_exp[cross_brent_id][sherrick_michael_id].append(1)
        # sentiment_matrix_exp[sherrick_michael_id][cross_brent_id].append(1)
        # sentiment_matrix_exp[wade_peter_id][sherrick_michael_id].append(1)
        # sentiment_matrix_obs, _ = check_balance_theory.create_matrix_for_balance_theory(im_df, True, user_list)
        # self.assertEqual(sentiment_matrix_exp, sentiment_matrix_obs)

        message_matrix_exp = [[0 for x in range(user_count + 1)] for y in range(user_count + 1)]  ## indexing from 1.
        message_matrix_exp[sherrick_michael_id][cross_brent_id] = 0
        message_matrix_exp[cross_brent_id][sherrick_michael_id] = 0.5
        message_matrix_exp[wade_peter_id][sherrick_michael_id] = 0.5
        print(cross_brent_id)

        message_matrix_obs, _ = check_balance_theory.create_matrix_for_balance_theory(im_df, True, user_list)
        self.assertEqual(message_matrix_exp, message_matrix_obs)


