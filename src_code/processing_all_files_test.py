from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import mpl_toolkits
import pandas as pd
import unittest
import os
from parameterized import parameterized
from pandas.testing import assert_frame_equal

import config as cfg
import processing_all_files
import sentiment

class ProcessingAllFilesModuleTest(unittest.TestCase):
    pd.set_option('display.max_colwidth', -1)

    # =========================================================================
    # ================Checks if the dataframe is modified correctly============
    # =========================================================================

    def test_modify_im_dfs(self):
        inp_df = pd.DataFrame(columns=["Unnamed: 0","sender","sender_buddy","receiver","receiver_buddy","time_stamp","content","classify","sentiment"])
        inp_df = inp_df.append({"Unnamed: 0":0,"sender":"5bc419633c1910982dd4f94740ef584f","sender_buddy":"rachacoso@diamondbackcap.com",
                       "receiver": "afe0a92b096535cd61516ffb2e18f6c9","receiver_buddy": "aghania2@yahoo.com", "time_stamp": "01-02-11T03:05:10",
                       "content": "Hello", "classify": 1 , "sentiment": 1},ignore_index=True)

        inp_df = inp_df.append({"Unnamed: 0": 1, "sender": "5bc419633c1910982dd4f94740ef584f","sender_buddy": "nikhilmaram@gmail.com",
                       "receiver": "afe0a92b096535cd61516ffb2e18f6c9", "receiver_buddy": "aghania2@yahoo.com","time_stamp": "01-02-11T03:05:10",
                       "content": "World", "classify": 0, "sentiment": 0},ignore_index=True)

        src_path = "./tmp_src.csv"
        dst_path = "./tmp_dst.csv"

        expected_df = pd.DataFrame(columns=["sender_user_name","receiver_user_name","content","time_stamp","day","sender_in_network","receiver_in_network","classify","sentiment"])
        expected_df = expected_df.append({"sender_user_name": "achacoso_ralph","receiver_user_name": "ameziane_ghania","content": "Hello",
                            "time_stamp": "01-02-11T03:05:10", "day":"01-02-2011","sender_in_network": 1,"receiver_in_network":0,
                            "classify": 1, "sentiment": 1}, ignore_index=True)

        expected_df = expected_df.append({"sender_user_name": "nikhilmaram","receiver_user_name": "ameziane_ghania","content": "World",
                             "time_stamp": "01-02-11T03:05:10",  "day":"01-02-2011","sender_in_network": 0,"receiver_in_network":0,
                             "classify": 0, "sentiment": 0}, ignore_index=True)

        inp_df.to_csv(src_path,index=False)
        processing_all_files.modify_im_dfs(src_path,dst_path)
        observed_df = pd.read_csv(dst_path)

        expected_dict = expected_df.to_dict()
        observed_dict = observed_df.to_dict()
        ## Easy to check dictionaries than the dataframes themselves
        os.remove(src_path)
        os.remove(dst_path)
        self.assertDictEqual(observed_dict,expected_dict)








