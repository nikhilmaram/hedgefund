"""Run functions in all files."""

import pandas as pd
import employee
import processing_all_files
import network
import config as cfg
import time

import plot
import misc
import performance
import sentiment


if __name__ == "__main__":
    pd.set_option('display.max_colwidth', -1)
    business_sentiment_src_dir = "/Users/sainikhilmaram/Desktop/OneDrive/UCSB_courses/project/hedgefund_analysis/data/sentiment_business/"
    personal_sentiment_src_dir = "/Users/sainikhilmaram/Desktop/OneDrive/UCSB_courses/project/hedgefund_analysis/data/sentiment_personal/"

    # =========================================================================
    # ==================== Processing All Files================================
    # =========================================================================

    # # Modifying files to have necessary information.
    # processing_all_files.modify_im_dfs_filelist_multiprocess(business_sentiment_src_dir,cfg.SENTIMENT_BUSINESS,16)
    # processing_all_files.modify_im_dfs_filelist_multiprocess(personal_sentiment_src_dir, cfg.SENTIMENT_PERSONAL, 8)

    # =========================================================================
    # Computing K-core values from the business and personal files.
    # =========================================================================s

    # processing_all_files.compute_kcore_values_filelist_multiprocess(cfg.SENTIMENT_PERSONAL,cfg.KCORE_PERSONAL,8)
    # processing_all_files.compute_kcore_values_filelist_multiprocess(cfg.SENTIMENT_BUSINESS, cfg.KCORE_BUSINESS, 8)
    # processing_all_files.compute_kcore_values_filelist_multiprocess(cfg.SENTIMENT_JOINT, cfg.KCORE_JOINT, 16)

    # processing_all_files.compute_kcore_values_filelist_multiprocess(cfg.SENTIMENT_PERSONAL,cfg.KCORE_PERSONAL_TOTAL,8,False)
    # processing_all_files.compute_kcore_values_filelist_multiprocess(cfg.SENTIMENT_BUSINESS, cfg.KCORE_BUSINESS_TOTAL, 8,False)
    # processing_all_files.compute_kcore_values_filelist_multiprocess(cfg.SENTIMENT_JOINT, cfg.KCORE_JOINT_TOTAL, 8,False)


    # =========================================================================
    # ==================== Employee Related ===================================
    # =========================================================================

    # employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
    # employee.create_employee_hierarchy(employee_dict)

    # =========================================================================
    # ==================== Building Network ===================================
    # =========================================================================

    # network.create_matrix(cfg.IM_TEST_FILE,False)
    # network.create_graph(cfg.IM_TEST_FILE,in_network=False)
    #


    # =========================================================================
    # ====================Sentiment functions==================================
    # =========================================================================

    # im_df = pd.read_csv(cfg.SENTIMENT_PERSONAL+"/im_df_week250.csv")
    # sent_sentiment,recv_sentiment = sentiment.sentiment_given_user(im_df,"carter_richard",in_network=False)
    # print("Sent_sentiment : {0}, Receive Sentiment : {1}".format(sent_sentiment,recv_sentiment))

    # im_df = pd.read_csv(cfg.IM_TEST_FILE)
    # sentiment.sentiment_given_user_list(im_df,["wade_peter","sherrick_michael"],in_network=True)

    # # generate sentiment from all files

    # employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)
    # employee_dict = employee.create_employee_hierarchy(employee_dict)
    # subordinates_list = employee.subordinates_given_employee(employee_dict, "winham_christopher")
    # sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict = sentiment.\
    #     compute_sentiments_from_filelist_multiproc(cfg.SENTIMENT_PERSONAL,subordinates_list,1,True,125,126)
    # #
    # print(sent_sentiment_dict)



