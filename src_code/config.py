#############################################################
### PREDEFINED VARIABLES
#############################################################
TOTAL_EMPLOYEES = 297

#############################################################
### DIRECTIORIES
#############################################################
USER_DATA_DIR               = "../data/user_data/"
GENERATED_DATA_DIR          = "../data/generated_data/"
TEST_DIR                    = "../data/test_data/"
PLOTS_DIR                   = "../plots/"

#############################################################
### USER DATA FILES
#############################################################

EMPLOYEE_MASTER_FILE        = USER_DATA_DIR + "/Employee_master.xlsx"
ADDRESS_LINK_FILE           = USER_DATA_DIR + "/Address_linkfile.txt"
PERFORMANCE_FILE            = USER_DATA_DIR + "/PnL_final.csv"
TRADER_BOOK_ACCOUNT_FILE    = USER_DATA_DIR + "/Trader_Book_Account.xlsx"
BOOK_FILE                   = USER_DATA_DIR + "/book.csv"
LIWC_DICTIONARY             = USER_DATA_DIR + "/LIWC2015_English.dic"

#############################################################
### GENERATED DATA FILES
#############################################################

SENTIMENT_PERSONAL          = GENERATED_DATA_DIR + "/sentiment_personal/"
SENTIMENT_BUSINESS          = GENERATED_DATA_DIR + "/sentiment_business/"
SENTIMENT_JOINT             = GENERATED_DATA_DIR + "/sentiment_joint/"

KCORE_PERSONAL              = GENERATED_DATA_DIR + "/kcore/in_network/personal/"
KCORE_BUSINESS              = GENERATED_DATA_DIR + "/kcore/in_network/business/"
KCORE_JOINT                 = GENERATED_DATA_DIR + "/kcore/in_network/joint/"

KCORE_PERSONAL_TOTAL        = GENERATED_DATA_DIR + "/kcore/total/personal/"
KCORE_BUSINESS_TOTAL        = GENERATED_DATA_DIR + "/kcore/total/business/"
KCORE_JOINT_TOTAL           = GENERATED_DATA_DIR + "/kcore/total/joint/"


KCORE_PERSONAL_USER_LIST    = GENERATED_DATA_DIR + "/kcore/user_list_with_PM_RA_TR/personal/"
KCORE_BUSINESS_USER_LIST    = GENERATED_DATA_DIR + "/kcore/user_list_with_PM_RA_TR/business/"
KCORE_JOINT_USER_LIST       = GENERATED_DATA_DIR + "/kcore/user_list_with_PM_RA_TR/joint/"

KCORE_PERSONAL_SAPANSKI_LAWRENCE    = GENERATED_DATA_DIR + "/kcore/sapanski_lawrence/personal/"
KCORE_BUSINESS_SAPANSKI_LAWRENCE    = GENERATED_DATA_DIR + "/kcore/sapanski_lawrence/business/"
KCORE_JOINT_SAPANSKI_LAWRENCE       = GENERATED_DATA_DIR + "/kcore/sapanski_lawrence/joint/"


KCORE_PERSONAL_TEMP    = GENERATED_DATA_DIR + "/kcore/temp/personal/"
KCORE_BUSINESS_TEMP    = GENERATED_DATA_DIR + "/kcore/temp/business/"
KCORE_JOINT_TEMP       = GENERATED_DATA_DIR + "/kcore/temp/joint/"

PKL_FILES                   = GENERATED_DATA_DIR + "/pkl_files/"
BALANCE_PKL_FILES           = GENERATED_DATA_DIR + "/balance/"

GCN_DATA_DIR                = "./gcn/gcn/data"
#############################################################
### TEST DATA FILES
#############################################################

IM_TEST_FILE                = TEST_DIR + "/im_df_week350.csv"