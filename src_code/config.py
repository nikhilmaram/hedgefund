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


#############################################################
### GENERATED DATA FILES
#############################################################

SENTIMENT_PERSONAL          = GENERATED_DATA_DIR + "/sentiment_personal/"
SENTIMENT_BUSINESS          = GENERATED_DATA_DIR + "/sentiment_business/"
KCORE_PERSONAL              = GENERATED_DATA_DIR + "/kcore/personal/"
KCORE_BUSINESS              = GENERATED_DATA_DIR + "/kcore/business/"


#############################################################
### TEST DATA FILES
#############################################################

IM_TEST_FILE                = TEST_DIR + "/im_df_week350.csv"