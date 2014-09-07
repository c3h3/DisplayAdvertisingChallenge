
import os


SOURCE_DATA_DIR = "/home/c3h3/c3h3works/Kaggles/GithubRepos/DisplayAdvertisingChallenge/data/"
MAX_N_DATA_COLUMN_DIVIDERS = 5


try:
    from local_settings import *
    
    # local_settings.py
    # SOURCE_DATA_DIR = "/home/c3h3/c3h3works/HunKaggle/DisplayAdvertisingChallenge/data/"
    # MAX_N_DATA_COLUMN_DIVIDERS = 12
    
except:
    pass
    


assert os.path.exists(SOURCE_DATA_DIR)

# if not os.path.exists(SOURCE_DATA_DIR):
#     os.mkdir(SOURCE_DATA_DIR)
    


TRAINING_DATA_PATH = os.path.join(SOURCE_DATA_DIR, "train.csv")
TESTING_DATA_PATH = os.path.join(SOURCE_DATA_DIR, "test.csv")

# TRAINING_COLUMNS_PATH = os.path.join(SOURCE_DATA_DIR, "train_cols")
# if not os.path.exists(TRAINING_COLUMNS_PATH):
#     os.mkdir(TRAINING_COLUMNS_PATH)
#   
# TESTING_COLUMNS_PATH = os.path.join(SOURCE_DATA_DIR, "test_cols")
# if not os.path.exists(TESTING_COLUMNS_PATH):
#     os.mkdir(TESTING_COLUMNS_PATH)


TRAINING_BLZ_PATH = os.path.join(SOURCE_DATA_DIR, "train_blzs")
if not os.path.exists(TRAINING_BLZ_PATH):
    os.mkdir(TRAINING_BLZ_PATH)
  
TESTING_BLZ_PATH = os.path.join(SOURCE_DATA_DIR, "test_blzs")
if not os.path.exists(TESTING_BLZ_PATH):
    os.mkdir(TESTING_BLZ_PATH)



with open(TRAINING_DATA_PATH,"r") as rf:
    colnames_line = rf.readline()
    TRAINING_COLUMN_NAMES  = colnames_line.strip().split(",")

with open(TESTING_DATA_PATH,"r") as rf:
    colnames_line = rf.readline()
    TESTING_COLUMN_NAMES  = colnames_line.strip().split(",")





