
import os

SOURCE_DATA_DIR = "/home/c3h3/c3h3works/HunKaggle/DisplayAdvertisingChallenge/data/"
MAX_N_DATA_COLUMN_DIVIDERS = 12

TRAINING_DATA_PATH = os.path.join(SOURCE_DATA_DIR, "train.csv")
TESTING_DATA_PATH = os.path.join(SOURCE_DATA_DIR, "test.csv")

TRAINING_COLUMNS_PATH = os.path.join(SOURCE_DATA_DIR, "train_cols")
TAR_TRAINING_COLUMNS_PATH = os.path.join(SOURCE_DATA_DIR, "tar_train_cols")

TESTING_COLUMNS_PATH = os.path.join(SOURCE_DATA_DIR, "test_cols")
TAR_TESTING_COLUMNS_PATH = os.path.join(SOURCE_DATA_DIR, "tar_test_cols")


if not ("train_cols" in os.listdir(SOURCE_DATA_DIR)):
    os.mkdir(TRAINING_COLUMNS_PATH)

if not ("tar_train_cols" in os.listdir(SOURCE_DATA_DIR)):
    os.mkdir(TAR_TRAINING_COLUMNS_PATH)

if not ("test_cols" in os.listdir(SOURCE_DATA_DIR)):
    os.mkdir(TESTING_COLUMNS_PATH)
    
if not ("tar_test_cols" in os.listdir(SOURCE_DATA_DIR)):
    os.mkdir(TAR_TESTING_COLUMNS_PATH)


with open(TRAINING_DATA_PATH,"r") as rf:
    colnames_line = rf.readline()
    COLUMN_NAMES  = colnames_line.strip().split(",")




try:
    from local_settings import *
except:
    pass 


# local_settings.py
# SOURCE_DATA_DIR = "/home/c3h3/c3h3works/Kaggles/GithubRepos/DisplayAdvertisingChallenge/data/"
# MAX_N_DATA_COLUMN_DIVIDERS = 5

