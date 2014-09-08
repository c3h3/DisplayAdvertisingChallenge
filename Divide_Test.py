
# coding: utf-8

"""
settings.py
"""
import os

SOURCE_DATA_DIR = "/Users/DboyLiao/Documents/kaggle/DisplayAdvertisingChallenge/data/"
MAX_N_DATA_COLUMN_DIVIDERS = 5

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


"""
columns_analysis_tooles

Created on Aug 17, 2014
@author: c3h3
"""
from settings import TRAINING_DATA_PATH, TESTING_DATA_PATH, TRAINING_COLUMNS_PATH, TESTING_COLUMNS_PATH
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import os


"""
Class ColumnData. 
"""
class ColumnData(object):
    def __init__(self, colname, data_list):
        data_array = np.array(data_list)
        self.states_vec, self.states_pos_vec, self.index_vec = np.unique(data_array,return_inverse=True,return_index=True)
        self.name = colname
    
    def save_as_pickle_file(self, dir_path):
        output_file_path = os.path.join(dir_path, "%s.col" % self.name)
        with open(output_file_path, "wb") as wf:
            pickle.dump(self, wf)
    def __getitem__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise KeyError, "%s has no %s attribute." % (self.__dict__["name"], name)
    def __str__(self):
        msg = "States Vector: %s \n" % self.states_vec +         "States Position: %s \n" % self.states_pos_vec +         "Index Vector: %s \n" % self.index_vec
        return msg


"""
Helper functions.
"""
def get_colnames(data_path):
    """
    Input: a string, the path to the source data. (.csv)
    """
    with open(data_path, "r") as rf:
        colnames_line = rf.readline()
        colnames = colnames_line.strip().split(",")
    return np.array(colnames)

def get_idx_by_colname(colname, data_path):
    """
    Input: 
          colname: string, the name of one column.
          data_path: string, source data path. (.csv)
    Output: integer, the index of colname which specifies the location of that column in the source data.
    """
    colnames = get_colnames(data_path)
    return np.where(colnames == colname)[0][0]


def get_one_column_data_list(colname, data_path):
    """
    Input:
          colname: string, the name of one column.
          data_path: string, source data path. (.csv)
    Output: list, a list which contains all data of the column with name which matchs with colname.
    """
    col_idx = get_idx_by_colname(colname, data_path)
    assert isinstance(col_idx, int)
    assert col_idx >= 0
    colnames = get_colnames(data_path)
    assert col_idx < len(colnames)
    
    with open(data_path, "r") as rf:
        data_list = []
        one_line_str = "temp_str"

        while one_line_str != "":
            one_line_str = rf.readline()
            one_line_str_data = one_line_str.strip().split(",")
            if len(one_line_str_data) == len(colnames):
                data_list.append(one_line_str.strip().split(",")[col_idx])
            
    return data_list


def get_one_data_column(colname, data_path):
    """
    Wrapper function of get_one_column_data_list(). 
    get_one_data_column() will return an ColumnData object rather than a list.
    """
    data_list = get_one_column_data_list(colname, data_path)
    return ColumnData(colname, data_list)

def pickle_one_data_column(name_path_tuple):
    colname = name_path_tuple[0]
    data_path = name_path_tuple[1]
    col_path = name_path_tuple[2]
    temp = get_one_data_column(colname, data_path)
    temp.save_as_pickle_file(col_path)

def list_all_data_columns(data_path): 
    return [xx.split(".")[0] for xx in os.listdir(data_path) if xx.endswith(".col")]


colnames = get_colnames(TESTING_DATA_PATH)
for colname in colnames:
    print "[Columnize] Processing " + colname + ".col"
    temp = get_one_data_column(colname, TESTING_DATA_PATH)
    temp.save_as_pickle_file(TESTING_COLUMNS_PATH)
    del temp
