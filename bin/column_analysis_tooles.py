'''
Created on Aug 17, 2014

@author: c3h3
'''

from settings import COLUMN_NAMES, TRAINING_DATA_PATH, TRAINING_COLUMNS_PATH
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import os

COLNAMES = np.array(COLUMN_NAMES)

def get_idx_by_colname(colname):
    return np.where(COLNAMES == colname)[0][0]


def get_one_column_data_list(col_idx):
    assert isinstance(col_idx, int)
    assert col_idx > 0
    assert col_idx < len(COLNAMES)
    
    with open(TRAINING_DATA_PATH,"r") as rf:
        colnames_line = rf.readline()
        colnames = colnames_line.strip().split(",")
        
        data_list = []
        one_line_str = "temp_str"

        while one_line_str != "":
            one_line_str = rf.readline()
            one_line_str_data = one_line_str.strip().split(",")
            if len(one_line_str_data) == len(colnames):
                data_list.append(one_line_str.strip().split(",")[col_idx])
            
    return data_list
    

class ColumnData(object):
    def __init__(self, colname, data_list):
        data_array = np.array(data_list)
        self.states_vec, self.states_pos_vec, self.index_vec = np.unique(data_array,return_inverse=True,return_index=True)
        self.name = colname
    
    def save_as_pickle_file(self, dir_path):
        output_file_path = os.path.join(dir_path, "%s.col" % self.name)
        with open(output_file_path, "wb") as wf:
            pickle.dump(self, wf)


def get_one_data_column(colname):
    col_idx = get_idx_by_colname(colname)
    assert isinstance(col_idx, int)
    assert col_idx > 0
    assert col_idx < len(COLNAMES)
    
    with open(TRAINING_DATA_PATH,"r") as rf:
        colnames_line = rf.readline()
        colnames = colnames_line.strip().split(",")
        
        data_list = []
        one_line_str = "temp_str"

        while one_line_str != "":
            one_line_str = rf.readline()
            one_line_str_data = one_line_str.strip().split(",")
            if len(one_line_str_data) == len(colnames):
                data_list.append(one_line_str.strip().split(",")[col_idx])
            
    return ColumnData(colname,data_list)


def pickle_one_data_column(colname):
    col_idx = get_idx_by_colname(colname)
    assert isinstance(col_idx, int)
    assert col_idx > 0
    assert col_idx < len(COLNAMES)
    
    with open(TRAINING_DATA_PATH,"r") as rf:
        colnames_line = rf.readline()
        colnames = colnames_line.strip().split(",")
        
        data_list = []
        one_line_str = "temp_str"

        while one_line_str != "":
            one_line_str = rf.readline()
            one_line_str_data = one_line_str.strip().split(",")
            if len(one_line_str_data) == len(colnames):
                data_list.append(one_line_str.strip().split(",")[col_idx])
            
    ColumnData(colname,data_list).save_as_pickle_file(TRAINING_COLUMNS_PATH)


def list_all_data_columns(): 
    return [xx.split(".")[0] for xx in os.listdir(TRAINING_COLUMNS_PATH) if xx.endswith(".col")]
