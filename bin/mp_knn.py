
import multiprocessing as mp

from hunkaggle.criteo import tools
from hunkaggle.criteo.settings import TRAINING_COLUMN_NAMES, TESTING_COLUMN_NAMES
import blz, os
import numpy as np
from hunkaggle.criteo.settings import MODELS_PATH, SUBMITS_PATH
from hunkaggle.criteo.settings import SOURCE_DATA_DIR
from hunkaggle.criteo.models import *

try:
    import cPickle as pickle
except:
    import pickle

import datetime
tic = datetime.datetime.now()
print "tic = ",tic
print "loading data ... "
    

select_slice = slice(0,None,20)

X = get_rf_v1_as_training_X((select_slice,1))

testX = get_rf_v1_as_testing_X()

y = get_labels_barray(select_slice)

print "finished loading data ... ", (datetime.datetime.now() - tic).seconds
print "start predicting ... "


result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)
    
def apply_async_with_callback(start=0,end=100):
    pool = mp.Pool()
    for ii in range(start,end):
        pool.apply_async(knn_predict, args = (ii,testX[ii,:],X,y), callback = log_result)
    pool.close()
    pool.join()

    print "finished predicting ... ", (datetime.datetime.now() - tic).seconds
    print "start pickling ... "


    with open("results_%s_%s.pickle" %(start,end),"wb") as wf:
        pickle.dump(result_list,wf)
    
    print "finished ... ", (datetime.datetime.now() - tic).seconds
    

if __name__ == '__main__':
    from hunkaggle.criteo.tools import get_separation_pairs
    
    pairs = get_separation_pairs(y.shape[0],100000)
    
    for one_pair in pairs:
        print "one_pair = ",one_pair
        apply_async_with_callback(*one_pair)
    
    
    print "all finished ... ", (datetime.datetime.now() - tic).seconds    
    