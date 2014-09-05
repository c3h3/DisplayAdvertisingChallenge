from settings import TRAINING_DATA_PATH, TESTING_DATA_PATH, TRAINING_COLUMNS_PATH, TESTING_COLUMNS_PATH, MAX_N_DATA_COLUMN_DIVIDERS
import numpy as np
from multiprocessing import Pool 
from column_analysis_tooles import *

colnames = get_colnames(TESTING_DATA_PATH)
dividing_jobs = [(colname, TESTING_DATA_PATH, TESTING_COLUMNS_PATH) for colname in colnames.to_list()]

print "dividing_jobs = ", dividing_jobs

p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
p.map(pickle_one_data_column, dividing_jobs)

p.close()
p.join()

