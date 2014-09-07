
from multiprocessing import Pool   

# dividing_jobs = [xx for xx in COLUMN_NAMES if xx not in list_all_data_columns()]
# 
# print "dividing_jobs = ", dividing_jobs
# 
# p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
# p.map(pickle_one_data_column, dividing_jobs)
# 
# p.close()
# p.join()

from hunkaggle.criteo.settings import TRAINING_COLUMN_NAMES
from hunkaggle.criteo.settings import MAX_N_DATA_COLUMN_DIVIDERS
from hunkaggle.criteo import tools

p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
p.map(tools.convert_column_to_blz, TRAINING_COLUMN_NAMES)
 
p.close()
p.join()


